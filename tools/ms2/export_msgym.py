"""Export in-domain MassSpecGym spectra and parent graphs for the Rust MS2 pipeline.

The MassSpecGym twin of `export_casmi.py`: it writes exactly the same JSON
schema (one file per subset with `schema_version`, `chemistry`, `rdkit`,
`source`, `seed`, `n_raw`, `spectra_per_molecule`, `spectrum_sampling`,
`skipped_spectra`, `subset`, `molecules`, where each molecule holds `key`,
`identity_group`, `fold_identity`, `atoms`, `bonds` and `spectra`, and each
spectrum holds the integer `SpectrumBatch` fields of docs/MS2_CONTRACTS.md
section 3.1), read from `data/pinned/MassSpecGym1.5.tsv` instead of the CASMI
parquet data. Labels are not exported: the Rust reference builds them from
these inputs, so there is one implementation of the recipe in the training path.

Differences from the CASMI tool (all intentional):

- subsets come from the TSV `fold` column (`train` -> train, `val` ->
  validation); rows whose fold is `test` are skipped at parse time and never
  exported. `fold_identity` is 2 for train, 1 for val (the CASMI numbering);
  `identity_group` is the index in the sorted list of all non-test InChIKey14
  values;
- polarity is derived from the adduct's last character (there is no
  `ionization_mode` column), so there is no `polarity_conflict` skip reason;
- each row carries a single `collision_energy` value (known=1, count=1 when
  present and finite; 0.0/0/0 when empty);
- provenance adds `tsv_sha256`, `folds` and `instrument_filter`. The Rust
  loader (`src/models/ms2/dataset.rs` `ExportFile`) has no
  `deny_unknown_fields`, so serde ignores these extra top-level keys;
- `--scaffold-holdout` additionally writes `<name>_scaffold_validation.json`.

The output is derived data and stays outside the repository (`--out-dir`, by
default `data/ms2`).

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy --with pyarrow \
        python tools/ms2/export_msgym.py --name smoke \
        --train-molecules 64 --validation-molecules 32

The versioned molecule-disjoint split `msgym-split-v1` (architecture §4.3)
divides the train fold into `fit` (80%) and `rank` (20%) and the val fold
into `calibration` (50%) and `report` (50%) by a hash of the InChIKey first
block; the test fold is never read:

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy --with pyarrow \
        python tools/ms2/export_msgym.py --name v1 --split msgym-split-v1 \
        --fit-molecules 30000 --rank-molecules 30000 \
        --calibration-molecules 30000 --report-molecules 30000 \
        --fit-only-artifacts-report /tmp/msgym_split_v1_audit.json

With `--split`, one file per requested part is written (`<name>_fit.json`,
`<name>_rank.json`, `<name>_calibration.json`, `<name>_report.json`;
0 = do not export that part). Without `--split` the behaviour and output
bytes are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

import ms2_reference as ref
from audit_casmi import instrument_class

RDLogger.DisableLog("rdApp.*")

ADDUCT_ID = {name: i for i, (name, _, _) in ref.ADDUCTS.items()}
INSTRUMENT_ID = {"timstof": 1, "orbitrap": 2, "qtof": 3, "other": 4}
N_RAW = 512
FOLD_OF = {"train": "train", "val": "validation"}
FOLD_IDENTITY = {"train": 2, "validation": 1}

SPLIT_MSGYM_V1 = "msgym-split-v1"
SPLIT_PARTS = ("fit", "rank", "calibration", "report")
SPLIT_RULE = ("sha256(inchikey14) first 8 bytes big-endian h; "
              "train fold: rank iff h%5==0 else fit; "
              "val fold: calibration iff h%2==0 else report")
PART_FOLD_IDENTITY = {"fit": 2, "rank": 2, "calibration": 1, "report": 1}


def normalize_identity(raw: str | None) -> str | None:
    """Normalised molecule-identity block of an InChIKey field.

    The identity is the text before the first ``-`` (so a full
    27-character InChIKey and its 14-character first block share an
    identity), upper-cased. Valid only when the block is exactly 14
    letters A-Z; otherwise None (the row is skipped as `invalid_inchikey`).
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
    same part. `h` is the first 8 bytes of `sha256(block)` as a big-endian
    integer; in the train fold `rank` when `h % 5 == 0` else `fit`, in the
    val fold `calibration` when `h % 2 == 0` else `report`. The hash is of
    the normalised block, so all SMILES sharing a block land in the same part.
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


def murcko_scaffold_smiles(mol) -> str:
    """Bemis-Murcko scaffold SMILES; '' for acyclic molecules (no scaffold)."""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def request_skip_reason(adduct: str, precursor_mz: float | None, n_peaks: int) -> str | None:
    """The request-level domain checks of an exported spectrum (contracts §4.3, §4.6)."""
    if adduct not in ADDUCT_ID:
        return "adduct"
    if precursor_mz is None or not math.isfinite(precursor_mz) \
            or not 50.0 <= precursor_mz <= 2000.0:
        return "precursor_range"
    if n_peaks == 0:
        return "empty"
    return None


def parse_floats(text: str) -> list[float] | None:
    """Comma-separated decimals as floats, or None when unparsable."""
    try:
        return [float(v) for v in text.split(",") if v.strip() != ""]
    except ValueError:
        return None


U32_MAX = 4294967295


def peaks_invalid(mz: list[float] | None, it: list[float] | None) -> bool:
    """True when a parsed peak list must not be exported.

    A spectrum is eligible only if every m/z is finite, > 0 and
    ``round(mz * ref.SCALE)`` fits an unsigned 32-bit integer, every
    intensity is finite and >= 0, both lists have equal non-zero length,
    and at least one intensity is > 0. Zero-length lists are valid here;
    the caller maps those to the existing ``empty`` skip reason.
    """
    if mz is None or it is None:
        return True
    if len(mz) != len(it):
        return True
    if len(mz) == 0:
        return False
    for m in mz:
        if not math.isfinite(m) or not m > 0:
            return True
        if round(m * ref.SCALE) > U32_MAX:
            return True
    seen_positive = False
    for v in it:
        if not math.isfinite(v) or not v >= 0:
            return True
        if v > 0:
            seen_positive = True
    return not seen_positive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path("data/pinned/MassSpecGym1.5.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/ms2"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-molecules", type=int, default=2000)
    ap.add_argument("--validation-molecules", type=int, default=400)
    ap.add_argument("--spectra-per-molecule", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--instrument", choices=["orbitrap", "qtof"], default=None,
                    help="keep only spectra of this instrument class")
    ap.add_argument("--exclude-instrument", choices=["orbitrap", "qtof"], default=None,
                    help="drop spectra of this instrument class")
    ap.add_argument("--scaffold-holdout", action="store_true",
                    help="also write <name>_scaffold_validation.json")
    ap.add_argument("--split", choices=[SPLIT_MSGYM_V1], default=None,
                    help="versioned molecule-disjoint split; with it, "
                    "--fit/--rank/--calibration/--report-molecules select the "
                    "parts and --train/--validation-molecules are ignored")
    ap.add_argument("--fit-molecules", type=int, default=0)
    ap.add_argument("--rank-molecules", type=int, default=0)
    ap.add_argument("--calibration-molecules", type=int, default=0)
    ap.add_argument("--report-molecules", type=int, default=0)
    ap.add_argument("--fit-only-artifacts-report", type=Path, default=None,
                    help="write a JSON of per-part molecule/formula counts and "
                    "the share of each non-fit part's formulas seen in fit "
                    "(aggregates only)")
    args = ap.parse_args()
    if args.split is not None and args.scaffold_holdout:
        ap.error("--scaffold-holdout cannot be combined with --split")
    if args.instrument and args.exclude_instrument:
        ap.error("--instrument and --exclude-instrument are mutually exclusive")
    if args.instrument:
        instrument_filter = args.instrument
    elif args.exclude_instrument:
        instrument_filter = f"not:{args.exclude_instrument}"
    else:
        instrument_filter = "none"

    def instrument_dropped(cls: str) -> bool:
        if args.instrument:
            return cls != args.instrument
        if args.exclude_instrument:
            return cls == args.exclude_instrument
        return False

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_sha = sha256_of(args.tsv)
    rng = np.random.default_rng(args.seed)

    with open(args.tsv, "r") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    col = {name: i for i, name in enumerate(header)}
    for needed in ("mzs", "intensities", "smiles", "inchikey", "precursor_mz",
                   "adduct", "instrument_type", "collision_energy", "fold"):
        if needed not in col:
            raise SystemExit(f"{args.tsv} has no {needed!r} column")

    # Pass 1: the distinct non-test SMILES. Test rows are skipped at parse time.
    # Molecule identity is the NORMALISED first InChIKey block
    # (`normalize_identity`): rows with an invalid block are skipped and
    # counted as `invalid_inchikey`. A SMILES seen with more than one
    # non-test fold is a fold conflict and is excluded from sampling
    # entirely, as is every SMILES of an identity block seen in more than
    # one non-test fold (cross-fold same-block variants share no part).
    smiles_fold: dict[str, str] = {}
    smiles_key: dict[str, str] = {}
    smiles_folds: dict[str, set[str]] = {}
    smiles_blocks: dict[str, set[str]] = {}
    identity_folds: dict[str, set[str]] = {}
    identity_smiles: dict[str, set[str]] = {}
    invalid_inchikey = 0
    with open(args.tsv, "r") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            fold = parts[col["fold"]]
            if fold == "test" or fold not in FOLD_OF:
                continue
            block = normalize_identity(parts[col["inchikey"]])
            if block is None:
                invalid_inchikey += 1
                continue
            smi = parts[col["smiles"]]
            smiles_folds.setdefault(smi, set()).add(fold)
            smiles_blocks.setdefault(smi, set()).add(block)
            identity_folds.setdefault(block, set()).add(fold)
            identity_smiles.setdefault(block, set()).add(smi)
            if smi not in smiles_fold:
                smiles_fold[smi] = fold
                smiles_key[smi] = block
    # A SMILES seen with more than one non-test fold is a SMILES-level
    # conflict; an identity block seen in more than one non-test fold is an
    # identity-level conflict and every SMILES of that block is excluded.
    # (A SMILES with two blocks in the SAME fold keeps its first block;
    # the pinned data has 9 such in-train cases and no cross-fold case.)
    fold_conflicts = {smi for smi, folds in smiles_folds.items() if len(folds) > 1}
    fold_conflict_identities = {b for b, folds in identity_folds.items() if len(folds) > 1}
    conflict_block_smiles: set[str] = set()
    for b in fold_conflict_identities:
        conflict_block_smiles |= identity_smiles[b]
    excluded = fold_conflicts | conflict_block_smiles
    for smi in excluded:
        smiles_fold.pop(smi, None)
        smiles_key.pop(smi, None)
    print(f"fold conflicts: {len(fold_conflicts)} molecules excluded "
          f"(seen in more than one non-test fold); "
          f"{len(fold_conflict_identities)} identity blocks cross-fold, "
          f"{invalid_inchikey} rows with invalid InChIKey")
    # A molecule is one distinct SMILES string (as in the CASMI tool).
    key_index = {k: i for i, k in enumerate(sorted(set(smiles_key.values())))}

    # In-domain graphs for every distinct non-test SMILES (peak lists are not held).
    info: dict[str, dict] = {}
    for smi in sorted(smiles_fold):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            continue
        if ref.classify(mol):
            continue
        atoms, bonds = ref.graph_of(mol)
        if args.split is None:
            subset = FOLD_OF[smiles_fold[smi]]
            fold_identity = FOLD_IDENTITY[subset]
        else:
            subset = part_of(smiles_key[smi], smiles_fold[smi])
            fold_identity = PART_FOLD_IDENTITY[subset]
        info[smi] = {"subset": subset, "key": smiles_key[smi], "smiles": smi,
                     "identity_group": key_index[smiles_key[smi]],
                     "fold_identity": fold_identity,
                     "atoms": atoms, "bonds": bonds, "spectra": []}

    # Sample molecules per subset, in-domain only, in a seeded order.
    if args.split is None:
        wanted = {"train": args.train_molecules, "validation": args.validation_molecules}
    else:
        wanted = {"fit": args.fit_molecules, "rank": args.rank_molecules,
                  "calibration": args.calibration_molecules, "report": args.report_molecules}
    chosen: dict[str, dict] = {}
    have = Counter()
    for smi in rng.permutation(sorted(smiles_fold)):
        if all(have[s] >= n for s, n in wanted.items()):
            break
        if args.split is None:
            subset = FOLD_OF[smiles_fold[smi]]
        else:
            subset = part_of(smiles_key[smi], smiles_fold[smi])
        if have[subset] >= wanted[subset]:
            continue
        if smi not in info:
            continue
        have[subset] += 1
        chosen[smi] = dict(info[smi])
        chosen[smi]["spectra"] = []

    # Pass 2: each chosen molecule's eligible rows (metadata only; no peak lists held).
    skipped = Counter()
    candidates: dict[str, list[int]] = {k: [] for k in chosen}
    with open(args.tsv, "r") as fh:
        fh.readline()
        for row, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if parts[col["fold"]] == "test" or parts[col["fold"]] not in FOLD_OF:
                continue
            smi = parts[col["smiles"]]
            if smi not in chosen:
                continue
            if parts[col["fold"]] != smiles_fold[smi]:
                continue
            # Rows with an invalid identity block can never be sampled.
            # (A row whose valid block differs from the molecule's first
            # block keeps the molecule's part; the pinned data has such
            # in-fold cases and old outputs are preserved.)
            if normalize_identity(parts[col["inchikey"]]) is None:
                skipped["invalid_inchikey"] += 1
                continue
            mz = parse_floats(parts[col["mzs"]])
            it = parse_floats(parts[col["intensities"]])
            if mz is None or it is None or len(mz) != len(it):
                skipped["invalid_peaks"] += 1
                continue
            n_peaks = len(mz)
            try:
                prec = float(parts[col["precursor_mz"]])
            except ValueError:
                prec = None
            reason = request_skip_reason(parts[col["adduct"]], prec, n_peaks)
            if reason:
                skipped[reason] += 1
                continue
            if peaks_invalid(mz, it):
                skipped["invalid_peaks"] += 1
                continue
            if instrument_dropped(instrument_class(parts[col["instrument_type"]])):
                skipped["instrument_filter"] += 1
                continue
            candidates[smi].append(row)
    # Each molecule's spectra are drawn uniformly (seeded) from its eligible rows.
    selected = set()
    for smi in sorted(candidates):
        rows = candidates[smi]
        take = min(args.spectra_per_molecule, len(rows))
        if take:
            selected.update(int(r) for r in
                            (rows[j] for j in rng.choice(len(rows), size=take, replace=False)))

    # Pass 3: read the selected rows' peak lists.
    with open(args.tsv, "r") as fh:
        fh.readline()
        for row, line in enumerate(fh):
            if row not in selected:
                continue
            parts = line.rstrip("\n").split("\t")
            assert parts[col["fold"]] != "test", f"test row {row} selected for export"
            smi = parts[col["smiles"]]
            assert parts[col["fold"]] == smiles_fold[smi], f"fold mismatch on row {row}"
            assert normalize_identity(parts[col["inchikey"]]) is not None, \
                f"invalid identity on row {row}"
            mol = chosen[smi]
            adduct = parts[col["adduct"]]
            mz = parse_floats(parts[col["mzs"]])
            it = parse_floats(parts[col["intensities"]])
            assert mz is not None and it is not None and len(mz) == len(it) and mz
            assert not peaks_invalid(mz, it), f"invalid peaks on row {row}"
            prec = float(parts[col["precursor_mz"]])
            assert request_skip_reason(adduct, prec, len(mz)) is None
            raw = len(mz)
            order = sorted(range(raw), key=lambda j: (-it[j], j))[:N_RAW]
            order.sort()
            ce_text = parts[col["collision_energy"]].strip()
            try:
                ce = float(ce_text) if ce_text else None
            except ValueError:
                ce = None
            known = 1 if ce is not None and math.isfinite(ce) else 0
            mol["spectra"].append({
                "row": row, "spectrum_id": row,
                "adduct": ADDUCT_ID[adduct], "polarity": 1 if adduct.endswith("+") else -1,
                "precursor_mz_udalton": round(prec * ref.SCALE),
                "precursor_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals([prec])),
                "raw_peak_count": raw, "peak_id": order,
                "mz_udalton": [round(mz[j] * ref.SCALE) for j in order],
                "intensity": [float(it[j]) for j in order],
                "mz_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals(mz)),
                "collision_energy_ev": float(ce) if known else 0.0,
                "collision_energy_known": known,
                "energy_count": 1 if known else 0,
                "instrument_class": INSTRUMENT_ID[instrument_class(parts[col["instrument_type"]])],
            })

    provenance = {
        "schema_version": 1, "chemistry": "ms2-chem-v0.1", "rdkit": rdkit.__version__,
        "source": "MassSpecGym1.5.tsv", "tsv_sha256": tsv_sha,
        "folds": {"train": "train", "validation": "val"},
        "instrument_filter": instrument_filter,
        "seed": args.seed, "n_raw": N_RAW,
        "spectra_per_molecule": args.spectra_per_molecule, "spectrum_sampling": "uniform-in-domain-v2",
        "skipped_spectra": dict(skipped),
        "fold_conflict_molecules": len(fold_conflicts),
        "fold_conflict_identities": len(fold_conflict_identities),
        "invalid_inchikey": invalid_inchikey,
    }
    if args.split is not None:
        provenance["split"] = args.split
        provenance["split_rule"] = SPLIT_RULE
    exported: dict[str, list] = {}
    for subset in wanted:
        if wanted[subset] == 0:
            continue
        molecules = [{k: v for k, v in m.items() if k != "subset"}
                     for m in chosen.values() if m["subset"] == subset and m["spectra"]]
        molecules.sort(key=lambda m: (m["key"], m["smiles"]))
        exported[subset] = molecules
        payload = json.dumps({**provenance, "subset": subset, "molecules": molecules},
                             separators=(",", ":"))
        path = out_dir / f"{args.name}_{subset}.json"
        path.write_text(payload)
        print(f"{path}: {len(molecules)} molecules, {sum(len(m['spectra']) for m in molecules)} spectra, "
              f"{len(payload) / 1e6:.1f} MB, sha256 {hashlib.sha256(payload.encode()).hexdigest()[:16]}")

    if args.fit_only_artifacts_report is not None:
        split_name = args.split if args.split is not None else SPLIT_MSGYM_V1
        part_smiles: dict[str, list[str]] = {p: [] for p in SPLIT_PARTS}
        for smi, entry in info.items():
            part_smiles[part_of(entry["key"], smiles_fold[smi])].append(smi)
        smiles_formula: dict[str, tuple] = {}
        for smis in part_smiles.values():
            for smi in smis:
                counts = ref.composition(info[smi]["atoms"])
                smiles_formula[smi] = tuple(counts[e] for e in ref.ELEMENT_ORDER)
        part_formulas = {p: {smiles_formula[s] for s in smis}
                         for p, smis in part_smiles.items()}
        fit_formulas = part_formulas["fit"]
        overlap = {}
        for p in ("rank", "calibration", "report"):
            smis = part_smiles[p]
            overlap[p] = (sum(1 for s in smis if smiles_formula[s] in fit_formulas)
                          / len(smis)) if smis else None
        audit = {
            "split": split_name,
            "source": "MassSpecGym1.5.tsv",
            "tsv_sha256": tsv_sha,
            "instrument_filter": instrument_filter,
            "note": ("aggregates only: in-domain distinct SMILES per part and "
                     "the share of each non-fit part's molecules whose formula "
                     "occurs among the fit formulas"),
            "parts": {p: {"molecules": len(part_smiles[p]),
                          "formulas": len(part_formulas[p])}
                      for p in SPLIT_PARTS},
            "formula_overlap_with_fit": overlap,
        }
        args.fit_only_artifacts_report.parent.mkdir(parents=True, exist_ok=True)
        args.fit_only_artifacts_report.write_text(json.dumps(audit, indent=1))
        print(json.dumps(audit, indent=1))

    if args.scaffold_holdout and wanted.get("validation", 0) != 0:
        # Scaffolds of ALL train-fold in-domain molecules, not only the sampled ones.
        train_scaffolds = set()
        for smi, e in info.items():
            if e["subset"] != "train":
                continue
            mol = Chem.MolFromSmiles(smi)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            scaffold = murcko_scaffold_smiles(mol)
            if scaffold:
                train_scaffolds.add(scaffold)
        kept = []
        for m in exported.get("validation", []):
            mol = Chem.MolFromSmiles(m["smiles"])
            Chem.Kekulize(mol, clearAromaticFlags=True)
            scaffold = murcko_scaffold_smiles(mol)
            if scaffold and scaffold not in train_scaffolds:
                kept.append(m)
        payload = json.dumps({**provenance, "subset": "scaffold_validation",
                              "scaffold_holdout": {"train_scaffolds": len(train_scaffolds),
                                                  "validation_molecules": len(exported.get("validation", [])),
                                                  "kept": len(kept)},
                              "molecules": kept}, separators=(",", ":"))
        path = out_dir / f"{args.name}_scaffold_validation.json"
        path.write_text(payload)
        print(f"{path}: {len(kept)} molecules, {sum(len(m['spectra']) for m in kept)} spectra, "
              f"{len(payload) / 1e6:.1f} MB, sha256 {hashlib.sha256(payload.encode()).hexdigest()[:16]}")
        print(f"scaffold holdout: {len(kept)} of {len(exported.get('validation', []))} validation "
              f"molecules kept ({len(train_scaffolds)} train scaffolds)")


if __name__ == "__main__":
    main()
