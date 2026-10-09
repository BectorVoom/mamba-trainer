"""Export structure-only molecules for the fingerprint-to-structure path.

Reads SMILES from an SDF property field (default: the pinned ChEBI 3-star
export, `> <SMILES>`) and writes them in the `export_msgym_spectral.py`
schema with an empty `spectra` list per molecule, plus the fingerprint
sidecar. `examples/ms2_spectral_completion.rs --extra-train` trains on them
with the true fingerprint and no spectral evidence.

`--candidates-json` reads instead a (possibly truncated) JSON object mapping a
query SMILES to its list of candidate SMILES, as in the pinned MassSpecGym
retrieval-candidate prefix: every complete entry's candidates are pooled,
deduplicated as strings, shuffled with `--seed`, and converted until
`--limit` molecules are kept. These are database structures, not labels; the
`--exclude` rule below still removes every evaluation molecule.

Every molecule whose InChIKey first block is a key of an `--exclude` export
(pass the validation and test exports, and the training export to avoid
duplicates) is dropped and counted, as are molecules outside the atom
vocabulary or above `--max-atoms` heavy atoms.

    PYTHONPATH=tools/ms2 python tools/ms2/export_molecules_fp.py --name chebi \
        --exclude data/ms2/specgen/msgym_validation.json \
        --exclude data/ms2/specgen/msgym_test.json \
        --exclude data/ms2/specgen/msgym_train.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")


def sdf_field(path: Path, field: str):
    """Yield the first line of every `> <field>` property block."""
    opener = gzip.open if path.suffix == ".gz" else open
    tag = f"> <{field}>"
    with opener(path, "rt", errors="replace") as fh:
        take = False
        for line in fh:
            if take:
                yield line.strip()
                take = False
            elif line.startswith(tag):
                take = True


def candidate_smiles(path: Path):
    """Candidate SMILES of every complete entry of a truncated JSON object."""
    text = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    out: set[str] = set()
    while True:
        while position < len(text) and text[position] in " \n\r\t,":
            position += 1
        if position >= len(text) or text[position] == "}":
            break
        try:
            _, position = decoder.raw_decode(text, position)
            position = text.index(":", position) + 1
            while text[position] in " \n\r\t":
                position += 1
            values, position = decoder.raw_decode(text, position)
        except (ValueError, IndexError):
            break  # the trailing partial entry
        out.update(v for v in values if isinstance(v, str))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sdf", type=Path, default=Path("data/pinned/chebi_3_stars.sdf.gz"))
    ap.add_argument("--field", default="SMILES")
    ap.add_argument("--candidates-json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many kept molecules (0: no limit)")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("data/ms2/specgen"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-atoms", type=int, default=32)
    ap.add_argument("--min-atoms", type=int, default=4)
    args = ap.parse_args()

    excluded: set[str] = set()
    for path in args.exclude:
        excluded |= {m["key"] for m in json.loads(path.read_text())["molecules"]}

    skipped: Counter = Counter()
    seen: set[str] = set()
    molecules, bits, keys = [], [], []
    if args.candidates_json is not None:
        source = candidate_smiles(args.candidates_json)
        random.Random(args.seed).shuffle(source)
        source_name = args.candidates_json.name
    else:
        source = sdf_field(args.sdf, args.field)
        source_name = args.sdf.name
    pooled = len(source) if isinstance(source, list) else None
    for smiles in source:
        if args.limit and len(molecules) >= args.limit:
            break
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            skipped["unparsable"] += 1
            continue
        try:
            kek = ref.kekulized(smiles)
        except Exception:
            skipped["kekulize_failed"] += 1
            continue
        reasons = ref.classify(kek)
        if reasons:
            skipped[reasons[0]] += 1
            continue
        atoms, bonds = ref.graph_of(kek)
        if not args.min_atoms <= len(atoms) <= args.max_atoms:
            skipped["atom_count"] += 1
            continue
        block = Chem.MolToInchiKey(mol)[:14]
        if len(block) != 14:
            skipped["no_inchikey"] += 1
            continue
        if block in excluded:
            skipped["excluded_key"] += 1
            continue
        if block in seen:
            skipped["duplicate_key"] += 1
            continue
        seen.add(block)
        index = len(molecules)
        molecules.append({"key": block, "smiles": smiles, "identity_group": 10_000_000 + index,
                          "fold_identity": 2, "atoms": atoms, "bonds": bonds, "spectra": []})
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
        bits.append(sorted(int(i) for i in fp.GetOnBits()))
        keys.append(f"{block}|{10_000_000 + index}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export = {"schema_version": 1, "chemistry": "ms2-chem-v0.1", "rdkit": rdkit.__version__,
              "source": source_name, "seed": args.seed, "n_raw": 0, "spectra_per_molecule": 0,
              "skipped_spectra": {}, "subset": "structures_only",
              "excluded_exports": [p.name for p in args.exclude], "molecules": molecules}
    path = args.out_dir / f"{args.name}_structures.json"
    path.write_text(json.dumps(export, separators=(",", ":")))
    (args.out_dir / f"{args.name}_structures_fp.json").write_text(json.dumps({
        "fingerprint": "morgan4096", "rdkit": rdkit.__version__, "export": path.name,
        "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_molecules": len(molecules), "keys_by_molecule": keys, "bits_by_molecule": bits,
    }, separators=(",", ":")))
    print(json.dumps({"molecules": len(molecules), "pooled_candidate_smiles": pooled,
                      "excluded_keys_loaded": len(excluded),
                      "skipped": dict(skipped)}, indent=1))


if __name__ == "__main__":
    main()
