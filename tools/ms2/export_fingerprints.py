"""Whole-parent fingerprint sidecar for an MS2 export file (task K10 / P7.3).

Reads an export JSON (``tools/ms2/export_msgym.py`` schema) and writes a
sidecar ``<export stem>_fp.json`` with, per molecule in the SAME order as the
export, the molecule ``key``, its ``smiles`` and the Morgan fingerprint of the
WHOLE PARENT molecule: radius 2, 1024 bits, computed as

    rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

applied to ``Chem.MolFromSmiles(smiles)`` (no kekulization step; the molecule
comes straight from the export's SMILES), stored as the sorted list of on-bit
indices. Provenance (fingerprint id, the exact RDKit call, the RDKit version,
the source export file name and its sha256) is stored with the sidecar.

Deterministic: molecules keep export order, bit lists are sorted, JSON uses
compact separators.

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy \
        python tools/ms2/export_fingerprints.py --export data/ms2/overfit_train.json
    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy \
        python tools/ms2/export_fingerprints.py --export data/ms2/overfit_train.json \
        --out /tmp/overfit_train_fp.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FINGERPRINT_ID = "morgan-r2-1024"
FINGERPRINT_BITS = 1024
FINGERPRINT_RADIUS = 2
GENERATOR_CALL = (
    "rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)"
)
SCHEMA_VERSION = 1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_bits(smiles: str, generator) -> list[int]:
    """Sorted on-bit indices of the whole-parent Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    fp = generator.GetFingerprint(mol)
    return sorted(fp.GetOnBits())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True,
                    help="export JSON (tools/ms2/export_msgym.py schema)")
    ap.add_argument("--out", type=Path, default=None,
                    help="sidecar output path (default: <export stem>_fp.json "
                    "next to the export)")
    args = ap.parse_args()

    export_path: Path = args.export
    payload_bytes = export_path.read_bytes()
    source_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    doc = json.loads(payload_bytes.decode("utf-8"))
    molecules = doc.get("molecules", [])

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    out_molecules = []
    for mol in molecules:
        bits = fingerprint_bits(mol["smiles"], generator)
        assert bits == sorted(bits), "bit list must be sorted"
        assert all(0 <= b < FINGERPRINT_BITS for b in bits), "bit out of range"
        out_molecules.append(
            {"key": mol["key"], "smiles": mol["smiles"], "bits": bits}
        )

    out_path = args.out
    if out_path is None:
        out_path = export_path.parent / f"{export_path.stem}_fp.json"
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": FINGERPRINT_ID,
        "bits": FINGERPRINT_BITS,
        "radius": FINGERPRINT_RADIUS,
        "generator": GENERATOR_CALL,
        "rdkit": rdkit.__version__,
        "source_export": export_path.name,
        "source_sha256": source_sha256,
        "molecules": out_molecules,
    }
    text = json.dumps(sidecar, separators=(",", ":"))
    out_path.write_text(text)
    print(f"{out_path}: {len(out_molecules)} molecules, "
          f"{len(text) / 1e6:.1f} MB, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
