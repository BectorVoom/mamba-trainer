"""Tests for `export_fingerprints.py` (task K10 / P7.3).

Runnable with:

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy \
        python tools/ms2/test_export_fingerprints.py

Builds a tiny synthetic export (three molecules, no real data) and checks:
order preservation, bit lists recomputed independently for a simple molecule,
determinism (two runs byte-equal) and provenance. Plain asserts only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "export_fingerprints.py"

SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]
KEYS = ["MOL0001", "MOL0002", "MOL0003"]


def write_synth_export(path: Path) -> None:
    molecules = [
        {"key": key, "smiles": smi, "identity_group": i,
         "fold_identity": 2, "atoms": [2], "bonds": [], "spectra": []}
        for i, (key, smi) in enumerate(zip(KEYS, SMILES))
    ]
    doc = {"schema_version": 1, "chemistry": "ms2-chem-v0.1",
           "rdkit": "test", "source": "synthetic", "seed": 7,
           "n_raw": 512, "spectra_per_molecule": 0,
           "spectrum_sampling": "synthetic",
           "skipped_spectra": {}, "subset": "train",
           "molecules": molecules}
    path.write_text(json.dumps(doc, separators=(",", ":")))


def run_exporter(export: Path, out: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    r = subprocess.run([sys.executable, str(TOOL),
                        "--export", str(export), "--out", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"exporter failed: {r.stderr[-3000:]}"


def independent_bits(smiles: str) -> list[int]:
    """Recomputed without importing the tool: the exact documented call."""
    from rdkit import Chem, RDLogger  # noqa: E402
    from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return sorted(gen.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import rdkit  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        export = tmp / "synth.json"
        write_synth_export(export)

        out_a = tmp / "a_fp.json"
        out_b = tmp / "b_fp.json"
        run_exporter(export, out_a)
        run_exporter(export, out_b)

        # 1. determinism: two runs are byte-equal.
        assert out_a.read_bytes() == out_b.read_bytes(), "not deterministic"
        print("PASS determinism (two runs byte-equal)")

        side = json.loads(out_a.read_text())
        assert side["schema_version"] == 1, side["schema_version"]
        assert side["fingerprint"] == "morgan-r2-1024", side["fingerprint"]
        assert side["bits"] == 1024, side["bits"]
        assert side["radius"] == 2, side["radius"]
        assert side["generator"] == ("rdFingerprintGenerator.GetMorganGenerator"
                                     "(radius=2, fpSize=1024)"), side["generator"]
        assert side["rdkit"] == rdkit.__version__, (side["rdkit"], rdkit.__version__)
        assert side["source_export"] == export.name, side["source_export"]
        assert side["source_sha256"] == sha256_of(export), "source hash mismatch"
        print("PASS provenance (id, call, rdkit version, source name and sha256)")

        # 2. order preserved: the sidecar follows the export molecule order.
        assert [m["key"] for m in side["molecules"]] == KEYS, "order not preserved"
        assert [m["smiles"] for m in side["molecules"]] == SMILES, "smiles mismatch"
        print("PASS order preserved (keys and SMILES follow the export)")

        # 3. known bits: every molecule's list matches an independent
        # recomputation; lists are sorted and in range.
        for m, smi in zip(side["molecules"], SMILES):
            want = independent_bits(smi)
            assert m["bits"] == want, (m["key"], m["bits"], want)
            assert m["bits"] == sorted(m["bits"]), "bits not sorted"
            assert all(0 <= b < 1024 for b in m["bits"]), "bit out of range"
            assert len(m["bits"]) > 0, f"empty fingerprint for {smi}"
        print("PASS known bits (independent recomputation for every molecule)")

        # 4. the three molecules differ (the sidecar is per-molecule, not constant).
        bitsets = [tuple(m["bits"]) for m in side["molecules"]]
        assert len(set(bitsets)) == 3, "fingerprints are not molecule-specific"
        print("PASS molecule-specific (three distinct fingerprints)")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
