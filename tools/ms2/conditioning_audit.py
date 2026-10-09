"""Matched, evaluation-only fingerprint audit of a saved spectral decoder.

Run with the existing specgen venv. Never trains or overwrites checkpoints.
The supplied MIST predictions still use oracle formulas; this is a diagnostic,
not a competition performance estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stable(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"prepared input changed; use a new output directory: {path}")
        return
    path.write_text(text)


def write_stable_rows(path, rows):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    try:
        if path.exists():
            if sha256(path) != sha256(temporary):
                raise ValueError(f"prepared input changed; use a new output directory: {path}")
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(data, out):
    molecules = json.loads((data / "msgym_validation.json").read_text())["molecules"]
    seen, roster = set(), []
    for molecule in molecules:
        if (len(molecule["atoms"]) > 32 or
                len(molecule["bonds"]) - len(molecule["atoms"]) + 1 > 6 or
                molecule["key"] in seen):
            continue
        seen.add(molecule["key"])
        roster.append({"key": molecule["key"], "identity_group": molecule["identity_group"],
                       "spectrum_id": molecule["spectra"][0]["spectrum_id"]})
        if len(roster) == 300:
            break
    assert len(roster) == 300
    write_stable(out / "roster.json", json.dumps(roster, indent=2))
    predictions = read_rows(data / "mist_pred_val.clean.jsonl")
    by_id = {int("".join(c for c in row["id"] if c.isdigit())): row for row in predictions}
    assert all(r["spectrum_id"] in by_id for r in roster), "missing MIST query predictions"
    for threshold, name in [(0.1, "binary01"), (0.5, "binary05")]:
        write_stable_rows(out / f"{name}.jsonl", (
            {**row, "bits": [[bit, 1.0] for bit, p in row["bits"] if p >= threshold]}
            for row in predictions))
    # A seeded cyclic permutation is a derangement of molecule identities.
    # Match the complete probability vector, not independently shuffled bits.
    usable = [m for m in molecules if m["spectra"][0]["spectrum_id"] in by_id]
    unique = {m["key"]: m for m in usable}
    order = sorted(unique)
    random.Random(20261007).shuffle(order)
    donors = dict(zip(order, order[1:] + order[:1]))
    donor_by_spectrum = {}
    for key, molecule in unique.items():
        donor = unique[donors[key]]
        vector = by_id[donor["spectra"][0]["spectrum_id"]]["bits"]
        for spectrum in molecule["spectra"]:
            donor_by_spectrum[spectrum["spectrum_id"]] = vector
    def shuffled_rows():
        for row in predictions:
            sid = int("".join(c for c in row["id"] if c.isdigit()))
            # Unused spectra outside the export do not affect these queries.
            yield {**row, "bits": donor_by_spectrum.get(sid, row["bits"])}
    write_stable_rows(out / "shuffled.jsonl", shuffled_rows())
    write_stable_rows(out / "binary01_shuffled.jsonl", (
        {**row, "bits": [[bit, 1.0] for bit, p in row["bits"] if p >= 0.1]}
        for row in shuffled_rows()))
    write_stable(out / "shuffle_donors.json", json.dumps(donors, indent=2))
    return roster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data, out, binary = args.data.resolve(), args.out.resolve(), args.binary.resolve()
    out.mkdir(parents=True, exist_ok=True)
    roster = prepare(data, out)
    if args.prepare_only:
        return
    checkpoint = data / "progress/model2.ckpt.best"
    common = [str(binary), "--eval-only", "--progress-features", "--load", str(checkpoint),
              "--validation", str(data / "msgym_validation.json"),
              "--validation-fp", str(data / "msgym_validation_fp.json"),
              "--eval-subset", "300", "--queries", "300", "--beam", "256",
              "--trajectories", "256", "--returned", "1024", "--gen-batch", "8",
              "--out", str(out)]
    real = ["--fp-eval", "predicted", "--predictions", str(data / "mist_pred_val.clean.jsonl")]
    arms = {
        "A_exact": [],
        "B_removed": ["--eval-drop", "fingerprint"],
        "C_mist": real,
        "D_binary01": ["--fp-eval", "predicted", "--predictions", str(out / "binary01.jsonl")],
        "E_binary05": ["--fp-eval", "predicted", "--predictions", str(out / "binary05.jsonl")],
        "F_shuffled": ["--fp-eval", "predicted", "--predictions", str(out / "shuffled.jsonl")],
        "G_exact_oracle_formula": ["--formula", "oracle"],
        "H_mist_oracle_formula": real + ["--formula", "oracle"],
        "I_binary01_shuffled": ["--fp-eval", "predicted", "--predictions", str(out / "binary01_shuffled.jsonl")],
    }
    selected = args.arms or list(arms)
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "binary_sha256": sha256(binary), "queries": 300, "scaffold": False,
                "formula_note": "MassSpecGym masses are nearly exact; stored MIST predictions use true formulas",
                "shuffle_seed": 20261007, "source_sha256": {
                    name: sha256(data / name) for name in
                    ["msgym_validation.json", "msgym_validation_fp.json", "mist_pred_val.clean.jsonl"]},
                "commands": {name: common + arms[name] + ["--tag", name] for name in selected}}
    invocation = {"arms": selected, "tools_sha256": {
        path.name: sha256(path) for path in [Path(__file__), Path(__file__).with_name("summarize_conditioning_audit.py")]},
        "derived_inputs_sha256": {name: sha256(out / name) for name in
                                  ["roster.json", "binary01.jsonl", "binary05.jsonl", "shuffled.jsonl",
                                   "binary01_shuffled.jsonl", "shuffle_donors.json"]}}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ["checkpoint_sha256", "binary_sha256", "source_sha256"]:
            assert previous[field] == manifest[field], f"provenance mismatch: {field}"
        manifest["commands"] = {**previous["commands"], **manifest["commands"]}
        manifest["invocations"] = previous.get("invocations", [])
    manifest.setdefault("invocations", []).append(invocation)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for name in selected:
        command = manifest["commands"][name]
        print(f"starting {name}", flush=True)
        with (out / f"{name}.out").open("w") as log:
            subprocess.run(command, cwd=data, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = read_rows(out / f"{name}_predictions.jsonl")
        assert [(r["key"], r["spectrum_id"]) for r in rows] == [
            (r["key"], r["spectrum_id"]) for r in roster], f"{name}: roster mismatch"
        assert not any(r["inputs"]["scaffold"]["supplied"] for r in rows)
        if name not in {"A_exact", "B_removed", "G_exact_oracle_formula"}:
            assert all(r["inputs"]["fingerprint"]["source"] == "predicted" for r in rows)
        evaluation = json.loads((out / f"{name}_report.json").read_text())["evaluation"]
        print(f"finished {name}: recovery={evaluation['target_in_pool']}/300, "
              f"nonempty={evaluation['queries_with_a_candidate']}, "
              f"nll={evaluation['teacher_forced_nll']['per_molecule']:.3f}", flush=True)
    assert sha256(checkpoint) == manifest["checkpoint_sha256"], "checkpoint changed"


if __name__ == "__main__":
    main()
