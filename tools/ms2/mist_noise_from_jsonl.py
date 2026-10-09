"""Build a `FingerprintNoise` file from predictions in the JSONL layout.

`tools/ms2/export_fingerprints_mist.py noise` reads the competition's panel
parquet files. This tool builds the same histograms from the files
`export_msgym_spectral.py` and an external predictor produce:

- `--predictions`: one JSON object per line, `{"id": "MassSpecGymID0000042",
  "bits": [[index, probability], ...]}` (any id whose digits are the
  export's `spectrum_id`); repeatable.
- `--export` with `--export-fp`: the molecules and their true on-bits, which
  give each spectrum its truth.

Output (the schema `completion_fingerprint.rs` `FingerprintNoise::load`
requires): 20 equal-width bins over `[0, 1]`, bin `i` covering
`[i/20, (i+1)/20)` with the last bin including 1.0, counting the predicted
probability of every true-on bit (`hist_pred_given_true_on`) and of every
true-off bit (`hist_pred_given_true_off`). A bit absent from a prediction
counts as probability 0 (bin 0), which is what an entry below the
predictor's own reporting threshold means. The `_molecule` variants pool the
per-molecule averaged probabilities (one row per molecule).

    PYTHONPATH=tools/ms2 python tools/ms2/mist_noise_from_jsonl.py \
        --export data/ms2/specgen/msgym_train.json \
        --export-fp data/ms2/specgen/msgym_train_fp.json \
        --predictions data/ms2/specgen/mist_pred_train.clean.jsonl \
        --out data/ms2/specgen/mist_noise.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BITS = 4096
N_BINS = 20


def bin_of(probabilities: np.ndarray) -> np.ndarray:
    """Bin index of each probability: `min(int(p * 20), 19)`."""
    return np.minimum((probabilities * N_BINS).astype(np.int64), N_BINS - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--export-fp", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    export = json.loads(args.export.read_text())
    sidecar = json.loads(args.export_fp.read_text())
    bits_by_molecule = sidecar["bits_by_molecule"]
    if len(bits_by_molecule) != len(export["molecules"]):
        raise SystemExit(
            f"{args.export_fp} holds {len(bits_by_molecule)} entries for "
            f"{len(export['molecules'])} molecules"
        )
    # spectrum_id -> molecule index, and the molecule's true on-bits.
    molecule_of: dict[int, int] = {}
    for index, molecule in enumerate(export["molecules"]):
        for spectrum in molecule["spectra"]:
            molecule_of[int(spectrum["spectrum_id"])] = index

    hist_on = np.zeros(N_BINS, dtype=np.int64)
    hist_off = np.zeros(N_BINS, dtype=np.int64)
    # Per-molecule averaged predictions: sum and count per molecule.
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    n_spectra = 0
    unmatched = 0
    for path in args.predictions:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                digits = "".join(c for c in row["id"] if c.isdigit())
                spectrum_id = int(digits)
                index = molecule_of.get(spectrum_id)
                if index is None:
                    unmatched += 1
                    continue
                dense = np.zeros(N_BITS, dtype=np.float64)
                for bit, probability in row["bits"]:
                    dense[int(bit)] = float(probability)
                truth = np.zeros(N_BITS, dtype=bool)
                truth[bits_by_molecule[index]] = True
                bins = bin_of(dense)
                np.add.at(hist_on, bins[truth], 1)
                np.add.at(hist_off, bins[~truth], 1)
                if index in sums:
                    sums[index] += dense
                    counts[index] += 1
                else:
                    sums[index] = dense
                    counts[index] = 1
                n_spectra += 1

    hist_on_molecule = np.zeros(N_BINS, dtype=np.int64)
    hist_off_molecule = np.zeros(N_BINS, dtype=np.int64)
    for index, total in sums.items():
        mean = total / counts[index]
        truth = np.zeros(N_BITS, dtype=bool)
        truth[bits_by_molecule[index]] = True
        bins = bin_of(mean)
        np.add.at(hist_on_molecule, bins[truth], 1)
        np.add.at(hist_off_molecule, bins[~truth], 1)

    if n_spectra == 0:
        raise SystemExit("no prediction matched a spectrum of the export")
    document = {
        "fingerprint": "morgan4096",
        "n_bins": N_BINS,
        "n_spectra": n_spectra,
        "n_molecules": len(sums),
        "source": {
            "export": args.export.name,
            "predictions": [p.name for p in args.predictions],
            "unmatched_predictions": unmatched,
        },
        "bins": "20 equal-width bins over [0, 1]; bin i covers [i/20, (i+1)/20), the last includes 1.0; a bit absent from a prediction counts as probability 0",
        "hist_pred_given_true_on": hist_on.tolist(),
        "hist_pred_given_true_off": hist_off.tolist(),
        "hist_pred_given_true_on_molecule": hist_on_molecule.tolist(),
        "hist_pred_given_true_off_molecule": hist_off_molecule.tolist(),
    }
    args.out.write_text(json.dumps(document, indent=1))
    on_mass = hist_on / max(hist_on.sum(), 1)
    print(json.dumps({
        "spectra": n_spectra, "molecules": len(sums), "unmatched": unmatched,
        "true_on_bits_total": int(hist_on.sum()), "true_off_bits_total": int(hist_off.sum()),
        "share_of_true_on_bits_predicted_at_or_above_0.1": round(float(on_mass[2:].sum()), 4),
        "share_of_true_on_bits_predicted_at_or_above_0.5": round(float(on_mass[10:].sum()), 4),
        "out": str(args.out),
    }, indent=1))


if __name__ == "__main__":
    main()
