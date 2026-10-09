"""Mix fitted fingerprint channels into one.

A decoder that will meet predictions of unknown quality should train on
more than one quality regime. Each input is a prefix written by
`fit_fingerprint_channel.py` with the share of molecules it should supply:

    python tools/ms2/merge_fingerprint_channels.py \\
        --part run/channel:0.8 --part run/channel_seen:0.2 --out run/channel_mixed

The classes of every part are kept as they are and their weights scaled by
the part's share, so a sample still draws one class per molecule. Writes the
same three files as the fit tool (`.json` for the Rust sampler, `.npz` for
`channel_rerank.py`, `.report.json` naming the parts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", action="append", required=True, help="PREFIX:SHARE")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    parts = []
    for text in args.part:
        prefix, share = text.rsplit(":", 1)
        parts.append((prefix, float(share)))
    total = sum(share for _, share in parts)
    if total <= 0 or any(share <= 0 for _, share in parts):
        raise SystemExit("every share must be positive")
    merged: dict | None = None
    tables, weights = [], []
    for prefix, share in parts:
        doc = json.loads(Path(f"{prefix}.json").read_text())
        scaled = [w * share / total for w in doc["weights"]]
        if merged is None:
            merged = {k: doc[k] for k in ("format", "fingerprint", "threshold", "buckets")}
            merged.update({"classes": 0, "weights": [], "off": [], "on": []})
        elif any(merged[k] != doc[k] for k in ("format", "fingerprint", "threshold", "buckets")):
            raise SystemExit(f"{prefix}.json was fitted for another format, fingerprint or threshold")
        merged["classes"] += doc["classes"]
        merged["weights"] += scaled
        merged["off"] += doc["off"]
        merged["on"] += doc["on"]
        data = np.load(f"{prefix}.npz")
        tables.append(data["logt"])
        weights.append(data["weights"] * share / total)
    merged["weights"] = [round(float(w), 8) for w in merged["weights"]]
    merged["provenance"] = {"tool": "tools/ms2/merge_fingerprint_channels.py", "parts": [{"prefix": p, "share": s / total} for p, s in parts]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(merged, separators=(",", ":")))
    np.savez_compressed(f"{args.out}.npz", logt=np.concatenate(tables), weights=np.concatenate(weights))
    Path(f"{args.out}.report.json").write_text(json.dumps(merged["provenance"], indent=1))
    print(f"{merged['classes']} classes, weights {merged['weights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
