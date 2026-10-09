"""Re-rank generated candidates by how well a real prediction fits them.

`tools/ms2/fit_fingerprint_channel.py` fits `P(prediction | true bits)` for a
fingerprint predictor. Read the other way round it scores a candidate: the
likelihood of the query's real predicted fingerprint if the candidate were
the answer,

    score(candidate) = log sum_k w_k prod_bit P_k(class of prediction | candidate bit, bit)

Unlike a plain Bernoulli likelihood of the candidate's bits under the
predicted probabilities, a bit the predictor never detects costs a candidate
nothing for carrying it, so the score does not drift towards small, plain
molecules.

Nothing taken from the answer enters a score: the inputs are the candidate
graphs of a `ms2_spectral_completion` predictions file and the predictor's
stored output for the query's spectrum. The answer is looked up only after
the order is fixed, under the competition's identity (tautomer-canonical
InChIKey first block, `competition_eval.scorer_key`).

Reported per order, over **all** queries of the file (an empty list is a
miss): pool recall, top-1, top-25 and MRR@25 with a bootstrap interval. The
orders are the run's own, the decoder's log-probability, the channel score,
and the channel score plus `--prior-weight` times the decoder
log-probability.

    PYTHONPATH=tools/ms2 python tools/ms2/channel_rerank.py \
        --predictions run/C_mist_predictions.jsonl \
        --fingerprints data/ms2/mist/out/mist_pred_val.jsonl \
        --channel data/ms2/specgen/conditioning_fix_20261008/channel.npz \
        --out run/C_mist_channel_rank.json --shortlists run/C_mist_shortlists.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from competition_eval import bootstrap, fingerprint, mol_of, scorer_key
from fit_fingerprint_channel import BITS, J, prediction_classes

_STATE: dict = {}


def _init(channel_path: str) -> None:
    data = np.load(channel_path)
    logt = data["logt"].astype(np.float64)  # [K, 2, BITS, CLASSES]
    _STATE["logt"] = logt
    _STATE["log_weights"] = np.log(data["weights"].astype(np.float64))


def channel_scores(pairs, candidate_bits: np.ndarray) -> np.ndarray:
    """Channel log-likelihood of one prediction for each candidate bit row."""
    logt, log_weights = _STATE["logt"], _STATE["log_weights"]
    classes = prediction_classes(pairs).astype(np.int64)
    off = logt[:, 0, J, classes]  # [K, BITS]
    gain = logt[:, 1, J, classes] - off
    per_class = off.sum(1)[None, :] + candidate_bits.astype(np.float64) @ gain.T  # [N, K]
    z = per_class + log_weights[None, :]
    m = z.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(z - m).sum(1))


def _rank_row(item):
    row, pairs, prior_weight = item
    keys, bits, log_probs = [], [], []
    for candidate in row["candidates"]:
        try:
            mol = mol_of(candidate["atoms"], candidate["bonds"])
            key = scorer_key(mol)
            fp = fingerprint(mol)
        except Exception:
            key, fp = None, None
        if not key or fp is None:
            continue
        on = np.zeros(BITS, dtype=bool)
        on[list(fp.GetOnBits())] = True
        keys.append(key)
        bits.append(on)
        log_probs.append(float(candidate.get("best_log_prob", 0.0)))
    try:
        target = scorer_key(mol_of(row["target"]["atoms"], row["target"]["bonds"]))
    except Exception:
        target = None
    out = {"key": row["key"], "spectrum_id": row["spectrum_id"], "candidates": len(keys), "ranks": {}}
    if not keys:
        return out, []
    log_probs = np.array(log_probs)
    orders = {
        "model": np.arange(len(keys)),
        "log_prob": np.argsort(-log_probs, kind="stable"),
    }
    if pairs is not None:
        scores = channel_scores(pairs, np.array(bits))
        orders["channel"] = np.argsort(-scores, kind="stable")
        orders["channel_plus_prior"] = np.argsort(-(scores + prior_weight * log_probs), kind="stable")
    shortlist = []
    for name, order in orders.items():
        distinct: list[str] = []
        for i in order:
            if keys[i] not in distinct:
                distinct.append(keys[i])
            if len(distinct) == 25:
                break
        # Scored only now that the order is fixed.
        out["ranks"][name] = distinct.index(target) + 1 if target in distinct else None
        if name == "channel_plus_prior":
            shortlist = distinct
    out["in_pool"] = target in keys
    return out, shortlist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--fingerprints", type=Path, required=True, help="the predictor's JSONL, by spectrum id")
    ap.add_argument("--channel", type=Path, required=True, help="the .npz of fit_fingerprint_channel.py")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--seen-keys", type=Path, default=None, help="molecule keys the predictor trained on")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shortlists", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.predictions.open()]
    wanted = {int(row["spectrum_id"]) for row in rows}
    pairs: dict[int, list] = {}
    with args.fingerprints.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            digits = "".join(c for c in str(record.get("id", "")) if c.isdigit())
            if digits and int(digits) in wanted:
                pairs[int(digits)] = record["bits"]
    items = [(row, pairs.get(int(row["spectrum_id"])), args.prior_weight) for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.channel),)) as pool:
        results = list(pool.map(_rank_row, items, chunksize=4))
    seen = set(args.seen_keys.read_text().split()) if args.seen_keys else set()

    def summary(selected) -> dict:
        out = {"queries": len(selected), "pool_recall": sum(bool(r.get("in_pool")) for r, _ in selected)}
        for name in ("model", "log_prob", "channel", "channel_plus_prior"):
            ranks = [r["ranks"].get(name) for r, _ in selected]
            rr = np.array([1.0 / x if x else 0.0 for x in ranks])
            out[name] = {
                "top1": int(sum(x == 1 for x in ranks)),
                "top25": int(sum(x is not None for x in ranks)),
                "mrr25": float(rr.mean()) if len(rr) else 0.0,
                "mrr25_ci95": bootstrap(rr),
            }
        return out

    report = {
        "all": summary(results),
        "predictor_unseen": summary([x for x in results if x[0]["key"] not in seen]) if seen else None,
        "predictor_seen": summary([x for x in results if x[0]["key"] in seen]) if seen else None,
        "queries_without_a_prediction": sum(1 for _, p, _ in items if p is None),
        "prior_weight": args.prior_weight,
        "hits": [
            {"key": r["key"], "spectrum_id": r["spectrum_id"], "candidates": r["candidates"], **r["ranks"]}
            for r, _ in results
            if r.get("in_pool")
        ],
        "per_query": [
            {"key": r["key"], "spectrum_id": r["spectrum_id"], "candidates": r["candidates"],
             "in_pool": bool(r.get("in_pool")), **r["ranks"]}
            for r, _ in results
        ],
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.predictions, args.fingerprints, args.channel)
        },
    }
    args.out.write_text(json.dumps(report, indent=1))
    if args.shortlists:
        with args.shortlists.open("w") as stream:
            for (r, shortlist), row in zip(results, rows):
                stream.write(json.dumps({"key": r["key"], "spectrum_id": r["spectrum_id"], "keys": shortlist}) + "\n")
    print(json.dumps({k: report[k] for k in ("all", "predictor_unseen", "queries_without_a_prediction")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
