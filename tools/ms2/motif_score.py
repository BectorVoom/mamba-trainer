"""Score a motif-decoder search under the competition's identity.

`examples/ms2_motif_decoder.rs --out FILE` writes, per query, the best
finished token sequences of its beam with their log-probability. This tool
rebuilds each sequence
into a molecule (`motif_tokens.MotifMachine`), keys it by the competition's
identity (tautomer-canonical InChIKey first block), and reports how often the
answer was generated and where it ranks.

Orders: the decoder's log-probability, and, when `--fingerprints` and
`--channel` are given, the channel likelihood of the query's real predicted
fingerprint (`channel_rerank.channel_scores`) alone and added to the
log-probability. Nothing from the answer enters an order; the answer's key
is looked up afterwards. It is taken from the export's typed graph (atoms and
bonds, no stereo), the same source `channel_rerank.py` uses for the
atom-level decoder: tautomer canonicalisation can give a stereo-marked and a
stereo-free form of one molecule different keys, so both scorers must start
from the same form.

`--keys FILE` (a JSON object or list of molecule keys) restricts the report
to that query set and fails if a key has no row, so two decoders can be
scored on exactly the same queries.

Every query of the file is in every denominator. `--groups` splits the
report by what the fingerprint predictor trained on. Candidates that do not
rebuild are dropped and identities that repeat are counted once before
ranking, which is what a submission built from this list would also have to
do. `writable` says the converter gave the answer a sequence; an answer
without one can still be hit through another sequence of the same identity.
The local RDKit (2026.03.6) is not the competition's pinned 2026.03.3.

    PYTHONPATH=tools/ms2 python tools/ms2/motif_score.py \
        --search run/exact.jsonl --vocab data/ms2/specgen/motif/lm_vocab.json \
        --export data/ms2/specgen/msgym_validation.json --out run/exact.score.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import channel_rerank
from competition_eval import bootstrap, fingerprint, mol_of, scorer_key
from fit_fingerprint_channel import BITS
from motif_tokens import ATOM_BASE, BOND_BASE, END_ID, MOTIF_BASE, MotifMachine

_VOCAB: list[str] = []


def _init(vocab_path: str, channel_path: str | None) -> None:
    _VOCAB.extend(m["smiles"] for m in json.loads(Path(vocab_path).read_text())["motifs"])
    if channel_path:
        channel_rerank._init(channel_path)


def symbolic(ids: list[int]) -> list:
    out = []
    for i in ids:
        if i == END_ID:
            out.append(["E"])
        elif i >= MOTIF_BASE:
            out.append(["M", _VOCAB[i - MOTIF_BASE]])
        elif i >= ATOM_BASE:
            out.append(["A", i - ATOM_BASE])
        else:
            out.append(["B", i - BOND_BASE])
    return out


def _score(item):
    row, graph, pairs, prior_weight = item
    try:
        target = scorer_key(mol_of(graph[0], graph[1]))
    except Exception:
        target = None
    keys, bits, log_probs = [], [], []
    invalid = 0
    for candidate in row["candidates"]:
        machine = MotifMachine()
        try:
            if machine.run(symbolic(candidate["tokens"])) is not None:
                invalid += 1
                continue
            built = machine.molecule()
            key = scorer_key(built)
            fp = fingerprint(built)
        except Exception:
            key, fp = None, None
        if not key or fp is None:
            invalid += 1
            continue
        on = np.zeros(BITS, dtype=bool)
        on[list(fp.GetOnBits())] = True
        keys.append(key)
        bits.append(on)
        log_probs.append(float(candidate["log_prob"]))
    out = {"key": row["key"], "index": row["index"], "candidates": len(keys), "invalid": invalid,
           "writable": row["target_tokens"] is not None, "in_pool": target in keys, "ranks": {}}
    if not keys:
        return out
    log_probs = np.array(log_probs)
    orders = {"log_prob": np.argsort(-log_probs, kind="stable")}
    if pairs is not None:
        scores = channel_rerank.channel_scores(pairs, np.array(bits))
        orders["channel"] = np.argsort(-scores, kind="stable")
        orders["channel_plus_prior"] = np.argsort(-(scores + prior_weight * log_probs), kind="stable")
    out["distinct"] = len(set(keys))
    for name, order in orders.items():
        distinct: list[str] = []
        for i in order:
            if keys[i] not in distinct:
                distinct.append(keys[i])
            if len(distinct) == 25:
                break
        out["ranks"][name] = distinct.index(target) + 1 if target in distinct else None
    return out


def summarise(rows: list[dict]) -> dict:
    out = {
        "queries": len(rows),
        "writable_by_the_alphabet": sum(r["writable"] for r in rows),
        "with_a_candidate": sum(r["candidates"] > 0 for r in rows),
        "pool_recall": sum(r["in_pool"] for r in rows),
        "mean_distinct_candidates": float(np.mean([r.get("distinct", 0) for r in rows])) if rows else 0.0,
        "invalid_candidates": sum(r["invalid"] for r in rows),
    }
    for name in ("log_prob", "channel", "channel_plus_prior"):
        ranks = [r["ranks"].get(name) for r in rows]
        rr = np.array([1.0 / x if x else 0.0 for x in ranks])
        out[name] = {
            "top1": int(sum(x == 1 for x in ranks)),
            "top25": int(sum(x is not None for x in ranks)),
            "mrr25": float(rr.mean()) if len(rr) else 0.0,
            "mrr25_ci95": bootstrap(rr),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--fingerprints", type=Path, default=None)
    ap.add_argument("--channel", type=Path, default=None)
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--groups", type=Path, action="append", default=[], help="NAME=FILE of molecule keys, repeatable")
    ap.add_argument("--keys", type=Path, default=None, help="JSON object or list of the molecule keys to report on")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.search.open()]
    if args.keys:
        manifest = list(json.loads(args.keys.read_text()))
        by_key = {row["key"]: row for row in rows}
        missing = [key for key in manifest if key not in by_key]
        if missing:
            raise SystemExit(f"{len(missing)} manifest queries have no search row, e.g. {missing[:3]}")
        rows = [by_key[key] for key in manifest]
    molecules = json.loads(args.export.read_text())["molecules"]
    pairs: dict[int, list] = {}
    if args.fingerprints and args.channel:
        wanted = {int(r["spectrum_id"]) for r in rows if r.get("spectrum_id") is not None}
        with args.fingerprints.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                digits = "".join(c for c in str(record.get("id", "")) if c.isdigit())
                if digits and int(digits) in wanted:
                    pairs[int(digits)] = record["bits"]
    items = []
    for row in rows:
        molecule = molecules[row["index"]]
        if molecule["key"] != row["key"]:
            raise SystemExit(f"query {row['query']}: key {row['key']} is not the export's {molecule['key']}")
        graph = (molecule["atoms"], molecule["bonds"])
        items.append((row, graph, pairs.get(row.get("spectrum_id")), args.prior_weight))
    channel = str(args.channel) if (args.fingerprints and args.channel) else None
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.vocab), channel)) as pool:
        scored = list(pool.map(_score, items, chunksize=4))
    report = {
        "all": summarise(scored),
        "groups": {},
        # A query without a stored prediction has no channel order and counts
        # as a miss in the two channel rows.
        "queries_without_a_prediction": sum(1 for item in items if item[2] is None) if channel else None,
        "duplicate_queries": len(rows) - len({row["key"] for row in rows}),
        "per_query": scored,
    }
    for spec in args.groups:
        name, path = str(spec).split("=", 1)
        keys = set(Path(path).read_text().split())
        report["groups"][name] = summarise([r for r in scored if r["key"] in keys])
        report["groups"][f"not_{name}"] = summarise([r for r in scored if r["key"] not in keys])
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"all": report["all"], "groups": report["groups"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
