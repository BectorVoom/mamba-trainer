"""Fit a per-bit error channel of a fingerprint predictor.

A completion model that trains on true fingerprints never learns what a
predicted one looks like, and the molecules that have a real prediction are
too few (and already memorised by the decoder) to teach it. This tool fits
the channel `P(prediction class | true bit value, bit index, quality class)`
from real predictions on molecules the predictor was **not** trained on, so
that `ms2_spectral_completion --fp-train channel` can degrade the true bits
of any molecule, including structure-only ones, the way the predictor would.

Model. A prediction is cut into 11 classes per bit: below 0.02 (not stored
by the prediction files), `[0.02, 0.05)`, `[0.05, 0.1)` and the eight
confidence buckets the decoder sees at or above 0.1 (`ceil(8 p)`). Each of
`K` latent quality classes owns one table per bit and truth value, shrunk
towards the class-wide table (`--prior` pseudo-counts) so a rare bit borrows
strength. The classes are fitted by EM on the training fold, started from
quantiles of per-spectrum recall.

Calibration. The training fold is easier for the predictor than a
structure-disjoint fold, so two things are re-estimated on a calibration
fold (never the evaluation roster): the class weights, and one miss rate
`lambda`, the probability that a true bit is predicted as if it were absent
(`P_on' = (1 - lambda) P_on + lambda P_off`), chosen on a grid by
calibration likelihood. The weights do most of the work; the likelihood is
nearly flat in `lambda` (the report carries the whole grid), so its value is
not identified beyond a broad range.

Limits. Bits are independent given the class, so the channel reproduces the
mean quality of a prediction but not all of its spread or the correlation
between a predictor's errors: real predictions score lower under the fitted
tables than samples drawn from them (`roster_real_log_likelihood` against
`roster_simulated_log_likelihood` in the report).

Outputs:

* `<out>.json` — `fingerprint_channel_v1`, what the Rust sampler reads: the
  class weights and, per class, bit and truth value, the probability of
  "no token" followed by the eight buckets.
* `<out>.npz` — the 11-class log tables and weights, for scoring candidate
  fingerprints against a prediction (`tools/ms2/channel_rerank.py`).
* `<out>.report.json` — held-out fidelity: real against simulated
  predictions on the excluded roster.

    python tools/ms2/fit_fingerprint_channel.py \
        --train-export data/ms2/specgen/msgym_train.json \
        --train-fp data/ms2/specgen/msgym_train_fp.json \
        --train-predictions data/ms2/mist/out/mist_pred_train.jsonl \
        --calibration-export data/ms2/specgen/msgym_validation.json \
        --calibration-fp data/ms2/specgen/msgym_validation_fp.json \
        --calibration-predictions data/ms2/mist/out/mist_pred_val.jsonl \
        --seen-keys data/ms2/mist/out/mist_train_val_aug_inchikeys.txt \
        --roster data/ms2/specgen/conditioning_audit_20261007/roster.json \
        --out data/ms2/specgen/conditioning_fix_20261008/channel
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BITS = 4096
CLASSES = 11  # 0: < 0.02, 1: [0.02, 0.05), 2: [0.05, 0.1), 3..10: decoder buckets 1..8
TOKEN_CLASS = 3  # first class the decoder sees as a token
THRESHOLD = 0.1
J = np.arange(BITS)


def prediction_classes(pairs) -> np.ndarray:
    """Class per bit of one prediction given as `[[bit, probability], ...]`."""
    out = np.zeros(BITS, dtype=np.int8)
    if pairs:
        a = np.asarray(pairs, dtype=np.float64)
        p = a[:, 1]
        bucket = np.clip(np.ceil(8.0 * p - 1e-9), 1, 8)
        k = np.where(p >= THRESHOLD, 2 + bucket, np.where(p >= 0.05, 2, np.where(p >= 0.02, 1, 0)))
        out[a[:, 0].astype(np.int64)] = k.astype(np.int8)
    return out


def load_fold(export: Path, fp: Path, predictions: Path, seen: set[str], exclude: set[str], only: set[str] | None = None,
              id_prefix: str = "MassSpecGymID"):
    """Prediction classes and true bits of every predicted spectrum of the
    export whose molecule is neither seen by the predictor nor excluded.
    With `only`, the population is instead exactly those molecules (minus
    the excluded ones), whatever the predictor saw."""
    data = json.loads(export.read_text())
    bits = json.loads(fp.read_text())["bits_by_molecule"]
    molecules = data["molecules"]
    by_id = {
        f"{id_prefix}{s['spectrum_id']:07d}": i
        for i, m in enumerate(molecules)
        for s in m["spectra"]
    }
    classes, truth, owner = [], [], []
    skipped = {"seen": 0, "excluded": 0, "not_in_export": 0}
    with predictions.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            i = by_id.get(row["id"])
            if i is None:
                # A prediction for a spectrum the export does not hold (the
                # prediction files cover the whole fold): counted, not fitted.
                skipped["not_in_export"] += 1
                continue
            key = molecules[i]["key"]
            if only is not None:
                if key not in only:
                    skipped["seen"] += 1
                    continue
            elif key in seen:
                skipped["seen"] += 1
                continue
            if key in exclude:
                skipped["excluded"] += 1
                continue
            classes.append(prediction_classes(row["bits"]))
            t = np.zeros(BITS, dtype=bool)
            t[bits[i]] = True
            truth.append(t)
            owner.append(i)
    return np.array(classes), np.array(truth), np.array(owner), skipped


def flat(classes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Index of every (truth, bit, class) cell in a `[2, BITS, CLASSES]` table."""
    return ((truth.astype(np.int64) * BITS + J[None, :]) * CLASSES + classes).astype(np.int32)


def fit_tables(cells: np.ndarray, resp: np.ndarray, prior: float) -> np.ndarray:
    """Log tables `[K, 2, BITS, CLASSES]` from responsibilities `[N, K]`."""
    k_count = resp.shape[1]
    counts = np.zeros((k_count, 2 * BITS * CLASSES))
    for k in range(k_count):
        for a in range(0, len(cells), 4096):
            block = cells[a : a + 4096]
            counts[k] += np.bincount(
                block.ravel(), weights=np.repeat(resp[a : a + 4096, k], BITS), minlength=counts.shape[1]
            )
    counts = counts.reshape(k_count, 2, BITS, CLASSES)
    glob = counts.sum(2) + 1e-3
    glob /= glob.sum(-1, keepdims=True)
    p = (counts + prior * glob[:, :, None, :]) / (counts.sum(-1, keepdims=True) + prior)
    return np.log(p)


def loglik(cells: np.ndarray, logt: np.ndarray) -> np.ndarray:
    """Log-likelihood `[N, K]` of every spectrum under every class."""
    k_count = logt.shape[0]
    tables = logt.reshape(k_count, -1)
    out = np.zeros((len(cells), k_count))
    for k in range(k_count):
        for a in range(0, len(cells), 2048):
            out[a : a + 2048, k] = tables[k][cells[a : a + 2048]].sum(1)
    return out


def mixture(ll: np.ndarray, weights: np.ndarray):
    """Mixture log-likelihood per spectrum and the responsibilities."""
    z = ll + np.log(weights)[None, :]
    m = z.max(1, keepdims=True)
    e = np.exp(z - m)
    return m[:, 0] + np.log(e.sum(1)), e / e.sum(1, keepdims=True)


def with_miss_rate(logt: np.ndarray, miss: float) -> np.ndarray:
    """Tables whose true bits are predicted like absent ones with probability `miss`."""
    p = np.exp(logt)
    out = p.copy()
    out[:, 1] = (1.0 - miss) * p[:, 1] + miss * p[:, 0]
    return np.log(out)


def simulate(logt: np.ndarray, weights: np.ndarray, truth: np.ndarray, rng) -> np.ndarray:
    p = np.exp(logt)
    out = np.zeros((len(truth), BITS), dtype=np.int8)
    for i, t in enumerate(truth):
        k = rng.choice(len(weights), p=weights / weights.sum())
        cdf = np.cumsum(p[k][t.astype(np.int64), J], axis=1)
        out[i] = (rng.random(BITS)[:, None] > cdf).sum(1).clip(0, CLASSES - 1)
    return out


def describe(classes: np.ndarray, truth: np.ndarray) -> dict:
    token = classes >= TOKEN_CLASS
    hit = (token & truth).sum(1)
    count = token.sum(1)
    confident = classes >= TOKEN_CLASS + 4  # probability above 0.5
    recall = hit / np.maximum(truth.sum(1), 1)
    precision = hit / np.maximum(count, 1)
    return {
        "spectra": int(len(classes)),
        "tokens_mean": float(count.mean()),
        "tokens_sd": float(count.std()),
        "precision_mean": float(precision.mean()),
        "precision_sd": float(precision.std()),
        "recall_mean": float(recall.mean()),
        "recall_sd": float(recall.std()),
        "confident_tokens_mean": float(confident.sum(1).mean()),
        "confident_precision": float((confident & truth).sum() / max(confident.sum(), 1)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("train", "calibration"):
        ap.add_argument(f"--{name}-export", type=Path, required=True)
        ap.add_argument(f"--{name}-fp", type=Path, required=True)
        ap.add_argument(f"--{name}-predictions", type=Path, required=True)
    ap.add_argument("--seen-keys", type=Path, required=True, help="molecule keys the predictor trained on, one per line")
    ap.add_argument("--roster", type=Path, required=True, help="JSON list of {key: ...} molecules kept out of calibration")
    ap.add_argument(
        "--only-keys",
        type=Path,
        default=None,
        help="fit on exactly these molecules instead of the predictor-unseen ones (e.g. the predictor's own "
        "training molecules, for the channel of a predictor that knows the answer's neighbourhood)",
    )
    ap.add_argument(
        "--skip-calibration",
        action="store_true",
        help="keep the training-fold class weights and no miss rate; the calibration fold only reports fidelity",
    )
    ap.add_argument(
        "--id-prefix",
        default="MassSpecGymID",
        help="prefix of the prediction ids before the zero-padded spectrum_id (CASMI exports use CASMI)",
    )
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--prior", type=float, default=20.0)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    args = ap.parse_args(argv)

    seen = set(args.seen_keys.read_text().split())
    roster = {q["key"] for q in json.loads(args.roster.read_text())}
    only = set(args.only_keys.read_text().split()) if args.only_keys else None
    c_train, t_train, _, skipped_train = load_fold(
        args.train_export, args.train_fp, args.train_predictions, seen, set(), only, args.id_prefix
    )
    c_cal, t_cal, _, skipped_cal = load_fold(
        args.calibration_export, args.calibration_fp, args.calibration_predictions, seen, roster, only, args.id_prefix
    )
    # The roster's own spectra: fidelity only, never fitted on.
    c_ros, t_ros, _, _ = load_fold(
        args.calibration_export,
        args.calibration_fp,
        args.calibration_predictions,
        seen,
        {m["key"] for m in json.loads(args.calibration_export.read_text())["molecules"]} - roster,
        only,
        args.id_prefix,
    )
    if args.skip_calibration:
        # Nothing is fitted on the calibration fold, so all of it can report fidelity.
        c_ros, t_ros = np.concatenate([c_ros, c_cal]), np.concatenate([t_ros, t_cal])
    print(f"train spectra {len(c_train)}, calibration spectra {len(c_cal)}, roster spectra {len(c_ros)}", flush=True)
    if min(len(c_train), len(c_cal), len(c_ros)) == 0:
        raise SystemExit("an input fold has no usable spectrum")

    cells = flat(c_train, t_train)
    recall = ((c_train >= TOKEN_CLASS) & t_train).sum(1) / np.maximum(t_train.sum(1), 1)
    k_count = args.classes
    if k_count == 1:
        resp = np.ones((len(c_train), 1))
    else:
        cuts = np.quantile(recall, np.linspace(0, 1, k_count + 1)[1:-1])
        resp = np.eye(k_count)[np.searchsorted(cuts, recall)]
    weights = resp.mean(0)
    curve = []
    for it in range(args.iterations if k_count > 1 else 1):
        logt = fit_tables(cells, resp, args.prior)
        train_ll, resp = mixture(loglik(cells, logt), weights)
        weights = resp.mean(0)
        curve.append(float(train_ll.mean()))
        print(f"EM {it}: train log-likelihood per spectrum {train_ll.mean():.2f}", flush=True)
    logt = fit_tables(cells, resp, args.prior)
    train_weights = weights.copy()

    cal_cells = flat(c_cal, t_cal)
    best = None
    grid = []
    for miss in (0.0,) if args.skip_calibration else (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        tables = with_miss_rate(logt, miss)
        ll = loglik(cal_cells, tables)
        w = train_weights.copy()
        for _ in range(0 if args.skip_calibration else 60):
            value, r = mixture(ll, w)
            w = np.maximum(r.mean(0), 1e-6)
            w /= w.sum()
        value = float(mixture(ll, w)[0].mean())
        grid.append({"miss_rate": miss, "calibration_log_likelihood": value})
        print(f"miss rate {miss:.2f}: calibration log-likelihood per spectrum {value:.2f}", flush=True)
        if best is None or value > best[0]:
            best = (value, miss, w, tables)
    _, miss, weights, tables = best

    rng = np.random.default_rng(args.seed)
    ros_cells = flat(c_ros, t_ros)
    simulated = simulate(tables, weights, t_ros, rng)
    uncalibrated = simulate(logt, train_weights, t_ros, rng)
    report = {
        "classes": k_count,
        "population": str(args.only_keys) if args.only_keys else "predictor-unseen molecules",
        "calibrated": not args.skip_calibration,
        "prior": args.prior,
        "miss_rate": miss,
        "weights": weights.tolist(),
        "train_weights": train_weights.tolist(),
        "em_train_log_likelihood": curve,
        "miss_rate_grid": grid,
        "spectra": {"train": int(len(c_train)), "calibration": int(len(c_cal)), "roster": int(len(c_ros))},
        "skipped": {"train": skipped_train, "calibration": skipped_cal},
        "roster_real": describe(c_ros, t_ros),
        "roster_simulated": describe(simulated, t_ros),
        "roster_simulated_uncalibrated": describe(uncalibrated, t_ros),
        "roster_real_log_likelihood": float(mixture(loglik(ros_cells, tables), weights)[0].mean()),
        "roster_simulated_log_likelihood": float(
            mixture(loglik(flat(simulated, t_ros), tables), weights)[0].mean()
        ),
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.train_export,
                args.train_predictions,
                args.calibration_export,
                args.calibration_predictions,
                args.seen_keys,
                args.roster,
            )
        },
    }

    # What the decoder sees: no token (classes 0..2) or one of eight buckets.
    p = np.exp(tables)
    visible = np.concatenate([p[..., :TOKEN_CLASS].sum(-1, keepdims=True), p[..., TOKEN_CLASS:]], axis=-1)
    visible /= visible.sum(-1, keepdims=True)
    channel = {
        "format": "fingerprint_channel_v1",
        "fingerprint": "morgan4096",
        "threshold": THRESHOLD,
        "buckets": 8,
        "classes": k_count,
        "miss_rate": miss,
        "weights": [round(float(w), 8) for w in weights],
        # [class][bit][outcome]: outcome 0 is "no token", 1..8 the bucket.
        "off": np.round(visible[:, 0], 7).tolist(),
        "on": np.round(visible[:, 1], 7).tolist(),
        "provenance": {
            "tool": "tools/ms2/fit_fingerprint_channel.py",
            "train_spectra": int(len(c_train)),
            "calibration_spectra": int(len(c_cal)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(channel, separators=(",", ":")))
    np.savez_compressed(f"{args.out}.npz", logt=tables.astype(np.float32), weights=weights)
    Path(f"{args.out}.report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("miss_rate", "roster_real", "roster_simulated", "roster_simulated_uncalibrated", "roster_real_log_likelihood", "roster_simulated_log_likelihood")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
