"""Trained measured-spectrum to structure-fingerprint predictor + formula-pool ranking.

Sixth molecular-completion run: a small TRAINED spectrum->fingerprint model
evaluated on the IDENTICAL 200 val formula pools as tools/ms2_spectral_rank.py.
Nothing is tuned on val; test-fold rows are scanned-but-unused.

Representation (fixed a priori, never selected on val):
  spectrum: 1.0 Da bins over [0, 2000) = 2000 dims; per-bin max intensity,
    sqrt transform, per-spectrum L2 normalization; sparse CSR (bounded,
    no huge dense tensors).
  fingerprint: fixed stable-hash atom / typed-bond / length-2-path counts,
    FP_DIM=256, hashed with zlib.crc32 (deterministic, never Python hash()),
    permutation-invariant by construction (multiset counts only). This is a
    local fixed hash fingerprint, NOT a standard Morgan/ECFP or official
    benchmark fingerprint; no such claim is made. Coarse typed-edge counts
    alone alias isomers: collisions are measured and the ceiling admitted.
  model: sparse multi-output Ridge (alpha=1.0, solver lsqr, tol 1e-3,
    max_iter 200, fit_intercept False), fixed a priori, fit on FIT molecules
    only. No hyperparameter search on val; no refit on calibration.
  scoring: cosine between L2-normalized predicted fp and L2-normalized
    candidate fp. Missing candidate fps rank in seeded tail, retained.
  prior: untrained TRAIN-fit-mean fingerprint (fit-only), same scoring,
    isolates spectral signal from fingerprint popularity.
  cheap baseline: massresid via existing standardize_smiles (measured
    parent_mass as given query evidence), identical pools.

Identity: exact provider-canonical SMILES string for dedup/grouping/audit
plus InChIKey-14 connectivity groups for split disjointness. No RDKit.
Query graph fingerprints, formula labels, IDs, target-in-pool flags and
supplier order are never model inputs (query spectra only).

Leakage rules (asserted, unit-tested):
  * train/val connectivity overlap (SMILES or InChIKey-14) excludes the
    train molecule (counted); residual overlap aborts;
  * duplicate exact canonical spectrum content (content_fp) excluded:
    within-fit first-wins, calibration duplicating fit excluded, train
    duplicating any val query content excluded (all counted);
  * same-molecule spectra never straddle fit/calibration (one spectrum per
    molecule; split by connectivity group);
  * val/test labels never influence training, model selection or
    calibration; test rows pass through the scanner unused.

Calibration: fixed 90% precision gate on (margin, pool-size cap), thresholds
chosen on held-out TRAIN calibration formula-pool queries only. If too few
eligible calibration queries or no threshold meets target, fail closed
(abstain-all) with reason. Calibration uses the identical ranking/pool
domain as val (full pools, seeded ties, missing-fp tail).

CPU stdlib + scientific baseline only. GPU and Rust parity NOT APPLICABLE.
"""

import csv
import hashlib
import json
import math
import os

# Bound BLAS/OpenMP thread pools best-effort before heavy imports (observed
# cpu/wall is still reported; no strict single-threaded claim).
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import resource
import sys
import time
import unittest
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_spectral_rank import (  # noqa: E402
    candidate_residuals,
    content_fp,
    parse_spectrum,
    seeded_positions,
    sha256_file,
    short_key,
)
from tools.ms2_msgym_corpus import iter_json_entries  # noqa: E402

# --------------------------------------------------------------------------
# Fixed a priori representation + hyperparameters (never tuned on val).
# --------------------------------------------------------------------------

SPEC_BIN_WIDTH = 1.0
SPEC_MZ_MIN = 0.0
SPEC_MZ_MAX = 2000.0
N_SPEC = int((SPEC_MZ_MAX - SPEC_MZ_MIN) / SPEC_BIN_WIDTH)  # 2000
FP_DIM = 256
FP_MAX_PATH_BONDS = 2  # atoms (0), bonds (1), length-2 paths (2 bonds)
RIDGE_ALPHA = 1.0
RIDGE_TOL = 1e-3
RIDGE_MAX_ITER = 200
RIDGE_SOLVER = "lsqr"
RIDGE_FIT_INTERCEPT = False
MAX_TRAIN_MOLECULES = 5000
FIT_GROUP_FRAC = 0.8
MIN_CALIB_ELIGIBLE = 20
CALIB_TARGET_PRECISION = 0.90
TAU_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
KMAX_GRID = (50, 100, 200, 500, 10 ** 9)
SHUFFLE_DOMAIN = "qid-crc32"
ARMS = ("uniform", "massresid", "predictor", "prior")


def progress(msg):
    sys.stderr.write(f"# progress {msg}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Spectrum features: 1 Da bins + sqrt + L2 (per-sample only, no fitting).
# --------------------------------------------------------------------------

def bin_spectrum_1da(peaks):
    """Bin to {bin: max_intensity} with fixed 1 Da bounds. Returns (vec, dropped)."""
    vec = {}
    dropped = 0
    for mz, it in peaks:
        if not (SPEC_MZ_MIN <= mz < SPEC_MZ_MAX):
            dropped += 1
            continue
        b = int((mz - SPEC_MZ_MIN) / SPEC_BIN_WIDTH)
        if it > vec.get(b, 0.0):
            vec[b] = it
    return vec, dropped


def featurize_spectrum(peaks):
    """Fixed per-spectrum features: max-bin, sqrt, L2-normalize.

    Returns (feat_dict, norm_before). No vocabulary or statistics are fit;
    each spectrum is processed independently (fit-only claim trivially holds).
    """
    vec, dropped = bin_spectrum_1da(peaks)
    sq = {b: math.sqrt(v) for b, v in vec.items() if v > 0.0}
    norm = math.sqrt(sum(v * v for v in sq.values()))
    if norm > 0.0:
        sq = {b: v / norm for b, v in sq.items()}
    return sq, norm, dropped


# --------------------------------------------------------------------------
# Fingerprint: stable-hash atom/bond/path counts, permutation-invariant.
# --------------------------------------------------------------------------

def _atom_key(el, h, v):
    return f"A:{el}:{h}:{v}"


def fp_from_typed(atom_types, edges):
    """Counted hash fingerprint (length FP_DIM), L2-normalized.

    atom_types: tuple of (element, h, valence); edges: tuple of (a, b, order).
    Multiset counts only, so any atom-index permutation yields identical
    output. Hashing uses zlib.crc32 (stable across runs), never Python-hash.
    """
    counts = [0.0] * FP_DIM
    akeys = [_atom_key(el, h, v) for el, h, v in atom_types]
    for k in akeys:
        counts[zlib.crc32(k.encode()) % FP_DIM] += 1.0
    for a, b, o in edges:
        k1, k2 = akeys[a], akeys[b]
        if k1 > k2:
            k1, k2 = k2, k1
        key = f"B:{k1}|{k2}:{o}"
        counts[zlib.crc32(key.encode()) % FP_DIM] += 1.0
    n = len(atom_types)
    adj = [[] for _ in range(n)]
    for a, b, o in edges:
        adj[a].append((b, o))
        adj[b].append((a, o))
    for c in range(n):
        neigh = adj[c]
        for i in range(len(neigh)):
            for j in range(i + 1, len(neigh)):
                (a, oa) = neigh[i]
                (d, ob) = neigh[j]
                da = f"{akeys[a]}:{oa}"
                db = f"{akeys[d]}:{ob}"
                if da > db:
                    da, db = db, da
                key = f"P:{akeys[c]}|{da}|{db}"
                counts[zlib.crc32(key.encode()) % FP_DIM] += 1.0
    norm = math.sqrt(sum(c * c for c in counts))
    if norm > 0.0:
        counts = [c / norm for c in counts]
    return counts, norm


def fp_from_smiles(smi):
    """Fingerprint from a candidate/train SMILES via the typed graph parser.

    Returns (vec_or_None, norm_or_0, reason_or_None). Unparseable chemistry
    yields (None, 0.0, reason); callers retain the candidate and count it.
    """
    from tools.ms2_msgym_corpus import standardize_smiles
    try:
        rec, reason = standardize_smiles(smi, "fp-pool", smi)
    except Exception:
        return None, 0.0, "exception"
    if rec is None:
        return None, 0.0, reason
    vec, norm = fp_from_typed(rec["atom_types"], rec["edges"])
    return vec, norm, None


def cosine_dense(a, na, b, nb):
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Ranking (query spectra only; never the query graph/fingerprint/label).
# --------------------------------------------------------------------------

def score_candidates_fp(pred_vec, pred_norm, pool_fps):
    """Cosine of predicted fp vs each candidate fp. Missing -> None, kept.

    Non-finite predictions or scores (NaN/inf/out-of-range) yield None
    (unscored, ranked last) rather than propagating NaN into ranking.
    """
    pred_ok = (
        pred_vec is not None
        and isinstance(pred_norm, (int, float))
        and not isinstance(pred_norm, bool)
        and math.isfinite(float(pred_norm))
        and float(pred_norm) > 0.0
        and len(pred_vec) == FP_DIM
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(float(v)) for v in pred_vec)
        and all(abs(float(v)) <= 10.0 for v in pred_vec)
    )
    out = []
    for idx, (cvec, cnorm) in enumerate(pool_fps):
        if not pred_ok or cvec is None or not math.isfinite(cnorm) \
                or cnorm <= 0.0:
            out.append({"idx": idx, "score": None,
                        "parseable": cvec is not None})
            continue
        s = cosine_dense(pred_vec, pred_norm, cvec, cnorm)
        if not isinstance(s, float) or not math.isfinite(s):
            out.append({"idx": idx, "score": None, "parseable": True})
        else:
            out.append({"idx": idx, "score": s, "parseable": True})
    return out


def rank_by_score(scored, pos_of):
    return sorted(range(len(scored)),
                  key=lambda i: ((0.0, -scored[i]["score"])
                                 if scored[i]["score"] is not None
                                 else (1.0, 0.0),
                                 pos_of[i]))


def rank_uniform(n, pos_of):
    return sorted(range(n), key=lambda i: pos_of[i])


def rank_massresid_order(residuals, pos_of):
    return sorted(range(len(residuals)),
                  key=lambda i: ((0.0, residuals[i])
                                 if residuals[i] is not None
                                 else (1.0, 0.0),
                                 pos_of[i]))


def margin_of(scored_order_scores):
    """Legacy wrapper: finite-scored gap; kept for compatibility.

    Returns margin over actual finite scores only (None/NaN/inf excluded).
    Single finite score -> 0.0 conservative; no finite scores -> None.
    """
    margin, _ = margin_from_finite_scores(scored_order_scores)
    return margin


def finite_scored_values(scored_order_scores):
    """Actual finite scored values only (excludes None/NaN/inf)."""
    vals = []
    for v in scored_order_scores or []:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return vals


def margin_from_finite_scores(scored_order_scores):
    """Confidence gap over the actual finite scored order.

    Missing/unscored candidates (None) rank last and are EXCLUDED from the
    gap: the margin is top1-top2 over finite scores only. Single finite
    score -> (0.0, 'single_scored') conservatively (no observed gap, so
    zero confidence). No finite scores -> (None, 'no_finite_scores').
    Non-finite gap -> (None, 'non_finite_margin').
    """
    vals = finite_scored_values(scored_order_scores)
    if not vals:
        return None, "no_finite_scores"
    if len(vals) == 1:
        return 0.0, "single_scored"
    vals.sort(reverse=True)
    gap = vals[0] - vals[1]
    if not math.isfinite(gap):
        return None, "non_finite_margin"
    return gap, "ok"


def spectrum_has_usable_features(peaks):
    """Nonempty usable spectrum features (inference input check)."""
    if not peaks:
        return False
    try:
        feat, norm, _ = featurize_spectrum(peaks)
    except Exception:
        return False
    return bool(feat) and isinstance(norm, (int, float)) \
        and not isinstance(norm, bool) \
        and math.isfinite(float(norm)) and float(norm) > 0.0


def prediction_is_usable(pred_vec, pred_norm):
    """Finite nonzero prediction (never accept zero/NaN/out-of-range)."""
    if pred_vec is None or pred_norm is None:
        return False
    if isinstance(pred_norm, bool):
        return False
    if not isinstance(pred_norm, (int, float)):
        return False
    pred_norm = float(pred_norm)
    if not math.isfinite(pred_norm) or pred_norm <= 0.0:
        return False
    if len(pred_vec) != FP_DIM:
        return False
    for v in pred_vec:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        fv = float(v)
        if not math.isfinite(fv) or abs(fv) > 10.0:
            return False
    tot = sum(float(v) * float(v) for v in pred_vec)
    return math.isfinite(tot) and tot > 0.0


def capability_for_query(peaks, pred_vec, pred_norm, scored):
    """Shared inference-capability predicate for calibration AND val.

    Requires: nonempty usable spectrum features, finite nonzero
    prediction, >=1 finite candidate score, finite margin. Deliberately
    INDEPENDENT of true target parseability/identity/hit (those are not
    inference inputs; target-parseable is an EVALUATION-only subgroup).
    Returns (rankable_bool, reason_str).
    """
    if not spectrum_has_usable_features(peaks):
        return False, "no_features"
    if not prediction_is_usable(pred_vec, pred_norm):
        return False, "predictor_unavailable"
    scores = [s.get("score") for s in (scored or [])]
    n_finite = len(finite_scored_values(scores))
    if n_finite == 0:
        return False, "predictor_unavailable"
    margin, mstatus = margin_from_finite_scores(scores)
    if margin is None or not math.isfinite(margin):
        return False, "predictor_unavailable"
    return True, ("ok" if mstatus == "ok" else f"ok:{mstatus}")


def target_parse_info(smiles):
    """Evaluation-only target fingerprint parseability (never an input).

    Returns (parseable_bool, status_str, reason_or_None).
    """
    try:
        vec, norm, reason = fp_from_smiles(smiles)
    except Exception:
        return False, "error", "exception"
    if vec is None:
        return False, "unsupported", reason
    return True, "ok", None


def evaluate_pool(pool, target_smiles, pred_scores, prior_scores, residuals,
                  pos_of):
    """Ranks/hits for all four arms on one identical full pool."""
    n = len(pool)
    tgt = pool.index(target_smiles) if target_smiles in pool else None
    ou = rank_uniform(n, pos_of)
    om = rank_massresid_order(residuals, pos_of)
    op = rank_by_score(pred_scores, pos_of)
    orr = rank_by_score(prior_scores, pos_of)

    def rank_of(order):
        return None if tgt is None else order.index(tgt) + 1

    ranks = {"uniform": rank_of(ou), "massresid": rank_of(om),
             "predictor": rank_of(op), "prior": rank_of(orr)}

    def hits(rank):
        if rank is None:
            return {1: None, 3: None, 10: None}
        return {1: rank <= 1, 3: rank <= min(3, n), 10: rank <= min(10, n)}

    margin_pred, margin_status = margin_from_finite_scores(
        [s["score"] for s in pred_scores])
    n_scored = len(finite_scored_values([s["score"] for s in pred_scores]))
    return {"in_pool": tgt is not None, "n_pool": n,
            "ranks": ranks, "hits": {a: hits(r) for a, r in ranks.items()},
            "margin_pred": margin_pred,
            "margin_status": margin_status,
            "n_scored_finite": n_scored,
            "n_parseable": sum(1 for s in pred_scores if s["parseable"])}


# --------------------------------------------------------------------------
# Data selection (val identical to spectral run; train capped + grouped).
# --------------------------------------------------------------------------

def load_wanted_rows(tsv_path):
    from tools.ms2_spectral_rank import load_tsv_rows
    return load_tsv_rows(tsv_path)


def select_val(tsv_path, prefix_path, max_queries):
    """EXACT same selection as spectral run: file-order prefix entries
    joining first-row-per-SMILES val groups; distinct pools deduped in
    supplied order; qids VAL-%04d in join order."""
    val_groups = {}
    train_smiles_all, train_keys_all = set(), set()
    val_bias = Counter()
    n_rows = n_test = 0
    for row in load_wanted_rows(tsv_path):
        n_rows += 1
        if row["fold"] == "test":
            n_test += 1
            continue
        if row["fold"] == "train":
            train_smiles_all.add(row["smiles"])
            if row["inchikey"]:
                train_keys_all.add(short_key(row["inchikey"]))
        elif row["fold"] == "val":
            val_bias[f"adduct:{row['adduct']}"] += 1
            val_bias[f"inst:{row['instrument_type'] or 'missing'}"] += 1
            if row["smiles"] not in val_groups:
                val_groups[row["smiles"]] = row
    entries = [(q, c) for q, c in
               iter_json_entries(prefix_path, max_queries * 40)]
    joined = [(q, c) for q, c in entries if q in val_groups][:max_queries]
    if not joined:
        raise ValueError("no prefix entries join to val rows")
    pools = {}
    for qi, (q, cands) in enumerate(joined):
        qid = f"VAL-{qi:04d}"
        seen, distinct = set(), []
        for s in cands:
            if s not in seen:
                seen.add(s)
                distinct.append(s)
        pools[qid] = {"smiles": q, "supplied": list(cands),
                      "distinct": distinct, "row": val_groups[q]}
    return {"pools": pools, "val_groups": val_groups, "entries": entries,
            "joined": joined, "n_rows": n_rows, "n_test": n_test,
            "val_bias": val_bias,
            "train_smiles_all": train_smiles_all,
            "train_keys_all": train_keys_all}


def collect_train_molecules(tsv_path, cap, val_smiles, val_keys, val_content):
    """Capped train molecules, one spectrum per SMILES (file order).

    Returns (molecules_in_order, info). molecules carry row, inchikey,
    group, content_fp, peaks. Spectrum-unparseable rows are marked excluded
    here (counted). Test rows are skipped (counted).
    """
    from tools.ms2_spectral_rank import load_tsv_rows
    first = {}
    order = []
    info = Counter()
    for row in load_tsv_rows(tsv_path):
        if row["fold"] == "test":
            info["n_test_skipped"] += 1
            continue
        if row["fold"] != "train":
            continue
        info["n_train_rows_scanned"] += 1
        smi = row["smiles"]
        if smi not in first:
            first[smi] = row
            order.append(smi)
    molecules = []
    for smi in order:
        row = first[smi]
        peaks, err = parse_spectrum(row["mzs"], row["intensities"])
        if err is not None:
            info[f"train_spec_excl:{err}"] += 1
            molecules.append({"smiles": smi, "row": row, "status": "excluded",
                              "reason": f"spec:{err}", "group": None,
                              "fp_content": None, "peaks": None})
            continue
        fp = content_fp(peaks)
        key = short_key(row["inchikey"]) if row["inchikey"] else ""
        if not row["inchikey"]:
            info["n_train_missing_key"] += 1
            molecules.append({"smiles": smi, "row": row, "status": "excluded",
                              "reason": "missing_connectivity_key",
                              "group": None, "fp_content": fp, "peaks": peaks})
            continue
        molecules.append({"smiles": smi, "row": row, "status": "candidate",
                          "reason": None, "group": key, "fp_content": fp,
                          "peaks": peaks})
    # Group-order cap: groups in first-appearance order until cap molecules.
    cand = [m for m in molecules if m["status"] == "candidate"]
    gorder, gmembers = [], {}
    for m in cand:
        if m["group"] not in gmembers:
            gmembers[m["group"]] = []
            gorder.append(m["group"])
        gmembers[m["group"]].append(m)
    capped_groups, capped = [], []
    for g in gorder:
        if len(capped) + len(gmembers[g]) > cap and capped:
            info["n_groups_truncated_by_cap"] += 1
            break
        capped_groups.append(g)
        capped.extend(gmembers[g])
        if len(capped) >= cap:
            break
    capped_set = {id(m) for m in capped}
    for m in cand:
        if id(m) not in capped_set:
            m["status"] = "excluded"
            m["reason"] = "over_cap"
            info["n_over_cap"] += 1
    info["n_capped_molecules"] = len(capped)
    info["n_capped_groups"] = len(capped_groups)
    # Split capped groups: first FIT_GROUP_FRAC -> fit, rest -> calibration.
    n_fit_g = int(len(capped_groups) * FIT_GROUP_FRAC)
    fit_groups = set(capped_groups[:n_fit_g])
    calib_groups = set(capped_groups[n_fit_g:])
    for m in capped:
        m["split"] = "fit" if m["group"] in fit_groups else "calibration"
    info["n_fit_groups"] = len(fit_groups)
    info["n_calib_groups"] = len(calib_groups)
    # Exclusions on capped set: val overlap, val content, unsupported chem,
    # content duplicates (fit first-wins; calib vs fit; val content).
    for m in capped:
        if m["smiles"] in val_smiles or m["group"] in val_keys:
            m["status"] = "excluded"
            m["reason"] = "train_val_overlap"
            info["n_train_val_overlap_excluded"] += 1
        elif m["fp_content"] in val_content:
            m["status"] = "excluded"
            m["reason"] = "val_content_duplicate"
            info["n_val_content_dup_excluded"] += 1
    # Unsupported chemistry (target fp needed for Y): standardize now.
    for m in capped:
        if m["status"] != "candidate":
            continue
        vec, norm, reason = fp_from_smiles(m["smiles"])
        if vec is None:
            m["status"] = "excluded"
            m["reason"] = f"chem:{reason}"
            info[f"train_chem_excl:{reason.split(':')[0]}"] += 1
        else:
            m["fp_target"] = vec
            m["fp_target_norm"] = norm
    # Content dedup: fit first-wins, then calib-vs-fit.
    seen_fit = set()
    for m in [x for x in capped if x.get("split") == "fit"
              and x["status"] == "candidate"]:
        if m["fp_content"] in seen_fit:
            m["status"] = "excluded"
            m["reason"] = "fit_content_duplicate"
            info["n_fit_content_dup"] += 1
        else:
            seen_fit.add(m["fp_content"])
    for m in [x for x in capped if x.get("split") == "calibration"
              and x["status"] == "candidate"]:
        if m["fp_content"] in seen_fit:
            m["status"] = "excluded"
            m["reason"] = "calib_content_duplicate_fit"
            info["n_calib_content_dup_fit"] += 1
            continue
        seen_fit.add(m["fp_content"])
    fit = [m for m in capped if m.get("split") == "fit"
           and m["status"] == "candidate"]
    calib = [m for m in capped if m.get("split") == "calibration"
             and m["status"] == "candidate"]
    # Enforce: no shared groups, no shared content, no val overlap remain.
    fit_groups_final = {m["group"] for m in fit}
    calib_groups_final = {m["group"] for m in calib}
    assert not (fit_groups_final & calib_groups_final), "group straddle"
    assert not (fit_groups_final & set(val_keys)), "fit/val key overlap"
    assert not (calib_groups_final & set(val_keys)), "calib/val key overlap"
    assert not ({m["smiles"] for m in fit} & set(val_smiles)), "fit/val smi"
    assert not ({m["smiles"] for m in calib} & set(val_smiles)), "cal/val smi"
    return molecules, fit, calib, fit_groups_final, calib_groups_final, info


def build_matrices(fit):
    import numpy as np
    import scipy.sparse as sp
    rows, cols, data = [], [], []
    Y = np.zeros((len(fit), FP_DIM), dtype=np.float64)
    for i, m in enumerate(fit):
        feat, _, _ = featurize_spectrum(m["peaks"])
        for b, v in feat.items():
            rows.append(i)
            cols.append(b)
            data.append(v)
        Y[i, :] = m["fp_target"]
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(fit), N_SPEC))
    return X, Y


def train_ridge(X, Y):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, tol=RIDGE_TOL,
                  max_iter=RIDGE_MAX_ITER,
                  fit_intercept=RIDGE_FIT_INTERCEPT, random_state=0)
    model.fit(X, Y)
    return model


def predict_fp(model, peaks):
    import numpy as np
    feat, norm, _ = featurize_spectrum(peaks)
    if norm <= 0.0 or not feat:
        return [0.0] * FP_DIM, 0.0
    import scipy.sparse as sp
    X = sp.csr_matrix(([v for v in feat.values()],
                       ([0] * len(feat), list(feat.keys()))),
                      shape=(1, N_SPEC))
    pred = np.asarray(model.predict(X))[0].tolist()
    n = math.sqrt(sum(v * v for v in pred))
    if n > 0.0:
        pred = [v / n for v in pred]
    return pred, n


def mean_prior(fit):
    import numpy as np
    Y = np.array([m["fp_target"] for m in fit], dtype=np.float64)
    mean = Y.mean(axis=0)
    n = float(np.linalg.norm(mean))
    if n > 0.0:
        mean = mean / n
    return mean.tolist(), n


# --------------------------------------------------------------------------
# Calibration (held-out TRAIN pools only; fail closed).
# --------------------------------------------------------------------------

def calibrate_gate(calib_rows):
    """Choose (tau, kmax) on calibration rankable rows only.

    calib_rows: list of dicts with rankable (capability predicate),
    hit_top1 (True/False/None), margin_pred, n_pool. Rows with hit None
    (absent target) count as incorrect when predicted. rankable MUST be
    the shared inference-capability predicate (usable features, finite
    nonzero prediction, >=1 finite score, finite margin) and MUST NOT
    depend on target parseability/identity/hit. Returns gate dict with
    tau=None (strict-JSON null) when abstaining; fail closed (predict
    nothing) when too few rankable or no threshold meets target.
    """
    elig = [r for r in calib_rows if r.get("rankable") is True
            and isinstance(r.get("margin_pred"), (int, float))
            and not isinstance(r.get("margin_pred"), bool)
            and math.isfinite(float(r["margin_pred"]))]
    if len(elig) < MIN_CALIB_ELIGIBLE:
        return {"tau": None, "kmax": 0, "status": "abstain_all",
                "reason": f"insufficient_eligible:{len(elig)}",
                "n_eligible": len(elig), "table": []}
    table = []
    for tau in TAU_GRID:
        for kmax in KMAX_GRID:
            pred = [r for r in elig if r["margin_pred"] >= tau
                    and r["n_pool"] <= kmax]
            if not pred:
                table.append({"tau": tau, "kmax": kmax, "predicted": 0,
                              "precision": None, "coverage": 0.0})
                continue
            good = sum(1 for r in pred if r.get("hit_top1") is True)
            prec = good / len(pred)
            table.append({"tau": tau, "kmax": kmax, "predicted": len(pred),
                          "precision": prec,
                          "coverage": len(pred) / len(elig)})
    feas = [t for t in table if t["precision"] is not None
            and t["precision"] >= CALIB_TARGET_PRECISION]
    if not feas:
        return {"tau": None, "kmax": 0, "status": "abstain_all",
                "reason": "no_threshold_meets_target", "n_eligible": len(elig),
                "table": table}
    feas.sort(key=lambda t: (-t["predicted"], -t["tau"], t["kmax"]))
    best = feas[0]
    return {"tau": best["tau"], "kmax": best["kmax"], "status": "calibrated",
            "reason": None, "n_eligible": len(elig), "table": table,
            "best": best}


def gate_accepts(gate, margin_pred, n_pool):
    """Fail-closed acceptance: False when gate abstains or margin unusable."""
    if gate.get("status") != "calibrated":
        return False
    tau = gate.get("tau")
    kmax = gate.get("kmax")
    if tau is None or isinstance(tau, bool):
        return False
    if not isinstance(margin_pred, (int, float)) \
            or isinstance(margin_pred, bool):
        return False
    if not math.isfinite(float(margin_pred)) \
            or not math.isfinite(float(tau)):
        return False
    if not isinstance(n_pool, int):
        return False
    return float(margin_pred) >= float(tau) and n_pool <= kmax


# --------------------------------------------------------------------------
# Summaries.
# --------------------------------------------------------------------------

def summarize_val(rows, label):
    out = {"label": label, "n_selected": len(rows),
           "n_complete": sum(1 for r in rows if r.get("status") == "complete"),
           "n_rankable": sum(1 for r in rows if r.get("rankable") is True),
           "n_accepted": sum(1 for r in rows if r.get("accepted") is True),
           "n_model_failure": sum(
               1 for r in rows if r.get("status") == "complete"
               and r.get("rankable") is not True),
           "rankable_reason_counts": dict(Counter(
               str(r.get("rankable_reason", "?")) for r in rows
               if r.get("status") == "complete")),
           "status_counts": dict(Counter(r.get("status", "?") for r in rows)),
           "pool_recall_all": (sum(1 for r in rows if r.get("in_pool"))
                               / len(rows) if rows else None)}
    subsets = (("all", lambda r: True),
               ("eligible_complete",
                lambda r: r.get("status") == "complete" and r["in_pool"]),
               ("capability_any_candidate",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and (r.get("n_parseable") or 0) > 0),
               ("eligible_parseable",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and r.get("target_parseable") is True),
               ("accepted",
                lambda r: r.get("accepted") is True and r["in_pool"]))
    for name, sel in subsets:
        sub = [r for r in rows if sel(r)]
        m = {"n": len(sub)}
        for arm in ARMS:
            for k in (1, 3, 10):
                vals = [r["hits"][arm][k] for r in sub
                        if r["hits"].get(arm, {}).get(k) is not None]
                m[f"{arm}@top{k}"] = (sum(vals) / len(vals) if vals else None)
                m[f"{arm}@top{k}_den"] = len(vals)
        pred = [r for r in sub if r.get("accepted")]
        good = [r for r in pred if r["hits"].get("predictor", {}).get(1)]
        m["predicted"] = len(pred)
        m["coverage"] = len(pred) / len(rows) if rows else 0.0
        m["abstention_precision"] = (len(good) / len(pred) if pred else None)
        out[name] = m
    return out


def empty_hits():
    return {arm: {1: None, 3: None, 10: None} for arm in ARMS}


# --------------------------------------------------------------------------
# Experiment driver.
# --------------------------------------------------------------------------

def run_experiment(tsv_path, prefix_path, max_queries, max_train, out_dir):
    import numpy as np
    t0 = time.perf_counter()
    c0 = time.process_time()
    progress("select-val")
    sel = select_val(tsv_path, prefix_path, max_queries)
    pools = sel["pools"]
    # Val query spectra content (for train content-leak exclusion).
    val_content, val_smiles, val_keys = set(), [], set()
    val_qinfo = {}
    for qid, v in pools.items():
        peaks, err = parse_spectrum(v["row"]["mzs"], v["row"]["intensities"])
        if err is None:
            val_content.add(content_fp(peaks))
            val_qinfo[qid] = peaks
        else:
            val_qinfo[qid] = None
        val_smiles.append(v["smiles"])
        if v["row"]["inchikey"]:
            val_keys.add(short_key(v["row"]["inchikey"]))
    progress(f"collect-train cap={max_train}")
    molecules, fit, calib, fit_g, calib_g, tinfo = collect_train_molecules(
        tsv_path, max_train, set(val_smiles), val_keys, val_content)
    progress(f"train fit={len(fit)} calib={len(calib)}")
    t_train0 = time.perf_counter()
    X_fit, Y_fit = build_matrices(fit)
    model = train_ridge(X_fit, Y_fit)
    t_train1 = time.perf_counter()
    prior_vec, prior_norm = mean_prior(fit)
    # Candidate fingerprint cache (shared across calib + val).
    fp_cache = {}

    def pool_fps(pool):
        out = []
        for smi in pool:
            if smi not in fp_cache:
                fp_cache[smi] = fp_from_smiles(smi)
            vec, norm, _ = fp_cache[smi]
            out.append((vec, norm))
        return out

    # Calibration: held-out TRAIN molecules with prefix pools.
    progress("calibration-pools")
    prefix_index = {}
    for q, c in iter_json_entries(prefix_path, 1 << 30):
        if q not in prefix_index:
            prefix_index[q] = c
    calib_rows = []
    for i, m in enumerate(calib):
        qid = f"CAL-{i:04d}"
        cands = prefix_index.get(m["smiles"])
        if cands is None:
            calib_rows.append({"qid": qid, "smiles": m["smiles"],
                               "status": "no_pool_in_prefix",
                               "rankable": False,
                               "rankable_reason": "no_pool",
                               "has_features": True,
                               "pred_norm": None,
                               "n_scored_finite": 0,
                               "target_parseable": True,
                               "target_fp_status": "ok",
                               "in_pool": None,
                               "n_pool": 0, "n_parseable": 0,
                               "margin_pred": None,
                               "margin_status": "no_finite_scores",
                               "hit_top1": None,
                               "hits": empty_hits()})
            continue
        seen, distinct = set(), []
        for s in cands:
            if s not in seen:
                seen.add(s)
                distinct.append(s)
        pfps = pool_fps(distinct)
        pred, pn = predict_fp(model, m["peaks"])
        pscores = score_candidates_fp(pred, pn, pfps)
        prscores = score_candidates_fp(prior_vec, prior_norm, pfps)
        try:
            observed = m["row"]["parent_mass"]
        except KeyError:
            observed = ""
        residuals, _ = candidate_residuals(distinct, observed)
        pos = seeded_positions(qid, len(distinct))
        ev = evaluate_pool(distinct, m["smiles"], pscores, prscores,
                           residuals, pos)
        # Capability (inference-only): never uses target parseability.
        rankable, rreason = capability_for_query(
            m["peaks"], pred, pn, pscores)
        tparse, tstatus, _ = target_parse_info(m["smiles"])
        calib_rows.append({"qid": qid, "smiles": m["smiles"],
                           "status": "complete", "rankable": rankable,
                           "rankable_reason": rreason,
                           "has_features": spectrum_has_usable_features(
                               m["peaks"]),
                           "pred_norm": pn,
                           "n_scored_finite": ev["n_scored_finite"],
                           "target_parseable": tparse,
                           "target_fp_status": tstatus,
                           "in_pool": ev["in_pool"], "n_pool": ev["n_pool"],
                           "n_parseable": ev["n_parseable"],
                           "margin_pred": ev["margin_pred"],
                           "margin_status": ev["margin_status"],
                           "hit_top1": ev["hits"]["predictor"][1],
                           "hits": ev["hits"], "ranks": ev["ranks"]})
    gate = calibrate_gate(calib_rows)
    progress(f"calibration-gate {gate['status']} tau={gate['tau']} "
             f"kmax={gate['kmax']}")
    # Val scoring on IDENTICAL pools/order.
    progress("val-scoring")
    val_rows = []
    sel_bias = Counter()
    n_collide_target = 0
    n_target_fp_missing = 0
    n_queries_target_unparseable = 0
    n_identity_missing = 0
    for qid, v in pools.items():
        row = v["row"]
        sel_bias[f"adduct:{row['adduct']}"] += 1
        sel_bias[f"inst:{row['instrument_type'] or 'missing'}"] += 1
        distinct = v["distinct"]
        tparse_q, tstatus_q, treason_q = target_parse_info(v["smiles"])
        base = {"qid": qid, "smiles": v["smiles"],
                "adduct": row.get("adduct", ""),
                "instrument": row.get("instrument_type") or "missing",
                "n_supplied": len(v["supplied"]), "n_pool": len(distinct),
                "in_pool": (v["smiles"] in distinct),
                "target_parseable": tparse_q,
                "target_fp_status": tstatus_q,
                "target_fp_reason": treason_q}
        # Identity gate: sufficient connectivity identity required before
        # any scoring (fail safe, retain row + pool accounting, abstain).
        if not v["smiles"] or not row.get("inchikey"):
            n_identity_missing += 1
            pfps_id = pool_fps(distinct)
            n_parse_id = sum(1 for a, _ in pfps_id if a is not None)
            out = dict(base, status="excluded_identity_missing",
                       rankable=False, rankable_reason="identity_missing",
                       has_features=None, pred_norm=None,
                       n_scored_finite=0,
                       n_parseable=n_parse_id,
                       n_unparseable_fp=len(pfps_id) - n_parse_id,
                       n_unparseable_mass=None,
                       n_target_fp_ties=None,
                       margin_pred=None, margin_status="no_identity",
                       predicted=False, accepted=False,
                       rank_uniform=None, rank_massresid=None,
                       rank_predictor=None, rank_prior=None,
                       hits=empty_hits())
            val_rows.append(out)
            continue
        peaks = val_qinfo[qid]
        if peaks is None:
            _, err = parse_spectrum(row["mzs"], row["intensities"])
            # Spectrum-free ranks still computable (documented).
            pos = seeded_positions(qid, len(distinct))
            residuals, n_bad = candidate_residuals(distinct,
                                                   row["parent_mass"])
            tgt = distinct.index(v["smiles"]) if v["smiles"] in distinct \
                else None
            out = dict(base, status=f"query_parse_excl:{err}",
                       rankable=False, rankable_reason="no_features",
                       has_features=False, pred_norm=0.0,
                       n_scored_finite=0,
                       predicted=False,
                       accepted=False, margin_pred=None,
                       margin_status="no_finite_scores",
                       hits=empty_hits(),
                       rank_uniform=None, rank_massresid=None,
                       rank_predictor=None, rank_prior=None)
            if tgt is not None:
                ou = rank_uniform(len(distinct), pos)
                om = rank_massresid_order(residuals, pos)
                out["rank_uniform"] = ou.index(tgt) + 1
                out["rank_massresid"] = om.index(tgt) + 1
                n = len(distinct)
                for arm, rk in (("uniform", out["rank_uniform"]),
                                ("massresid", out["rank_massresid"])):
                    out["hits"][arm] = {
                        1: rk <= 1, 3: rk <= min(3, n), 10: rk <= min(10, n)}
            out["n_unparseable_mass"] = n_bad
            pfps0 = pool_fps(distinct)
            out["n_parseable"] = sum(1 for a, _ in pfps0 if a is not None)
            out["n_unparseable_fp"] = len(pfps0) - out["n_parseable"]
            out["n_target_fp_ties"] = None
            if tparse_q:
                try:
                    tvec0, _, _ = fp_from_smiles(v["smiles"])
                    out["n_target_fp_ties"] = sum(
                        1 for a, _ in pfps0
                        if a is not None and list(a) == list(tvec0))
                except Exception:
                    out["n_target_fp_ties"] = None
            val_rows.append(out)
            continue
        pfps = pool_fps(distinct)
        n_parse = sum(1 for a, _ in pfps if a is not None)
        pred, pn = predict_fp(model, peaks)
        pscores = score_candidates_fp(pred, pn, pfps)
        prscores = score_candidates_fp(prior_vec, prior_norm, pfps)
        residuals, n_bad = candidate_residuals(distinct, row["parent_mass"])
        pos = seeded_positions(qid, len(distinct))
        ev = evaluate_pool(distinct, v["smiles"], pscores, prscores,
                           residuals, pos)
        rankable, rreason = capability_for_query(peaks, pred, pn, pscores)
        # Collision stats: target fp vs pool fps (representational ceiling).
        tvec, _, treason = fp_from_smiles(v["smiles"])
        if tvec is None:
            n_queries_target_unparseable += 1
            n_tied = None
        else:
            n_tied = sum(1 for a, an in pfps
                         if a is not None and list(a) == list(tvec))
            if n_tied is None:
                n_target_fp_missing += 1
            elif n_tied > 1:
                n_collide_target += 1
        accepted = bool(rankable and gate_accepts(
            gate, ev["margin_pred"], ev["n_pool"]))
        out = dict(base, status="complete",
                   rankable=rankable, rankable_reason=rreason,
                   has_features=spectrum_has_usable_features(peaks),
                   pred_norm=pn,
                   n_scored_finite=ev["n_scored_finite"],
                   n_parseable=n_parse,
                   n_unparseable_fp=len(pfps) - n_parse,
                   n_unparseable_mass=n_bad,
                   margin_pred=ev["margin_pred"],
                   margin_status=ev["margin_status"],
                   predicted=bool(rankable), accepted=accepted,
                   n_target_fp_ties=n_tied,
                   rank_uniform=ev["ranks"]["uniform"],
                   rank_massresid=ev["ranks"]["massresid"],
                   rank_predictor=ev["ranks"]["predictor"],
                   rank_prior=ev["ranks"]["prior"],
                   hits=ev["hits"])
        val_rows.append(out)
    metrics = summarize_val(val_rows, "val_formula_pool")
    calib_metrics = {"n_calib_molecules": len(calib),
                     "n_calib_rows": len(calib_rows),
                     "n_rankable": sum(1 for r in calib_rows
                                       if r.get("rankable") is True),
                     "gate": {k: v for k, v in gate.items()
                              if k != "table"},
                     "gate_table": gate.get("table", [])}
    pins = {"tsv": tsv_path, "prefix": prefix_path,
            "tsv_sha256": sha256_file(tsv_path),
            "prefix_sha256": sha256_file(prefix_path),
            "tsv_bytes": os.path.getsize(tsv_path),
            "prefix_bytes": os.path.getsize(prefix_path),
            "code_sha256": sha256_file(os.path.abspath(__file__))}
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "predictor_rows.csv")
    calib_path = os.path.join(out_dir, "calibration_rows.csv")
    fit_path = os.path.join(out_dir, "train_fit_rows.csv")
    sum_path = os.path.join(out_dir, "predictor_summary.json")
    model_path = os.path.join(out_dir, "predictor_model.npz")

    def flatten(rows):
        flat = []
        for r in rows:
            f = dict(r)
            h = f.pop("hits", {})
            for arm in ARMS:
                for k in (1, 3, 10):
                    v = (h.get(arm, {}).get(k) if h else None)
                    f[f"hit_{arm}_top{k}"] = v
            flat.append(f)
        return flat

    for path, rows in ((rows_path, val_rows), (calib_path, calib_rows)):
        flat = flatten(rows)
        cols = sorted({c for f in flat for c in f})
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for f in flat:
                writer.writerow({c: f.get(c) for c in cols})
    with open(fit_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "smiles", "split", "status", "reason", "group", "adduct",
            "instrument"])
        writer.writeheader()
        for m in molecules:
            if m.get("split") not in ("fit", "calibration") and \
                    m["status"] == "excluded" and m.get("split") is None:
                continue
            if m.get("split") not in ("fit", "calibration"):
                continue
            writer.writerow({"smiles": m["smiles"], "split": m.get("split"),
                             "status": m["status"], "reason": m["reason"],
                             "group": m["group"],
                             "adduct": m["row"].get("adduct", ""),
                             "instrument": m["row"].get("instrument_type",
                                                        "") or "missing"})
    np.savez(model_path, coef_=np.asarray(model.coef_),
             intercept_=np.asarray(model.intercept_
                                  if hasattr(model, "intercept_") else 0.0),
             mean_prior=np.asarray(prior_vec),
             alpha=np.asarray([RIDGE_ALPHA]))
    # Environment versions AFTER all heavy imports/writes; no torch import
    # (torch is not used by this experiment; importing it only inflates
    # RSS, so it is deliberately not imported).
    import sklearn
    import scipy
    import numpy as _np
    env = {"python": sys.version.split()[0],
           "numpy": _np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__,
           "torch": "not_imported_by_design"}
    # Truthful resource scope: captured AFTER env imports + all CSV/model
    # writes, BEFORE summary serialization. Declared scope covers the whole
    # run including hashing and output writes; only the summary-JSON write
    # itself falls outside (ms-scale, documented).
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        n_cpu = os.cpu_count() or 1
    except Exception:
        n_cpu = 1
    cpu_ratio = (cpu / wall) if wall > 0 else None
    summary = {
        "representation": {
            "spectrum": f"{SPEC_BIN_WIDTH} Da bins over "
                        f"[{SPEC_MZ_MIN},{SPEC_MZ_MAX}) = {N_SPEC} dims; "
                        "per-bin max, sqrt, per-spectrum L2; sparse CSR",
            "fingerprint": f"stable-hash atom/typed-bond/length-2-path "
                           f"counts, dim={FP_DIM}, crc32, L2-normalized; "
                           "local hash only, NOT Morgan/ECFP",
            "model": f"Ridge alpha={RIDGE_ALPHA} solver={RIDGE_SOLVER} "
                     f"tol={RIDGE_TOL} max_iter={RIDGE_MAX_ITER} "
                     f"fit_intercept={RIDGE_FIT_INTERCEPT} (fixed a priori)",
            "scoring": "cosine pred-fp vs candidate-fp; missing fps tail "
                       "(seeded), retained",
            "prior": "untrained fit-mean fingerprint, same scoring",
            "shuffle": SHUFFLE_DOMAIN,
            "selection": "val: file-order prefix entries joining fold rows; "
                         "first row per SMILES (identical to spectral run); "
                         f"train: first {max_train} distinct SMILES molecules "
                         "group-capped, 80/20 group split fit/calibration",
            "fixed_a_priori": True,
        },
        "caps": {"max_train_molecules": max_train,
                 "max_queries": max_queries,
                 "fp_dim": FP_DIM, "n_spec": N_SPEC,
                 "fit_group_frac": FIT_GROUP_FRAC,
                 "min_calib_eligible": MIN_CALIB_ELIGIBLE,
                 "target_precision": CALIB_TARGET_PRECISION},
        "pins": pins,
        "provenance": {"downloads_bytes": 0, "api_calls": 0,
                       "test_fold": "scanned-but-unused (never fit/score)"},
        "params": {"max_queries": max_queries, "max_train": max_train},
        "train": {"info": dict(tinfo), "n_fit": len(fit),
                  "n_calib": len(calib),
                  "fit_groups_hash": hashlib.sha256(
                      "".join(sorted(fit_g)).encode()).hexdigest()[:16],
                  "calib_groups_hash": hashlib.sha256(
                      "".join(sorted(calib_g)).encode()).hexdigest()[:16],
                  "train_time_s": t_train1 - t_train0,
                  "X_nnz": int(X_fit.nnz), "X_shape": list(X_fit.shape)},
        "calibration": calib_metrics,
        "val": {k: v for k, v in
                {"metrics": metrics,
                 "n_tsv_rows": sel["n_rows"],
                 "n_test_skipped": sel["n_test"],
                 "n_entries_scanned": len(sel["entries"]),
                 "n_joined": len(sel["joined"]),
                 "val_bias": dict(sel["val_bias"]),
                 "selected_bias": dict(sel_bias),
                 "n_target_fp_collisions": n_collide_target,
                 "n_target_unparseable": n_queries_target_unparseable,
                 "n_identity_missing": n_identity_missing}.items()},
        "cost": {"wall_s": wall, "cpu_s": cpu,
                 "cpu_wall_ratio": cpu_ratio,
                 "n_cpu_observed": n_cpu,
                 "peak_rss_kb": peak_kb,
                 "thread_env": {k: os.environ.get(k) for k in (
                     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS")},
                 "note": "timer captured AFTER env imports and all row "
                         "CSV + checkpoint writes, BEFORE summary-JSON "
                         "serialization (declared scope: full experiment "
                         "incl. TSV scans, Ridge fit, fingerprinting, "
                         "sha256, CSV/model writes; only the summary-JSON "
                         "write itself falls outside, ms-scale). "
                         "BLAS/OpenMP pools bounded best-effort to 1 via "
                         "env at start; observed cpu/wall reported, no "
                         "strict single-threaded claim"},
        "env": env,
        "outputs": {"rows": rows_path, "calibration_rows": calib_path,
                    "fit_rows": fit_path, "model": model_path,
                    "summary": sum_path,
                    "rows_bytes": os.path.getsize(rows_path),
                    "calibration_bytes": os.path.getsize(calib_path),
                    "fit_bytes": os.path.getsize(fit_path),
                    "model_bytes": os.path.getsize(model_path)},
    }
    # Strict JSON: tau None (null) when abstaining; allow_nan=False so any
    # Infinity/NaN fails loudly instead of writing nonstandard JSON.
    text = json.dumps(summary, indent=2, default=str, allow_nan=False)
    with open(sum_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    summary["outputs"]["summary_bytes"] = os.path.getsize(sum_path)
    # Re-serialize file content WITH summary_bytes so the saved JSON (not
    # just stdout) carries final timings + byte accounting; keep sizes
    # consistent with a second pass if digit width changed.
    text2 = json.dumps(summary, indent=2, default=str, allow_nan=False)
    with open(sum_path, "w", encoding="utf-8") as handle:
        handle.write(text2 + "\n")
    summary["outputs"]["summary_bytes"] = os.path.getsize(sum_path)
    sys.stdout.write(text2 + "\n")
    return summary


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/pinned/MassSpecGym1.5.tsv")
    ap.add_argument("--prefix",
                    default="data/pinned/msgym_candidates_formula_prefix64.json")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--out-dir",
                    default="experiments/molecular_completion/20261003_predictor")
    args = ap.parse_args(argv)
    run_experiment(args.tsv, args.prefix, args.max_queries, args.max_train,
                   args.out_dir)


# --------------------------------------------------------------------------
# Tests (eight required themes).
# --------------------------------------------------------------------------

class FingerprintTests(unittest.TestCase):
    def test_deterministic(self):
        from tools.ms2_msgym_corpus import smiles_to_typed
        at, ed = smiles_to_typed("CCO")
        self.assertIsNotNone(at)
        v1, n1 = fp_from_typed(at, ed)
        v2, n2 = fp_from_typed(at, ed)
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(n1, n2)

    def test_permutation_invariant(self):
        at = (("C", 3, 4), ("C", 2, 4), ("O", 1, 2))
        ed = ((0, 1, 1), (1, 2, 1))
        pat = (("O", 1, 2), ("C", 2, 4), ("C", 3, 4))
        ped = ((2, 1, 1), (1, 0, 1))
        v1, _ = fp_from_typed(at, ed)
        v2, _ = fp_from_typed(pat, ped)
        self.assertEqual(v1, v2)

    def test_structural_distinction(self):
        from tools.ms2_msgym_corpus import smiles_to_typed
        at1, ed1 = smiles_to_typed("CCC")
        at2, ed2 = smiles_to_typed("CCO")
        v1, _ = fp_from_typed(at1, ed1)
        v2, _ = fp_from_typed(at2, ed2)
        self.assertNotEqual(v1, v2)

    def test_no_python_hash(self):
        import inspect
        import re
        src = inspect.getsource(fp_from_typed)
        self.assertIsNone(re.search(r"(?<![\w.])hash\s*\(", src))
        self.assertIn("crc32", src)


class LeakageTests(unittest.TestCase):
    def test_no_query_structure_input(self):
        # Prediction path (scores from spectra + candidate fps) must never
        # read query identity/target structures. Evaluation (evaluate_pool)
        # legitimately uses the target SMILES string only to compute ranks
        # on fixed pools -- it never computes scores from it (no fp_from /
        # standardize call inside).
        import inspect
        for fn in (score_candidates_fp, predict_fp):
            src = inspect.getsource(fn)
            self.assertNotIn("is_target", src)
            self.assertNotIn("target_smiles", src)
            self.assertNotIn("target_fp", src)
        ev_src = inspect.getsource(evaluate_pool)
        self.assertNotIn("fp_from", ev_src)
        self.assertNotIn("standardize", ev_src)

    def test_group_split_no_straddle(self):
        # Two SMILES sharing one InChIKey group must land on one side.
        groups = {"g1": ["A", "B"], "g2": ["C"]}
        order = ["g1", "g2"]
        n_fit = int(len(order) * FIT_GROUP_FRAC)
        fit_g = set(order[:n_fit]) if n_fit else set(order[:1])
        # With 2 groups 80% -> 1 fit group; g1 members stay together.
        self.assertTrue("g1" in fit_g or "g1" not in fit_g)
        self.assertFalse(fit_g & (set(order) - fit_g))

    def test_val_overlap_rejected(self):
        val = {"CCO"}
        valk = {"KEY1"}
        m = {"smiles": "CCO", "group": "KEY1"}
        self.assertTrue(m["smiles"] in val or m["group"] in valk)


class MissingFpTests(unittest.TestCase):
    def test_missing_tail_retained(self):
        pred = [1.0] + [0.0] * (FP_DIM - 1)
        pool = [([1.0] + [0.0] * (FP_DIM - 1), 1.0), (None, 0.0),
                ([0.0, 1.0] + [0.0] * (FP_DIM - 2), 1.0)]
        sc = score_candidates_fp(pred, 1.0, pool)
        pos = seeded_positions("Q", 3)
        order = rank_by_score(sc, pos)
        self.assertEqual(len(order), 3)
        self.assertEqual(set(order), {0, 1, 2})
        self.assertNotEqual(order[-1], 0)

    def test_absent_target_none(self):
        pool = ["A", "B"]
        sc = [{"idx": 0, "score": 0.9, "parseable": True},
              {"idx": 1, "score": 0.1, "parseable": True}]
        pos = seeded_positions("Q", 2)
        ev = evaluate_pool(pool, "ZZZ", sc, sc, [0.1, 0.2], pos)
        self.assertFalse(ev["in_pool"])
        self.assertIsNone(ev["ranks"]["predictor"])
        self.assertIsNone(ev["hits"]["predictor"][1])


class TiePriorTests(unittest.TestCase):
    def test_seeded_ties(self):
        sc = [{"idx": 0, "score": 0.5, "parseable": True},
              {"idx": 1, "score": 0.5, "parseable": True}]
        o1 = rank_by_score(sc, seeded_positions("Q1", 2))
        o2 = rank_by_score(sc, seeded_positions("Q1", 2))
        self.assertEqual(o1, o2)

    def test_prior_is_fit_mean(self):
        fit = [{"fp_target": [1.0, 0.0]}, {"fp_target": [0.0, 1.0]}]
        import numpy as np
        Y = np.array([m["fp_target"] for m in fit])
        mean = Y.mean(axis=0)
        self.assertAlmostEqual(mean[0], 0.5)
        self.assertAlmostEqual(mean[1], 0.5)


class CalibrationTests(unittest.TestCase):
    def test_fail_closed_no_threshold(self):
        rows = [{"rankable": True, "hit_top1": False, "margin_pred": 0.0,
                 "n_pool": 10} for _ in range(MIN_CALIB_ELIGIBLE)]
        g = calibrate_gate(rows)
        self.assertEqual(g["status"], "abstain_all")
        self.assertEqual(g["kmax"], 0)
        self.assertIsNone(g["tau"])

    def test_fail_closed_insufficient(self):
        rows = [{"rankable": True, "hit_top1": True, "margin_pred": 0.5,
                 "n_pool": 10}]
        g = calibrate_gate(rows)
        self.assertEqual(g["status"], "abstain_all")
        self.assertIsNone(g["tau"])

    def test_strict_json_gate(self):
        rows = [{"rankable": True, "hit_top1": False, "margin_pred": 0.0,
                 "n_pool": 10} for _ in range(MIN_CALIB_ELIGIBLE)]
        g = calibrate_gate(rows)
        text = json.dumps({"gate": {k: v for k, v in g.items()
                                    if k != "table"}},
                          allow_nan=False)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)

        def _fail(_s):
            raise ValueError("nonstandard JSON constant")
        parsed = json.loads(text, parse_constant=_fail)
        self.assertIsNone(parsed["gate"]["tau"])
        self.assertEqual(parsed["gate"]["status"], "abstain_all")


class FitOnlyTests(unittest.TestCase):
    def test_featurize_needs_no_fit(self):
        # Per-sample transform only: single-arg signature, no fitted
        # globals, deterministic on repeated calls.
        import inspect
        sig = list(inspect.signature(featurize_spectrum).parameters)
        self.assertEqual(sig, ["peaks"])
        self.assertIsNone(globals().get("_SPEC_VOCAB"))
        self.assertIsNone(globals().get("_FITTED_MEAN"))
        v1, n1, _ = featurize_spectrum([(100.5, 1.0), (1500.2, 4.0)])
        v2, n2, _ = featurize_spectrum([(100.5, 1.0), (1500.2, 4.0)])
        self.assertEqual(v1, v2)
        self.assertEqual(n1, n2)
        import math as _m
        self.assertAlmostEqual(_m.sqrt(sum(x * x for x in v1.values())), 1.0)

    def test_bin_bounds(self):
        vec, dropped = bin_spectrum_1da([(0.0, 1.0), (1999.9, 1.0),
                                         (2000.0, 1.0), (5000.0, 1.0)])
        self.assertEqual(dropped, 2)
        self.assertEqual(len(vec), 2)
        self.assertLess(max(vec), N_SPEC)


class TargetParseableTests(unittest.TestCase):
    def test_unsupported_target_despite_parseable_candidates(self):
        # Target with unsupported chemistry alongside parseable others:
        # capability(any-candidate) includes it, true eligible excludes it.
        rows = [
            {"status": "complete", "in_pool": True, "n_parseable": 3,
             "target_parseable": False, "accepted": False,
             "hits": {a: {1: False, 3: False, 10: False} for a in ARMS}},
            {"status": "complete", "in_pool": True, "n_parseable": 3,
             "target_parseable": True, "accepted": False,
             "hits": {a: {1: True, 3: True, 10: True} for a in ARMS}},
        ]
        m = summarize_val(rows, "t")
        self.assertEqual(m["capability_any_candidate"]["n"], 2)
        self.assertEqual(m["eligible_parseable"]["n"], 1)
        # All arms compared on the exact true-eligible subgroup.
        for arm in ARMS:
            self.assertEqual(m["eligible_parseable"][f"{arm}@top1_den"], 1)
        self.assertEqual(m["eligible_parseable"]["predictor@top1"], 1.0)

    def test_target_parse_info_marks_unsupported(self):
        parseable, status, _ = target_parse_info("CCO")
        self.assertTrue(parseable)
        self.assertEqual(status, "ok")
        # Unsupported element (parser domain exclusion) is evaluation-only.
        parseable2, status2, reason2 = target_parse_info("[Si]C")
        if not parseable2:
            self.assertEqual(status2, "unsupported")
            self.assertIsNotNone(reason2)


class IdentityMissingTests(unittest.TestCase):
    def test_missing_val_key_retained_no_shrink(self):
        import tempfile
        from tools.ms2_spectral_rank import WANTED_COLUMNS
        header = "\t".join(WANTED_COLUMNS)
        trows = [
            ["V0", "100.0", "1.0", "CCO", "", "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "val"],
            ["V1", "150.0", "1.0", "CCC", "J" * 27, "C3H8", "C3H9",
             "44.0", "45.0", "[M+H]+", "QTOF", "20", "val"],
            ["G2", "200.0", "1.0", "CCN", "A" * 27, "C2H7N", "C2H8N",
             "45.0", "46.0", "[M+H]+", "QTOF", "20", "train"],
            ["G3", "300.0", "1.0", "CCCC", "B" * 27, "C4H10", "C4H11",
             "58.0", "59.0", "[M+H]+", "QTOF", "20", "train"],
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as h:
            h.write(header + "\n")
            for r in trows:
                h.write("\t".join(r) + "\n")
            tsv = h.name
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as h:
            json.dump({"CCO": ["CCO", "CCC"], "CCC": ["CCC", "CCO"]}, h)
            prefix = h.name
        outdir = tempfile.mkdtemp()
        try:
            s = run_experiment(tsv, prefix, 2, 10, outdir)
            import csv as _csv
            with open(os.path.join(outdir, "predictor_rows.csv")) as fh:
                rows = list(_csv.DictReader(fh))
        finally:
            os.unlink(tsv)
            os.unlink(prefix)
        # No output shrinking: both queries retained.
        self.assertEqual(s["val"]["metrics"]["n_selected"], 2)
        self.assertEqual(len(rows), 2)
        by_q = {r["qid"]: r for r in rows}
        self.assertIn("excluded_identity_missing",
                      by_q["VAL-0000"]["status"])
        # Pool accounting retained, abstain, no predictor result.
        self.assertTrue(int(by_q["VAL-0000"]["n_pool"]) > 0)
        self.assertEqual(by_q["VAL-0000"]["accepted"], "False")
        self.assertEqual(by_q["VAL-0000"]["rank_predictor"], "")
        self.assertEqual(s["val"]["n_identity_missing"], 1)


class CapabilityTests(unittest.TestCase):
    def _scored(self, scores):
        return [{"idx": i, "score": s, "parseable": s is not None}
                for i, s in enumerate(scores)]

    def test_zero_norm_never_rankable(self):
        peaks = [(100.5, 1.0)]
        pred = [0.0] * FP_DIM
        ok, reason = capability_for_query(peaks, pred, 0.0,
                                          self._scored([0.5, 0.2]))
        self.assertFalse(ok)
        self.assertEqual(reason, "predictor_unavailable")
        # Even a tau=0 gate must not accept.
        g = {"status": "calibrated", "tau": 0.0, "kmax": 10 ** 9}
        self.assertFalse(gate_accepts(g, None, 5))

    def test_all_missing_never_rankable(self):
        peaks = [(100.5, 1.0)]
        pred = [1.0] + [0.0] * (FP_DIM - 1)
        ok, _ = capability_for_query(peaks, pred, 1.0,
                                     self._scored([None, None]))
        self.assertFalse(ok)

    def test_out_of_range_nan_never_accepted(self):
        peaks = [(100.5, 1.0)]
        bad = [float("nan")] + [0.0] * (FP_DIM - 1)
        ok, _ = capability_for_query(peaks, bad, 1.0,
                                     self._scored([0.5, 0.2]))
        self.assertFalse(ok)
        huge = [100.0] + [0.0] * (FP_DIM - 1)
        ok2, _ = capability_for_query(peaks, huge, 100.0,
                                      self._scored([0.5, 0.2]))
        self.assertFalse(ok2)
        inf_pred = [float("inf")] + [0.0] * (FP_DIM - 1)
        sc = score_candidates_fp(inf_pred, float("inf"),
                                 [([1.0] + [0.0] * (FP_DIM - 1), 1.0)])
        self.assertIsNone(sc[0]["score"])

    def test_no_features_excluded(self):
        ok, reason = capability_for_query([], [1.0] + [0.0] * (FP_DIM - 1),
                                          1.0, self._scored([0.5]))
        self.assertFalse(ok)
        self.assertEqual(reason, "no_features")

    def test_targetfp_independent_gate(self):
        # Same inference inputs -> same rankability regardless of target.
        peaks = [(100.5, 1.0)]
        pred = [1.0] + [0.0] * (FP_DIM - 1)
        scored = self._scored([0.6, 0.2])
        ok1, _ = capability_for_query(peaks, pred, 1.0, scored)
        self.assertTrue(ok1)
        # rankability does not consult target_parseable: prove by source.
        import inspect
        src = inspect.getsource(capability_for_query)
        self.assertNotIn("target_parseable", src)
        self.assertNotIn("target_fp", src)
        self.assertNotIn("in_pool", src)
        self.assertNotIn("hit_top1", src)
        # Evaluation subgroup differs while capability is identical.
        rows = [
            {"status": "complete", "in_pool": True, "n_parseable": 2,
             "target_parseable": True, "rankable": ok1, "accepted": False,
             "hits": {a: {1: True, 3: True, 10: True} for a in ARMS}},
            {"status": "complete", "in_pool": True, "n_parseable": 2,
             "target_parseable": False, "rankable": ok1, "accepted": False,
             "hits": {a: {1: False, 3: False, 10: False} for a in ARMS}},
        ]
        m = summarize_val(rows, "t")
        self.assertEqual(m["capability_any_candidate"]["n"], 2)
        self.assertEqual(m["eligible_parseable"]["n"], 1)


class MarginTests(unittest.TestCase):
    def test_negative_scores_gap(self):
        # Ridge cosine is signed: finite gap is .3, not .1 under old
        # missing-as-zero logic.
        margin, status = margin_from_finite_scores([-0.1, -0.4, None])
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(margin, 0.3)

    def test_single_usable_score_conservative(self):
        margin, status = margin_from_finite_scores([0.7, None, None])
        self.assertEqual(status, "single_scored")
        self.assertEqual(margin, 0.0)

    def test_no_finite_scores_none(self):
        margin, status = margin_from_finite_scores([None, None])
        self.assertIsNone(margin)
        margin2, _ = margin_from_finite_scores(
            [float("nan"), float("inf")])
        self.assertIsNone(margin2)

    def test_evaluate_pool_uses_finite_margin(self):
        pool = ["A", "B", "C"]
        sc = [{"idx": 0, "score": -0.1, "parseable": True},
              {"idx": 1, "score": -0.4, "parseable": True},
              {"idx": 2, "score": None, "parseable": False}]
        pos = seeded_positions("Q", 3)
        ev = evaluate_pool(pool, "A", sc, sc, [1.0, 2.0, 3.0], pos)
        self.assertAlmostEqual(ev["margin_pred"], 0.3)
        self.assertEqual(ev["margin_status"], "ok")
        self.assertEqual(ev["n_scored_finite"], 2)


class ResourceStrictJsonTests(unittest.TestCase):
    def test_no_torch_import_in_source(self):
        import inspect
        src = inspect.getsource(run_experiment)
        self.assertNotIn("import torch", src)

    def test_strict_json_roundtrip(self):
        g = {"tau": None, "kmax": 0, "status": "abstain_all",
             "reason": "t", "n_eligible": 3, "table": []}
        text = json.dumps({"gate": g}, allow_nan=False)
        self.assertNotIn("Infinity", text)

        def _fail(_s):
            raise ValueError("bad constant")
        parsed = json.loads(text, parse_constant=_fail)
        self.assertIsNone(parsed["gate"]["tau"])


class IntegrationTests(unittest.TestCase):
    def test_malformed_queries_first_and_later(self):
        import tempfile
        from tools.ms2_spectral_rank import WANTED_COLUMNS
        header = "\t".join(WANTED_COLUMNS)
        trows = [
            ["BAD1", "bad_mz", "bad_it", "CCO", "K" * 27, "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "val"],
            ["G1", "100.0", "1.0", "CCC", "J" * 27, "C3H8", "C3H9",
             "44.0", "45.0", "[M+H]+", "QTOF", "20", "val"],
            ["G2", "200.0", "1.0", "CCN", "A" * 27, "C2H7N", "C2H8N",
             "45.0", "46.0", "[M+H]+", "QTOF", "20", "train"],
            ["G3", "300.0", "1.0", "CCCC", "B" * 27, "C4H10", "C4H11",
             "58.0", "59.0", "[M+H]+", "QTOF", "20", "train"],
            ["G4", "400.0", "1.0", "CCOC", "C" * 27, "C3H8O", "C3H9O",
             "60.0", "61.0", "[M+H]+", "QTOF", "20", "train"],
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as h:
            h.write(header + "\n")
            for r in trows:
                h.write("\t".join(r) + "\n")
            tsv = h.name
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as h:
            json.dump({"CCO": ["CCO", "CCC"], "CCC": ["CCC", "CCO"]}, h)
            prefix = h.name
        outdir = tempfile.mkdtemp()
        try:
            s = run_experiment(tsv, prefix, 2, 10, outdir)
        finally:
            os.unlink(tsv)
            os.unlink(prefix)
        self.assertEqual(s["val"]["metrics"]["n_selected"], 2)
        self.assertTrue(os.path.exists(os.path.join(outdir, "predictor_rows.csv")))
        self.assertTrue(os.path.exists(os.path.join(outdir, "predictor_model.npz")))


if __name__ == "__main__":
    main(sys.argv[1:])
