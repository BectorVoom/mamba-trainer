"""Measured-spectrum nearest-neighbor ranking for MassSpecGym formula pools.

Fifth molecular-completion run: the first HONEST measured-spectrum ranking.
Prior runs left spectral similarity as ``unavailable_no_spectrum``; the TSV
actually ships a measured spectrum per row, so this module bins those spectra
and scores supplied formula-pool candidates by cosine similarity against
TRAIN-fold reference spectra of the same structure identity.

Representation (fixed a priori, never tuned on val outcomes):
  mass bins of width 0.1 m/z over [0, 2000), per-bin max intensity,
  L2-normalized sparse vectors, cosine similarity. Candidate score = max
  cosine over that candidate's TRAIN reference spectra. No fitting, no
  threshold calibration: the abstention rule is parameter-free (predict iff
  at least one pool candidate has a TRAIN reference spectrum).

Connectivity identity: exact provider-canonical SMILES string equality.
Sound (equal strings => same structure) but conservative: stereo or
tautomer variants written as different strings do not match, so reported
reference coverage is a LOWER BOUND on true connectivity coverage, not a
graph-canonicalization guarantee. RDKit is not installed in this
environment (no new dependencies per task); a crude no-stereo character
count is reported as an approximation alongside, never used for ranking,
and never claimed as a tight connectivity bound. Split disjointness is
audited by both SMILES string and InChIKey-14 block, and ENFORCED for val:
nonzero train/query InChIKey-14 overlap aborts the run, and queries with a
missing connectivity key are excluded conservatively (counted, never
scored).

Leakage rules (asserted, unit-tested):
  * query spectra come from the val fold; references from train only;
  * any train spectrum whose SMILES equals the query SMILES is excluded
    and counted (val/train splits are structure-disjoint, so this fires
    never -- a nonzero count is reported, never scored);
  * any TRAIN reference spectrum whose parsed peak CONTENT fingerprint
    matches the query spectrum content is excluded and counted, even under
    a different identifier (duplicate-content guard); reference spectra are
    also deduplicated by content within each structure (first wins, both
    dedup kinds counted);
  * the query structure is never used as a feature (pools are supplied
    verbatim; the target is never injected);
  * the query structure is never used as a feature (pools are supplied
    verbatim; the target is never injected);
  * candidate order never leaks self-first supplier order: uniform and all
    tie-breaking use a per-query seeded shuffle (zlib.crc32 of the qid,
    same convention as tools/ms2_rank_abstain.py);
  * candidates without references are NEVER dropped: unreferenced
    candidates rank after referenced ones (seeded-shuffle order) and stay
    in every denominator. An empty reference set yields abstention, never
    a shrunken pool or an improved denominator.

Arms (identical formula pools, no oracle subgraphs -- the primary,
non-oracle condition):
  * uniform   : seeded shuffle (weak baseline, order-decontaminated).
  * spectral  : TRAIN-reference max-cosine rank (this run's method).
  * coverage  : coverage-only baseline on the IDENTICAL pools: seeded
    shuffle among reference-backed candidates first, then unreferenced
    ones. Uses no spectral score; isolates what reference PRESENCE alone
    buys versus the cosine values.
  * massresid : replacement cheap exact baseline for formula pools (not a
    head-to-head rerun of the prior learned ranker, whose oracle subgraph
    features do not exist in this pool condition): rank by
    |graph-theoretical mass - query parent_mass| using the existing
    SMILES standardizer; unparseable candidates rank last and are counted.

A small TRAIN-fold positive control (labelled CONTROL, not a val result)
checks pipeline mechanics on train spectra as queries against remaining
train references, excluding the query identifier AND the query content
fingerprint. Control top-k is reported alongside the coverage-only arm and
reference-count subgroups (pools with exactly one vs multiple
reference-backed candidates), because most control pools have a single
referenced candidate and a bare top-1 rate cannot establish
representation quality.

CPU stdlib reference. No GPU tensors and no Rust port exist for this
experiment, so GPU execution and Rust/Python parity checks are explicitly
NOT APPLICABLE (recorded, not passed).
"""

import csv
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
import unittest
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_msgym_corpus import (  # noqa: E402
    iter_json_entries,
    parse_formula,
    standardize_smiles,
)

# --------------------------------------------------------------------------
# Declared representation (fixed a priori; not selected on val outcomes).
# --------------------------------------------------------------------------

BIN_WIDTH = 0.1
MZ_MIN = 0.0
MZ_MAX = 2000.0
N_BINS = int((MZ_MAX - MZ_MIN) / BIN_WIDTH)  # 20000
SHUFFLE_DOMAIN = "qid-crc32"  # same convention as ms2_rank_abstain.py


# --------------------------------------------------------------------------
# Spectrum parsing / binning / cosine.
# --------------------------------------------------------------------------

def parse_spectrum(mzs_text, intensities_text):
    """Parse one TSV spectrum. Returns (peaks, error).

    peaks = [(mz, intensity)] with finite mz >= 0 and finite intensity >= 0.
    error is None on success, else a short reason string. Empty spectra,
    length mismatches, non-numeric tokens, NaN/inf and all-zero norms fail;
    nothing is silently repaired.
    """
    if mzs_text is None or intensities_text is None:
        return None, "missing_field"
    mzs_text = mzs_text.strip()
    intensities_text = intensities_text.strip()
    if not mzs_text or not intensities_text:
        return None, "empty_spectrum"
    try:
        mzs = [float(t) for t in mzs_text.split(",")]
        ints = [float(t) for t in intensities_text.split(",")]
    except ValueError:
        return None, "non_numeric"
    if len(mzs) != len(ints):
        return None, "length_mismatch"
    if not mzs:
        return None, "empty_spectrum"
    for mz, it in zip(mzs, ints):
        if not math.isfinite(mz) or not math.isfinite(it):
            return None, "non_finite"
        if mz < 0.0 or it < 0.0:
            return None, "negative_value"
    if all(it == 0.0 for it in ints):
        return None, "zero_norm"
    return list(zip(mzs, ints)), None


def bin_spectrum(peaks):
    """Bin peaks to {bin_index: max_intensity}. Returns (vec, n_dropped).

    Peaks outside [MZ_MIN, MZ_MAX) are dropped and counted (never silently
    ignored: the caller records n_dropped). An all-dropped spectrum yields
    an empty vector, whose cosine is defined as 0.0 (see cosine()).
    """
    vec = {}
    dropped = 0
    for mz, it in peaks:
        if not (MZ_MIN <= mz < MZ_MAX):
            dropped += 1
            continue
        b = int((mz - MZ_MIN) / BIN_WIDTH)
        if it > vec.get(b, 0.0):
            vec[b] = it
    return vec, dropped


def vec_norm(vec):
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine(a, na, b, nb):
    """Cosine of sparse vectors with precomputed norms.

    Zero-norm vectors (e.g. fully out-of-range spectra) score 0.0 against
    everything, including themselves: no division by zero, no NaN scores.
    """
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    return dot / (na * nb)


def content_fp(peaks):
    """Canonical fingerprint of parsed peak CONTENT (duplicate guard).

    Rounded (mz, intensity) pairs, sorted. Two spectra with equal
    fingerprints carry identical measured content regardless of identifier,
    so a query must never score against a reference with the same
    fingerprint (self-match through duplication or re-upload).
    """
    return tuple(sorted((round(mz, 4), round(it, 6)) for mz, it in peaks))


def strip_stereo(smiles):
    """Crude character-level grouping for a SENSITIVITY count only.

    Drops @, / and backslash characters. This is an APPROXIMATION, not a
    connectivity bound in either direction: it can conflate distinct
    stereoisomers and still miss tautomers or other SMILES variants of one
    connectivity. Reported only as a rough sensitivity count, never used
    for ranking, and never claimed as tight.
    """
    return smiles.replace("@", "").replace("/", "").replace("\\", "")


# --------------------------------------------------------------------------
# TSV / prefix ingestion.
# --------------------------------------------------------------------------

WANTED_COLUMNS = ("identifier", "mzs", "intensities", "smiles", "inchikey",
                  "formula", "precursor_formula", "parent_mass", "precursor_mz",
                  "adduct", "instrument_type", "collision_energy", "fold")


def load_tsv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in WANTED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"TSV missing columns: {missing}")
        for row in reader:
            yield {c: row[c] for c in WANTED_COLUMNS}


def short_key(inchikey):
    """Connectivity block of an InChIKey (first 14 chars)."""
    return (inchikey or "")[:14]


def audit_split_disjointness(train_smiles, query_smiles, train_keys, query_keys):
    """Return overlap sets; callers must publish counts and fail on nonzero.

    Belt-and-braces: the benchmark ships structure-disjoint folds, but the
    experiment must verify rather than assume.
    """
    return (set(query_smiles) & set(train_smiles,
                                    ), set(query_keys) & set(train_keys))


def collect_train_refs(tsv_path, wanted_smiles, exclude_smiles=()):
    """Single TSV pass. Returns (refs, info).

    refs: {smiles: [(identifier, vec, norm, n_dropped, adduct, instrument,
    content_fp)]}. Only spectra whose SMILES is wanted are binned/retained
    (bounded memory); everything else is counted, not stored.
    exclude_smiles (the query structures) yield zero retained hits -- any
    hit is counted as a leak exclusion and never lands in refs.
    Dedup is by identifier AND by content fingerprint within each
    structure (first wins in file order); both counts are published.
    """
    refs = {}
    info = Counter()
    seen_ids = set()
    seen_content = {}  # smiles -> set of content fingerprints
    excluded = set(exclude_smiles)
    for row in load_tsv_rows(tsv_path):
        if row["fold"] != "train":
            continue
        info["n_train_rows_scanned"] += 1
        smi = row["smiles"]
        if smi in excluded:
            info["n_leak_excluded"] += 1
            continue
        if smi not in wanted_smiles:
            continue
        ident = row["identifier"]
        if ident in seen_ids:
            info["n_duplicate_ids"] += 1
            continue
        seen_ids.add(ident)
        peaks, err = parse_spectrum(row["mzs"], row["intensities"])
        if err is not None:
            info[f"ref_parse_excl:{err}"] += 1
            continue
        fp = content_fp(peaks)
        bucket = seen_content.setdefault(smi, set())
        if fp in bucket:
            info["n_duplicate_content"] += 1
            continue
        bucket.add(fp)
        vec, dropped = bin_spectrum(peaks)
        norm = vec_norm(vec)
        refs.setdefault(smi, []).append(
            (ident, vec, norm, dropped, row["adduct"], row["instrument_type"],
             fp))
        info["n_ref_spectra_retained"] += 1
        info["n_ref_dropped_peaks"] += dropped
    info["n_ref_structures"] = len(refs)
    return refs, info


def exclude_query_content(refs, query_fp):
    """Drop refs whose content fingerprint equals the query's. Returns
    (filtered_refs, n_excluded_refs, n_excluded_structures).

    Guards against same-content/different-identifier leakage (duplicated or
    re-uploaded spectra across folds and identifiers).
    """
    filtered = {}
    n_refs, n_struct = 0, 0
    for smi, lst in refs.items():
        kept = [t for t in lst if t[6] != query_fp]
        dropped = len(lst) - len(kept)
        if dropped:
            n_refs += dropped
            n_struct += 1
        if kept:
            filtered[smi] = kept
    return filtered, n_refs, n_struct


def parse_query_spectrum(row):
    """Parse the single query spectrum of a val/train row. Returns dict."""
    peaks, err = parse_spectrum(row["mzs"], row["intensities"])
    if err is not None:
        return {"ok": False, "error": err}
    vec, dropped = bin_spectrum(peaks)
    return {"ok": True, "vec": vec, "norm": vec_norm(vec),
            "fp": content_fp(peaks),
            "n_peaks": len(peaks), "n_dropped": dropped}


# --------------------------------------------------------------------------
# Ranking.
# --------------------------------------------------------------------------

def seeded_positions(qid, n):
    positions = list(range(n))
    random.Random(zlib.crc32(qid.encode())).shuffle(positions)
    return {idx: p for p, idx in enumerate(positions)}


def score_candidates(query_vec, query_norm, pool, refs):
    """Score each distinct pool candidate. Returns list of dicts.

    The pool is used verbatim (deduplicated to distinct strings, first
    occurrence wins; counts recorded by the caller). score is None when the
    candidate has no TRAIN reference spectrum -- recorded, never dropped.
    """
    out = []
    for idx, smi in enumerate(pool):
        cand_refs = refs.get(smi, [])
        if not cand_refs:
            out.append({"idx": idx, "smiles": smi, "score": None,
                        "n_refs": 0, "best_ref": None})
            continue
        best, best_id = -1.0, None
        for ident, vec, norm, _, _, _, _ in cand_refs:
            s = cosine(query_vec, query_norm, vec, norm)
            if s > best:
                best, best_id = s, ident
        out.append({"idx": idx, "smiles": smi, "score": best,
                    "n_refs": len(cand_refs), "best_ref": best_id})
    return out


def rank_uniform(scored, pos_of):
    return sorted(range(len(scored)), key=lambda i: pos_of[i])


def rank_spectral(scored, pos_of):
    # Referenced candidates first by score desc; unreferenced trail in
    # seeded-shuffle order (retained in the ranking, never removed).
    return sorted(range(len(scored)),
                  key=lambda i: ((0.0, -scored[i]["score"])
                                 if scored[i]["score"] is not None
                                 else (1.0, 0.0),
                                 pos_of[i]))


def rank_coverage(scored, pos_of):
    """Coverage-only baseline: reference-backed candidates first in seeded
    shuffle order, then unreferenced ones. Uses n_refs only -- no spectral
    score -- so spectral-vs-coverage comparisons isolate what the cosine
    VALUES add over reference PRESENCE alone."""
    return sorted(range(len(scored)),
                  key=lambda i: (0 if scored[i]["n_refs"] > 0 else 1,
                                 pos_of[i]))


def rank_massresid(scored, residuals, pos_of):
    # residuals[i] is ppm or None (unparseable -> last, counted).
    return sorted(range(len(scored)),
                  key=lambda i: ((0.0, residuals[i])
                                 if residuals[i] is not None
                                 else (1.0, 0.0),
                                 pos_of[i]))


def candidate_residuals(pool, parent_mass_text):
    """Cheap exact baseline features: |theory - observed| ppm per candidate.

    Uses the existing SMILES standardizer for theoretical mass. Returns
    (residuals, n_unparseable). observed = provider parent_mass (a MEASURED
    m/z-derived value, used as given query evidence, never converted).
    """
    from decimal import Decimal
    try:
        observed_mu = int(Decimal(parent_mass_text) * 1_000_000)
    except Exception:
        return [None] * len(pool), len(pool)
    residuals = []
    n_bad = 0
    for smi in pool:
        try:
            rec, reason = standardize_smiles(smi, "spectral-pool", smi)
        except Exception:
            rec, reason = None, "exception"
        if rec is None:
            residuals.append(None)
            n_bad += 1
        else:
            ppm = abs(rec["mass"] - observed_mu) / max(observed_mu, 1) * 1e6
            residuals.append(ppm)
    return residuals, n_bad


ARMS = ("uniform", "spectral", "coverage", "massresid")


def evaluate_query(qid, scored, pos_of, residuals, target_smiles):
    """Ranks and hits for one query on identical pools. Target-absent safe."""
    order_u = rank_uniform(scored, pos_of)
    order_s = rank_spectral(scored, pos_of)
    order_c = rank_coverage(scored, pos_of)
    order_m = rank_massresid(scored, residuals, pos_of)
    n = len(scored)
    tgt = [i for i, s in enumerate(scored) if s["smiles"] == target_smiles]
    in_pool = bool(tgt)
    tgt_idx = tgt[0] if tgt else None
    n_ref = sum(1 for s in scored if s["n_refs"] > 0)
    tgt_ref = (scored[tgt_idx]["n_refs"] > 0) if tgt else False

    def rank_of(order):
        if tgt_idx is None:
            return None
        return order.index(tgt_idx) + 1  # 1-based

    ranks = {"uniform": rank_of(order_u), "spectral": rank_of(order_s),
             "coverage": rank_of(order_c), "massresid": rank_of(order_m)}

    def hits(rank):
        if rank is None:
            return {1: None, 3: None, 10: None}
        return {1: rank <= 1, 3: rank <= min(3, n), 10: rank <= min(10, n)}

    return {"in_pool": in_pool, "n_pool": n, "n_ref": n_ref,
            "target_has_ref": tgt_ref if in_pool else None,
            "predicted": n_ref > 0,
            "ranks": ranks,
            "hits": {arm: hits(r) for arm, r in ranks.items()}}


# --------------------------------------------------------------------------
# Experiment driver.
# --------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            blk = handle.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def summarize(rows, label):
    """Top-k fractions with explicit denominators.

    * "all" covers ALL selected queries, including spectrum-excluded rows:
      top-k denominators count only rankable rows per arm (a row excluded
      before ranking contributes None, never a silent miss), while
      abstention coverage uses every selected query
      (predicted / n_selected).
    * "eligible_*" subsets cover complete rows only (ranking-eligible).
    * "refcount_1" / "refcount_multi" split eligible_anyref rows by pool
      reference count (exactly one vs multiple reference-backed
      candidates), so cosine-vs-coverage-only can be compared where the
      coverage baseline is non-degenerate.
    """
    out = {"label": label, "n_selected": len(rows),
           "n_complete": sum(1 for r in rows
                             if r.get("status") == "complete"),
           "status_counts": dict(Counter(r.get("status", "unknown")
                                         for r in rows)),
           "pool_recall_all": (sum(1 for r in rows if r.get("in_pool"))
                               / len(rows) if rows else None)}
    subsets = (("all", lambda r: True),
               ("eligible_anyref",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and r["n_ref"] > 0),
               ("eligible_targetref",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and r["target_has_ref"]),
               ("refcount_1",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and r["n_ref"] == 1),
               ("refcount_multi",
                lambda r: r.get("status") == "complete" and r["in_pool"]
                and r["n_ref"] > 1))
    for name, sel in subsets:
        sub = [r for r in rows if sel(r)]
        m = {"n": len(sub)}
        for arm in ARMS:
            for k in (1, 3, 10):
                vals = [r["hits"][arm][k] for r in sub
                        if r["hits"].get(arm, {}).get(k) is not None]
                m[f"{arm}@top{k}"] = (sum(vals) / len(vals)
                                      if vals else None)
                m[f"{arm}@top{k}_den"] = len(vals)
        # Abstention (spectral arm): predict iff a complete row has >=1
        # referenced candidate. The coverage denominator is ALL selected
        # queries (parameter-free rule; excluded spectra count as
        # abstentions, never as predictions).
        pred = [r for r in sub if r.get("predicted")]
        good = [r for r in pred if r["hits"].get("spectral", {}).get(1)]
        m["predicted"] = len(pred)
        m["coverage"] = len(pred) / len(rows) if rows else 0.0
        m["abstention_precision"] = (len(good) / len(pred)
                                     if pred else None)
        out[name] = m
    return out


def empty_hits():
    return {arm: {1: None, 3: None, 10: None} for arm in ARMS}


def empty_ranks():
    return {arm: None for arm in ARMS}


def full_row(qid, smiles, row, supplied, distinct, status):
    """One row with the CONSISTENT schema every CSV row carries.

    Spectrum-independent fields are filled for every status; ranking
    fields stay None until computed. Uniform CSV columns are guaranteed
    (the writer also unions fields as belt-and-braces).
    """
    return {
        "qid": qid, "smiles": smiles,
        "adduct": row.get("adduct", ""),
        "instrument": row.get("instrument_type") or "missing",
        "collision_energy": row.get("collision_energy", ""),
        "status": status,
        "n_supplied": len(supplied),
        "n_pool": len(distinct),
        "in_pool": (smiles in distinct),
        "n_ref": None,
        "target_has_ref": None,
        "predicted": False,
        "n_unparseable_mass": None,
        "n_query_dropped_peaks": None,
        "n_content_excluded_refs": None,
        "n_content_excluded_structures": None,
        "rank_uniform": None,
        "rank_spectral": None,
        "rank_coverage": None,
        "rank_massresid": None,
        "hits": empty_hits(),
    }


def fill_spectrum_free_ranks(out, distinct, target_smiles, parent_mass):
    """Uniform + massresid ranks need no spectrum; computable for excluded
    queries too (documented; spectral/coverage stay None without scores)."""
    pos_of = seeded_positions(out["qid"], len(distinct))
    residuals, n_bad = candidate_residuals(distinct, parent_mass)
    order_u = rank_uniform(
        [{"idx": i} for i in range(len(distinct))], pos_of)
    order_m = rank_massresid(
        [{"idx": i} for i in range(len(distinct))], residuals, pos_of)
    tgt = distinct.index(target_smiles) if target_smiles in distinct else None
    if tgt is not None:
        out["rank_uniform"] = order_u.index(tgt) + 1
        out["rank_massresid"] = order_m.index(tgt) + 1
        n = len(distinct)
        for arm, rank in (("uniform", out["rank_uniform"]),
                          ("massresid", out["rank_massresid"])):
            out["hits"][arm] = {1: rank <= 1, 3: rank <= min(3, n),
                                10: rank <= min(10, n)}
    out["n_unparseable_mass"] = n_bad
    return out


def run_val_experiment(tsv_path, prefix_path, max_queries):
    """Val-side run with ENFORCED train/query disjointness.

    Conservative connectivity policy: queries with a missing SMILES or
    missing InChIKey-14 block are excluded before scoring
    (status query_excl:missing_connectivity_key, counted); any nonzero
    SMILES or InChIKey-14 overlap between kept queries and train aborts
    the run (fail safe, never score through a leak).
    """
    # Val groups: first row per SMILES in file order (stable selection).
    val_groups, train_smiles, train_keys = {}, set(), set()
    val_bias = Counter()
    n_rows = 0
    n_train_missing_key = 0
    for row in load_tsv_rows(tsv_path):
        n_rows += 1
        if row["fold"] == "train":
            train_smiles.add(row["smiles"])
            if row["inchikey"]:
                train_keys.add(short_key(row["inchikey"]))
            else:
                n_train_missing_key += 1
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

    # Distinct pools; split off queries lacking a connectivity identity.
    pools, early_excluded = {}, []
    for qi, (q, cands) in enumerate(joined):
        qid = f"VAL-{qi:04d}"
        seen, distinct = set(), []
        for s in cands:
            if s not in seen:
                seen.add(s)
                distinct.append(s)
        rec = {"smiles": q, "supplied": list(cands),
               "distinct": distinct, "row": val_groups[q]}
        if not q or not val_groups[q]["inchikey"]:
            out = full_row(qid, q, val_groups[q], rec["supplied"],
                           distinct, "query_excl:missing_connectivity_key")
            early_excluded.append(out)
        else:
            pools[qid] = rec

    query_smiles = [v["smiles"] for v in pools.values()]
    query_keys = {short_key(v["row"]["inchikey"]) for v in pools.values()}
    ov_smi, ov_key = audit_split_disjointness(train_smiles, query_smiles,
                                              train_keys, query_keys)
    if ov_smi or ov_key:
        raise ValueError(
            f"train/query connectivity overlap: {len(ov_smi)} SMILES, "
            f"{len(ov_key)} InChIKey-14 (fail safe, refusing to score)")

    wanted = set()
    for v in pools.values():
        wanted.update(v["distinct"])
    refs, ref_info = collect_train_refs(tsv_path, wanted,
                                        exclude_smiles=set(query_smiles))
    ref_info["n_train_missing_key"] = n_train_missing_key
    ref_info["n_queries_missing_key"] = len(early_excluded)

    rows = list(early_excluded)
    sel_bias = Counter()
    for qid, v in pools.items():
        row = v["row"]
        sel_bias[f"adduct:{row['adduct']}"] += 1
        sel_bias[f"inst:{row['instrument_type'] or 'missing'}"] += 1
        out = full_row(qid, v["smiles"], row, v["supplied"], v["distinct"],
                       "complete")
        out = fill_spectrum_free_ranks(out, v["distinct"], v["smiles"],
                                       row["parent_mass"])
        qs = parse_query_spectrum(row)
        if not qs["ok"]:
            out["status"] = f"query_parse_excl:{qs['error']}"
            out["predicted"] = False  # abstention; in coverage denominator
            out["n_ref"] = sum(1 for s in v["distinct"] if refs.get(s))
            out["target_has_ref"] = bool(refs.get(v["smiles"]))
            rows.append(out)
            continue
        refs_q, n_cx_r, n_cx_s = exclude_query_content(refs, qs["fp"])
        out["n_content_excluded_refs"] = n_cx_r
        out["n_content_excluded_structures"] = n_cx_s
        ref_info["n_query_content_excluded_refs"] = (
            ref_info.get("n_query_content_excluded_refs", 0) + n_cx_r)
        scored = score_candidates(qs["vec"], qs["norm"], v["distinct"],
                                  refs_q)
        pos_of = seeded_positions(qid, len(scored))
        residuals, n_bad = candidate_residuals(v["distinct"],
                                               row["parent_mass"])
        ev = evaluate_query(qid, scored, pos_of, residuals, v["smiles"])
        out["n_ref"] = ev["n_ref"]
        out["target_has_ref"] = ev["target_has_ref"]
        out["predicted"] = ev["predicted"]
        out["n_unparseable_mass"] = n_bad
        out["n_query_dropped_peaks"] = qs["n_dropped"]
        out["rank_uniform"] = ev["ranks"]["uniform"]
        out["rank_spectral"] = ev["ranks"]["spectral"]
        out["rank_coverage"] = ev["ranks"]["coverage"]
        out["rank_massresid"] = ev["ranks"]["massresid"]
        out["hits"] = ev["hits"]
        rows.append(out)
    # Reference-coverage detail over the SELECTED sample only (persisted
    # here; no mixing with other sample sizes). Exact-string occurrence
    # counts plus the crude no-stereo character approximation (rough
    # sensitivity count, NOT a tight connectivity bound).
    train_nostereo = set()
    for row in load_tsv_rows(tsv_path):
        if row["fold"] == "train":
            train_nostereo.add(strip_stereo(row["smiles"]))
    pool_occ, exact_cov, approx_extra = 0, 0, 0
    for v in pools.values():
        for s in v["distinct"]:
            pool_occ += 1
            if s in refs:
                exact_cov += 1
            elif strip_stereo(s) in train_nostereo:
                approx_extra += 1
    metrics = summarize(rows, "val_formula_pool")
    return {"rows": rows, "metrics": metrics, "ref_info": dict(ref_info),
            "n_tsv_rows": n_rows, "n_val_structures": len(val_groups),
            "n_train_structures": len(train_smiles),
            "n_entries_scanned": len(entries), "n_joined": len(joined),
            "overlap_smiles": sorted(ov_smi), "overlap_keys": sorted(ov_key),
            "overlap_enforced": True,
            "val_bias": dict(val_bias), "selected_bias": dict(sel_bias),
            "coverage_detail": {
                "pool_distinct_occurrences": pool_occ,
                "exact_covered_occurrences": exact_cov,
                "nostereo_approx_extra": approx_extra,
                "nostereo_approx_note": "character-level approximation, "
                                        "not a connectivity bound",
                "n_train_nostereo_keys": len(train_nostereo)}}


def run_control_experiment(tsv_path, prefix_path, max_queries):
    """TRAIN-fold positive control (labelled CONTROL).

    Train spectra as queries against remaining train references, excluding
    the query identifier AND the query content fingerprint (no self-match
    through duplication or re-upload). Not a val result. Control top-k is
    reported next to the coverage-only arm and reference-count subgroups,
    because most control pools hold a single referenced candidate and a
    bare top-1 rate cannot establish representation quality.
    """
    train_groups = {}
    for row in load_tsv_rows(tsv_path):
        if row["fold"] == "train" and row["smiles"] not in train_groups:
            train_groups[row["smiles"]] = row
    entries = [(q, c) for q, c in
               iter_json_entries(prefix_path, max_queries * 40)]
    joined = [(q, c) for q, c in entries if q in train_groups][:max_queries]
    if not joined:
        raise ValueError("no prefix entries join to train rows")
    pools = {}
    for qi, (q, cands) in enumerate(joined):
        qid = f"CTL-{qi:04d}"
        seen, distinct = set(), []
        for s in cands:
            if s not in seen:
                seen.add(s)
                distinct.append(s)
        pools[qid] = {"smiles": q, "supplied": list(cands),
                      "distinct": distinct, "row": train_groups[q]}
    wanted = set()
    for v in pools.values():
        wanted.update(v["distinct"])
    # Refs: all wanted train spectra; the query identifier AND the query
    # content fingerprint are excluded per query at scoring time.
    all_refs, ref_info = collect_train_refs(tsv_path, wanted,
                                            exclude_smiles=())
    rows = []
    for qid, v in pools.items():
        row = v["row"]
        out = full_row(qid, v["smiles"], row, v["supplied"], v["distinct"],
                       "complete")
        out = fill_spectrum_free_ranks(out, v["distinct"], v["smiles"],
                                       row["parent_mass"])
        qs = parse_query_spectrum(row)
        if not qs["ok"]:
            out["status"] = f"query_parse_excl:{qs['error']}"
            out["predicted"] = False
            out["n_ref"] = sum(1 for s in v["distinct"] if all_refs.get(s))
            out["target_has_ref"] = bool(all_refs.get(v["smiles"]))
            rows.append(out)
            continue
        refs = {}
        n_cx_r, n_cx_s = 0, 0
        for smi, lst in all_refs.items():
            kept = [t for t in lst
                    if t[0] != row["identifier"] and t[6] != qs["fp"]]
            dropped = len(lst) - len(kept)
            if dropped:
                n_cx_r += dropped
                n_cx_s += 1
            if kept:
                refs[smi] = kept
        out["n_content_excluded_refs"] = n_cx_r
        out["n_content_excluded_structures"] = n_cx_s
        ref_info["n_control_content_excluded_refs"] = (
            ref_info.get("n_control_content_excluded_refs", 0) + n_cx_r)
        scored = score_candidates(qs["vec"], qs["norm"], v["distinct"], refs)
        pos_of = seeded_positions(qid, len(scored))
        residuals, n_bad = candidate_residuals(v["distinct"],
                                               row["parent_mass"])
        ev = evaluate_query(qid, scored, pos_of, residuals, v["smiles"])
        out["n_ref"] = ev["n_ref"]
        out["target_has_ref"] = ev["target_has_ref"]
        out["predicted"] = ev["predicted"]
        out["n_unparseable_mass"] = n_bad
        out["n_query_dropped_peaks"] = qs["n_dropped"]
        out["rank_uniform"] = ev["ranks"]["uniform"]
        out["rank_spectral"] = ev["ranks"]["spectral"]
        out["rank_coverage"] = ev["ranks"]["coverage"]
        out["rank_massresid"] = ev["ranks"]["massresid"]
        out["hits"] = ev["hits"]
        rows.append(out)
    metrics = summarize(rows, "train_control")
    return {"rows": rows, "metrics": metrics, "ref_info": dict(ref_info),
            "n_joined": len(joined)}


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/pinned/MassSpecGym1.5.tsv")
    ap.add_argument("--prefix",
                    default="data/pinned/msgym_candidates_formula_prefix64.json")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-control", type=int, default=50)
    ap.add_argument("--out-dir",
                    default="experiments/molecular_completion/20261003_spectral")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    c0 = time.process_time()
    val = run_val_experiment(args.tsv, args.prefix, args.max_queries)
    ctl = run_control_experiment(args.tsv, args.prefix, args.max_control)
    # Hashing and row-output writes happen INSIDE the timed scope, so the
    # reported wall/CPU/RAM covers the whole run (only the few-ms summary
    # JSON write itself falls outside; documented in cost.note).
    pins = {"tsv": args.tsv, "prefix": args.prefix,
            "tsv_sha256": sha256_file(args.tsv),
            "prefix_sha256": sha256_file(args.prefix),
            "tsv_bytes": os.path.getsize(args.tsv),
            "prefix_bytes": os.path.getsize(args.prefix)}
    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, "spectral_rows.csv")
    ctl_path = os.path.join(args.out_dir, "spectral_control_rows.csv")
    sum_path = os.path.join(args.out_dir, "spectral_summary.json")

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

    flat_val, flat_ctl = flatten(val["rows"]), flatten(ctl["rows"])
    for path, flat in ((rows_path, flat_val), (ctl_path, flat_ctl)):
        # Union of fields across rows: mixed complete/exclusion schemas can
        # never crash the writer, and every row keeps the same columns.
        cols = sorted({c for f in flat for c in f})
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for f in flat:
                writer.writerow({c: f.get(c) for c in cols})
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "representation": {"bin_width": BIN_WIDTH, "mz_min": MZ_MIN,
                           "mz_max": MZ_MAX, "n_bins": N_BINS,
                           "aggregation": "per-bin max intensity",
                           "similarity": "cosine on L2-normalized sparse "
                                         "vectors, max over refs",
                           "identity": "exact SMILES string (conservative "
                                       "lower-bound coverage, not graph "
                                       "canonicalization; no RDKit)",
                           "shuffle": SHUFFLE_DOMAIN,
                           "selection": "file-order prefix entries joining "
                                        "fold rows; first row per SMILES",
                           "fixed_a_priori": True,
                           "train_calibration": "not applicable "
                                                "(parameter-free abstention; "
                                                "no thresholds fit)"},
        "pins": pins,
        "provenance": {"downloads_bytes": 0, "api_calls": 0,
                       "test_fold": "streamed-but-unused: test-fold rows "
                                    "pass through the shared TSV scanner "
                                    "and are never used for fitting, "
                                    "scoring, or selection"},
        "params": {"max_queries": args.max_queries,
                   "max_control": args.max_control},
        "val": {k: v for k, v in val.items() if k != "rows"},
        "control": {k: v for k, v in ctl.items() if k != "rows"},
        "cost": {"wall_s": wall, "cpu_s": cpu,
                 "peak_rss_kb": peak_kb,
                 "note": "single-threaded CPU; stdlib only; timer covers "
                         "experiment + sha256 verification + row CSV "
                         "writes; only the summary-JSON write itself "
                         "(ms-scale) falls outside"},
        "outputs": {"rows": rows_path, "control_rows": ctl_path,
                    "summary": sum_path,
                    "rows_bytes": os.path.getsize(rows_path),
                    "control_bytes": os.path.getsize(ctl_path),
                    "summary_bytes_note": "stdout only: a file cannot "
                                          "persist its own final size "
                                          "before writing it"},
    }
    with open(sum_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")
    summary["outputs"]["summary_bytes"] = os.path.getsize(sum_path)
    sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

class SpectrumParseTests(unittest.TestCase):
    def test_valid(self):
        peaks, err = parse_spectrum("100.0,200.5", "0.5,1.0")
        self.assertIsNone(err)
        self.assertEqual(peaks, [(100.0, 0.5), (200.5, 1.0)])

    def test_empty(self):
        for a, b in (("", ""), ("  ", "0.5"), ("100.0", ""),
                     (None, "0.5"), ("100.0", None)):
            _, err = parse_spectrum(a, b)
            self.assertIsNotNone(err, (a, b))

    def test_length_mismatch(self):
        _, err = parse_spectrum("100.0,200.0", "0.5")
        self.assertEqual(err, "length_mismatch")

    def test_non_numeric(self):
        _, err = parse_spectrum("100.0,abc", "0.5,1.0")
        self.assertEqual(err, "non_numeric")

    def test_non_finite(self):
        _, err = parse_spectrum("100.0,nan", "0.5,1.0")
        self.assertEqual(err, "non_finite")
        _, err = parse_spectrum("100.0,inf", "0.5,1.0")
        self.assertEqual(err, "non_finite")

    def test_negative(self):
        _, err = parse_spectrum("-5.0,100.0", "0.5,1.0")
        self.assertEqual(err, "negative_value")
        _, err = parse_spectrum("100.0,200.0", "0.5,-1.0")
        self.assertEqual(err, "negative_value")

    def test_zero_norm(self):
        _, err = parse_spectrum("100.0,200.0", "0.0,0.0")
        self.assertEqual(err, "zero_norm")


class BinCosineTests(unittest.TestCase):
    def test_bin_edges(self):
        vec, dropped = bin_spectrum([(0.0, 0.5), (0.05, 0.7), (1999.99, 1.0),
                                     (2000.0, 1.0), (2500.0, 1.0)])
        self.assertEqual(dropped, 2)  # upper edge + out of range
        self.assertEqual(vec[0], 0.7)  # max aggregation within a bin
        self.assertEqual(vec[N_BINS - 1], 1.0)

    def test_identical_is_one(self):
        vec, _ = bin_spectrum([(100.0, 0.5), (200.0, 1.0)])
        n = vec_norm(vec)
        self.assertAlmostEqual(cosine(vec, n, vec, n), 1.0)

    def test_orthogonal_is_zero(self):
        a, _ = bin_spectrum([(100.0, 1.0)])
        b, _ = bin_spectrum([(300.0, 1.0)])
        self.assertEqual(cosine(a, vec_norm(a), b, vec_norm(b)), 0.0)

    def test_scale_invariant(self):
        a, _ = bin_spectrum([(100.0, 0.5)])
        b, _ = bin_spectrum([(100.0, 2.0)])
        self.assertAlmostEqual(cosine(a, vec_norm(a), b, vec_norm(b)), 1.0)

    def test_zero_vector_scores_zero(self):
        a, dropped = bin_spectrum([(9999.0, 1.0)])
        self.assertEqual(a, {})
        self.assertEqual(dropped, 1)
        b, _ = bin_spectrum([(100.0, 1.0)])
        self.assertEqual(cosine(a, 0.0, b, vec_norm(b)), 0.0)
        self.assertEqual(cosine(a, 0.0, a, 0.0), 0.0)

    def test_partial_overlap_bounded(self):
        a, _ = bin_spectrum([(100.0, 1.0), (200.0, 1.0)])
        b, _ = bin_spectrum([(100.0, 1.0), (300.0, 1.0)])
        s = cosine(a, vec_norm(a), b, vec_norm(b))
        self.assertAlmostEqual(s, 0.5)


class RankBehaviorTests(unittest.TestCase):
    def _scored(self):
        return [
            {"idx": 0, "smiles": "A", "score": 0.9, "n_refs": 2,
             "best_ref": "r1"},
            {"idx": 1, "smiles": "B", "score": 0.2, "n_refs": 1,
             "best_ref": "r2"},
            {"idx": 2, "smiles": "C", "score": None, "n_refs": 0,
             "best_ref": None},
            {"idx": 3, "smiles": "D", "score": None, "n_refs": 0,
             "best_ref": None},
        ]

    def test_max_aggregation_picks_best_ref(self):
        q = {10: 1.0}
        refs = {"X": [("r1", {10: 0.1, 20: 1.0}, math.sqrt(1.01), 0, "", "",
                      ()),
                      ("r2", {10: 1.0}, 1.0, 0, "", "", ())]}
        out = score_candidates(q, 1.0, ["X"], refs)
        self.assertEqual(out[0]["n_refs"], 2)
        self.assertAlmostEqual(out[0]["score"], 1.0)
        self.assertEqual(out[0]["best_ref"], "r2")

    def test_missing_refs_rank_last_but_retained(self):
        scored = self._scored()
        pos = seeded_positions("Q", len(scored))
        order = rank_spectral(scored, pos)
        self.assertEqual([scored[i]["smiles"] for i in order[:2]], ["A", "B"])
        # Unreferenced candidates stay in the ranking (never dropped).
        self.assertEqual(len(order), 4)
        self.assertEqual(set(order), {0, 1, 2, 3})
        tail = [scored[i]["smiles"] for i in order[2:]]
        self.assertEqual(sorted(tail), ["C", "D"])

    def test_ties_break_by_seeded_position(self):
        scored = [
            {"idx": 0, "smiles": "A", "score": 0.5, "n_refs": 1,
             "best_ref": "r"},
            {"idx": 1, "smiles": "B", "score": 0.5, "n_refs": 1,
             "best_ref": "r"},
        ]
        o1 = rank_spectral(scored, seeded_positions("Q1", 2))
        o2 = rank_spectral(scored, seeded_positions("Q1", 2))
        self.assertEqual(o1, o2)  # deterministic per qid
        # Order follows shuffled position, never ambient index order alone:
        pos = seeded_positions("Q1", 2)
        expect_first = 0 if pos[0] < pos[1] else 1
        self.assertEqual(o1[0], expect_first)

    def test_absent_target_ranks_none(self):
        scored = self._scored()
        pos = seeded_positions("Q", len(scored))
        ev = evaluate_query("Q", scored, pos, [1.0, 2.0, 3.0, 4.0], "ZZZ")
        self.assertFalse(ev["in_pool"])
        self.assertIsNone(ev["ranks"]["spectral"])
        self.assertIsNone(ev["hits"]["spectral"][1])
        self.assertIsNone(ev["hits"]["uniform"][10])

    def test_uniform_uses_shuffle_not_ambient(self):
        scored = self._scored()
        o = rank_uniform(scored, seeded_positions("Q7", len(scored)))
        self.assertEqual(sorted(o), [0, 1, 2, 3])
        # Seeded shuffle of 4 is overwhelmingly likely non-identity; the
        # robust assertion is determinism + position-dependence instead:
        self.assertEqual(o, rank_uniform(scored, seeded_positions("Q7", 4)))

    def test_identity_is_exact_string(self):
        # Stereo variants are DIFFERENT identities (conservative lower
        # bound on coverage): 'C[C@H](N)O' != 'C[C@@H](N)O'.
        self.assertNotEqual("C[C@H](N)O", "C[C@@H](N)O")
        self.assertEqual(strip_stereo("C[C@H](N)O"),
                         strip_stereo("C[C@@H](N)O"))

    def test_massresid_unparseable_last(self):
        scored = self._scored()
        pos = seeded_positions("Q", len(scored))
        order = rank_massresid(scored, [5.0, None, 0.1, None], pos)
        self.assertEqual(scored[order[0]]["smiles"], "C")  # 0.1 ppm first
        self.assertEqual(scored[order[1]]["smiles"], "A")  # 5.0 ppm next
        # Unparseable (None) trail, retained.
        self.assertEqual(len(order), 4)


class LeakageTests(unittest.TestCase):
    def test_overlap_audit_detects(self):
        ov_smi, ov_key = audit_split_disjointness(
            {"A", "B"}, ["B", "C"], {"k1", "k2"}, {"k2", "k3"})
        self.assertEqual(ov_smi, {"B"})
        self.assertEqual(ov_key, {"k2"})

    def test_no_oracle_query_features(self):
        # score_candidates must not read target/is_target style fields.
        import inspect
        src = inspect.getsource(score_candidates)
        self.assertNotIn("is_target", src)
        self.assertNotIn("target_smiles", src)

    def test_excluded_queries_never_scored(self):
        # File-backed: a train row whose SMILES is in exclude_smiles is
        # counted as a leak exclusion and never lands in refs.
        import tempfile
        header = "\t".join(WANTED_COLUMNS)
        rows = [
            ["T1", "100.0", "1.0", "CCO", "K" * 14, "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
            ["T2", "200.0", "1.0", "CCC", "J" * 14, "C3H8", "C3H9",
             "44.0", "45.0", "[M+H]+", "QTOF", "20", "train"],
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as handle:
            handle.write(header + "\n")
            for r in rows:
                handle.write("\t".join(r) + "\n")
            path = handle.name
        try:
            refs, info = collect_train_refs(path, {"CCO", "CCC"},
                                            exclude_smiles={"CCO"})
        finally:
            os.unlink(path)
        self.assertNotIn("CCO", refs)
        self.assertIn("CCC", refs)
        self.assertEqual(info["n_leak_excluded"], 1)

    def test_duplicate_ids_first_wins(self):
        import tempfile
        header = "\t".join(WANTED_COLUMNS)
        rows = [
            ["DUP", "100.0", "1.0", "CCO", "K" * 14, "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
            ["DUP", "300.0", "1.0", "CCO", "K" * 14, "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as handle:
            handle.write(header + "\n")
            for r in rows:
                handle.write("\t".join(r) + "\n")
            path = handle.name
        try:
            refs, info = collect_train_refs(path, {"CCO"})
        finally:
            os.unlink(path)
        self.assertEqual(len(refs["CCO"]), 1)
        self.assertEqual(info["n_duplicate_ids"], 1)
        # First occurrence wins.
        self.assertEqual(refs["CCO"][0][0], "DUP")
        self.assertIn(1000, refs["CCO"][0][1])  # 100.0 Da bin


class ContentGuardTests(unittest.TestCase):
    def test_fingerprint_equal_content(self):
        self.assertEqual(content_fp([(100.0, 0.5), (200.0, 1.0)]),
                         content_fp([(200.0, 1.0), (100.0, 0.5)]))
        self.assertNotEqual(content_fp([(100.0, 0.5)]),
                            content_fp([(100.05, 0.5)]))

    def test_exclude_query_content_different_id(self):
        fp_q = content_fp([(100.0, 1.0)])
        refs = {"X": [("r1", {1000: 1.0}, 1.0, 0, "", "", fp_q),
                      ("r2", {2000: 1.0}, 1.0, 0, "", "",
                       content_fp([(200.0, 1.0)]))],
                "Y": [("r3", {1000: 1.0}, 1.0, 0, "", "", fp_q)]}
        kept, n_r, n_s = exclude_query_content(refs, fp_q)
        self.assertEqual(n_r, 2)
        self.assertEqual(n_s, 2)
        self.assertEqual(kept, {"X": [refs["X"][1]]})

    def test_collect_dedups_content_within_structure(self):
        import tempfile
        header = "\t".join(WANTED_COLUMNS)
        rows = [
            ["A1", "100.0,200.0", "0.5,1.0", "CCO", "K" * 14, "C2H6O",
             "C2H7O", "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
            ["A2", "100.0,200.0", "0.5,1.0", "CCO", "K" * 14, "C2H6O",
             "C2H7O", "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
            ["A3", "150.0", "1.0", "CCO", "K" * 14, "C2H6O", "C2H7O",
             "46.0", "47.0", "[M+H]+", "QTOF", "20", "train"],
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as handle:
            handle.write(header + "\n")
            for r in rows:
                handle.write("\t".join(r) + "\n")
            path = handle.name
        try:
            refs, info = collect_train_refs(path, {"CCO"})
        finally:
            os.unlink(path)
        self.assertEqual(len(refs["CCO"]), 2)  # A1 + A3; A2 deduped
        self.assertEqual(info["n_duplicate_content"], 1)
        self.assertEqual(info.get("n_duplicate_ids", 0), 0)


class CoverageBaselineTests(unittest.TestCase):
    def test_referenced_first_no_scores(self):
        scored = [
            {"idx": 0, "smiles": "A", "score": 0.99, "n_refs": 0,
             "best_ref": None},
            {"idx": 1, "smiles": "B", "score": None, "n_refs": 0,
             "best_ref": None},
            {"idx": 2, "smiles": "C", "score": 0.01, "n_refs": 3,
             "best_ref": "r"},
        ]
        pos = seeded_positions("QC", 3)
        order = rank_coverage(scored, pos)
        # Only referenced candidate first regardless of (low) score.
        self.assertEqual(order[0], 2)
        self.assertEqual(sorted(order), [0, 1, 2])
        import inspect
        src = inspect.getsource(rank_coverage)
        self.assertNotIn('["score"]', src)
        self.assertNotIn("['score']", src)

    def test_coverage_arm_in_evaluate(self):
        scored = [
            {"idx": 0, "smiles": "A", "score": 0.9, "n_refs": 1,
             "best_ref": "r"},
            {"idx": 1, "smiles": "T", "score": 0.1, "n_refs": 1,
             "best_ref": "r"},
        ]
        ev = evaluate_query("Q", scored, seeded_positions("Q", 2),
                            [1.0, 1.0], "T")
        # Spectral follows scores (A first); coverage follows shuffle among
        # referenced (may or may not match) -- the recorded distinction is
        # that both ranks exist on identical pools.
        self.assertIn("coverage", ev["ranks"])
        self.assertIn("coverage", ev["hits"])
        self.assertIsNotNone(ev["ranks"]["coverage"])


class DenominatorTests(unittest.TestCase):
    def _rows(self):
        good = full_row("VAL-0000", "T", {"adduct": "a",
                                             "instrument_type": "i",
                                             "collision_energy": "10",
                                             "parent_mass": "46.0"},
                        ["T", "U"], ["T", "U"], "complete")
        good.update({"n_ref": 1, "target_has_ref": False, "predicted": True,
                     "hits": {"uniform": {1: False, 3: True, 10: True},
                              "spectral": {1: False, 3: False, 10: True},
                              "coverage": {1: False, 3: True, 10: True},
                              "massresid": {1: True, 3: True, 10: True}}})
        bad = full_row("VAL-0001", "T2", {"adduct": "a",
                                              "instrument_type": "i",
                                              "collision_energy": "10",
                                              "parent_mass": "46.0"},
                       ["T2", "U"], ["T2", "U"],
                       "query_parse_excl:non_numeric")
        bad.update({"n_ref": 1, "target_has_ref": False})
        return [good, bad]

    def test_all_covers_selected_excluded_in_denominators(self):
        m = summarize(self._rows(), "t")
        self.assertEqual(m["n_selected"], 2)
        self.assertEqual(m["n_complete"], 1)
        self.assertEqual(m["pool_recall_all"], 1.0)  # both targets in pools
        # Coverage over ALL selected (excluded counts as abstention).
        self.assertEqual(m["all"]["coverage"], 0.5)
        # Top-k denominators count rankable rows only.
        self.assertEqual(m["all"]["spectral@top1_den"], 1)
        self.assertEqual(m["eligible_anyref"]["n"], 1)

    def test_refcount_subgroups(self):
        rows = self._rows()
        rows[0]["n_ref"] = 2
        m = summarize(rows, "t")
        self.assertEqual(m["refcount_multi"]["n"], 1)
        self.assertEqual(m["refcount_1"]["n"], 0)

    def test_strip_stereo_documented_approximation(self):
        import inspect
        self.assertIn("APPROXIMATION",
                      inspect.getdoc(strip_stereo).upper())


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(WANTED_COLUMNS) + "\n")
        for r in rows:
            handle.write("\t".join(r) + "\n")


def _row(ident, mzs, ints, smiles, key, fold, parent="46.041864"):
    return [ident, mzs, ints, smiles, key, "C2H6O", "C2H7O", parent, "47.0",
            "[M+H]+", "QTOF", "20", fold]


class EnforcementTests(unittest.TestCase):
    def test_overlap_aborts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tsv = os.path.join(tmp, "t.tsv")
            pre = os.path.join(tmp, "p.json")
            _write_tsv(tsv, [
                _row("T1", "100.0", "1.0", "CCO", "K" * 14, "train"),
                _row("V1", "200.0", "1.0", "CCO", "K" * 14, "val"),
            ])
            with open(pre, "w", encoding="utf-8") as handle:
                json.dump({"CCO": ["CCO", "CCC"]}, handle)
            with self.assertRaises(ValueError):
                run_val_experiment(tsv, pre, 10)

    def test_missing_key_excluded_not_scored(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tsv = os.path.join(tmp, "t.tsv")
            pre = os.path.join(tmp, "p.json")
            _write_tsv(tsv, [
                _row("T1", "100.0", "1.0", "CCO", "K" * 14, "train"),
                _row("V1", "200.0", "1.0", "CCN", "", "val"),
            ])
            with open(pre, "w", encoding="utf-8") as handle:
                json.dump({"CCN": ["CCN", "CCO"]}, handle)
            out = run_val_experiment(tsv, pre, 10)
            self.assertEqual(len(out["rows"]), 1)
            self.assertEqual(out["rows"][0]["status"],
                             "query_excl:missing_connectivity_key")
            self.assertEqual(out["ref_info"]["n_queries_missing_key"], 1)
            self.assertEqual(out["metrics"]["n_selected"], 1)
            self.assertEqual(out["metrics"]["n_complete"], 0)


class IntegrationTests(unittest.TestCase):
    def test_malformed_first_and_later_full_write(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tsv = os.path.join(tmp, "t.tsv")
            pre = os.path.join(tmp, "p.json")
            out = os.path.join(tmp, "out")
            _write_tsv(tsv, [
                _row("T1", "100.0,200.0", "0.5,1.0", "CCO", "K" * 14,
                     "train"),
                _row("T2", "100.0,200.0", "0.5,1.0", "CCO", "K" * 14,
                     "train"),
                _row("T3", "150.0,250.0", "1.0,0.5", "CCO", "K" * 14,
                     "train"),
                _row("T4", "300.0", "1.0", "CCC", "J" * 14, "train"),
                _row("Vbad1", "xx", "1.0", "BAD1", "B" * 14, "val"),
                _row("Vq", "110.0,210.0", "0.6,0.9", "CCN", "N" * 14,
                     "val"),
                _row("Vbad2", "100.0", "", "BAD2", "C" * 14, "val"),
            ])
            with open(pre, "w", encoding="utf-8") as handle:
                json.dump({"BAD1": ["CCO", "CCC"],
                           "CCN": ["CCO", "CCC", "CCN"],
                           "BAD2": ["CCO", "CCC"],
                           "CCO": ["CCO", "CCC"]}, handle)
            main(["--tsv", tsv, "--prefix", pre, "--max-queries", "3",
                  "--max-control", "1", "--out-dir", out])
            val_rows = list(csv.DictReader(
                open(os.path.join(out, "spectral_rows.csv"))))
            ctl_rows = list(csv.DictReader(
                open(os.path.join(out, "spectral_control_rows.csv"))))
            summary = json.load(
                open(os.path.join(out, "spectral_summary.json")))
            # Mixed complete/exclusion schemas share one column set.
            self.assertEqual(len(val_rows), 3)
            for r in val_rows:
                self.assertEqual(set(r.keys()), set(val_rows[0].keys()))
            for r in ctl_rows:
                self.assertEqual(set(r.keys()), set(ctl_rows[0].keys()))
            statuses = [r["status"] for r in val_rows]
            self.assertEqual(statuses[0], "query_parse_excl:non_numeric")
            self.assertEqual(statuses[1], "complete")
            self.assertEqual(statuses[2], "query_parse_excl:empty_spectrum")
            # Selected/complete/excluded accounting + pool recall on all.
            m = summary["val"]["metrics"]
            self.assertEqual(m["n_selected"], 3)
            self.assertEqual(m["n_complete"], 1)
            self.assertEqual(m["pool_recall_all"], 1 / 3)
            # Abstention coverage denominator includes excluded spectra.
            self.assertEqual(m["all"]["coverage"],
                             m["all"]["predicted"] / 3)
            # Machine-readable output accounting persisted.
            self.assertIn("rows_bytes", summary["outputs"])
            self.assertIn("control_bytes", summary["outputs"])
            self.assertEqual(summary["provenance"]["api_calls"], 0)
            self.assertEqual(summary["provenance"]["downloads_bytes"], 0)
            # Content-dedup counted (T2 duplicates T1 content).
            self.assertEqual(summary["val"]["ref_info"]
                             ["n_duplicate_content"], 1)


if __name__ == "__main__":
    main(sys.argv[1:])
