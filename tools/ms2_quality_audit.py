"""Quality gates, confidence analysis and protocol-status inputs (no downloads).

Reads ONLY pinned local data and frozen prior outputs:
  data/pinned/MassSpecGym1.5.tsv (read-only scan; test fold scanned for
    identity/scaffold overlap audit only, never scored or trained on),
  data/pinned/PIN.json,
  experiments/molecular_completion/20261003_predictor/predictor_rows.csv
    (frozen 200 val rows with fixed ranks from the reviewed run),
  experiments/molecular_completion/20261003_predictor/predictor_summary.json,
  experiments/molecular_completion/20261003_predictor/train_fit_rows.csv
    (fit/calibration identities for the leakage-exclusion list).

Produces (into experiments/molecular_completion/20261003_remaining/):
  quality_rows.csv      per-query top-1/3/10/25 hits per arm + denominators
  quality_summary.json  strict JSON (allow_nan=False): rates, 95% Wilson
                        intervals, paired predictor-vs-baseline differences,
                        denominators, overlap counts, budgets, per-1000-query
                        costs, PubChem/official-baseline/scaffold statuses.

No network, no API calls, no refit, no val tuning. The earlier 200 stay
frozen; ranks are reused verbatim from the reviewed predictor run.

CPU stdlib + scientific baseline only: GPU execution and Rust/Python parity
are NOT APPLICABLE (recorded as limitations, not passing checks).
"""

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_spectral_rank import (  # noqa: E402
    parse_spectrum,
    sha256_file,
)

ARMS = ("uniform", "massresid", "predictor", "prior")
KS = (1, 3, 10, 25)
Z95 = 1.959963984540054

# Adducts with validated neutral-mass conversion rules. Anything else is
# rejected as unsupported (never silently converted).
SUPPORTED_ADDUCTS = ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+")

# Ion-mass constants (Da), CODATA 2018 rounded as used for conversion rules.
PROTON_MASS = 1.007276466879
ELECTRON_MASS = 0.000548579909
NA_MASS = 22.9897692809
K_MASS = 38.9637074864


def progress(msg):
    sys.stderr.write(f"# progress {msg}\n")
    sys.stderr.flush()


def wilson_interval(hits, n, z=Z95):
    """95% Wilson score interval for a binomial rate. Returns (lo, hi) or (None, None) if n == 0."""
    if n <= 0:
        return None, None
    if hits < 0 or hits > n:
        raise ValueError(f"hits out of range: {hits}/{n}")
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    if lo < 1e-12:
        lo = 0.0
    if hi > 1.0 - 1e-12:
        hi = 1.0
    return lo, hi


def paired_diff_ci(a_hits, b_hits, n, z=Z95):
    """Paired difference (a-b) rate CI via Wald SE on paired differences.

    a_hits/b_hits are per-query 0/1 lists of equal length n. Returns
    (diff, lo, hi, n_a_better, n_b_better, n_tie). Descriptive only; no
    tuning, no selection on val outcomes.
    """
    if len(a_hits) != n or len(b_hits) != n:
        raise ValueError("length mismatch")
    if n <= 0:
        return 0.0, None, None, 0, 0, 0
    diffs = [1 if (a and not b) else (-1 if (b and not a) else 0)
             for a, b in zip(a_hits, b_hits)]
    n_ab = sum(1 for d in diffs if d == 1)
    n_ba = sum(1 for d in diffs if d == -1)
    mean = sum(diffs) / n
    if n == 1:
        return mean, None, None, n_ab, n_ba, n - n_ab - n_ba
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se, n_ab, n_ba, n - n_ab - n_ba


def hit_at_k(rank_text, k):
    """Hit predicate from a persisted rank cell. None/empty -> False (absent target)."""
    if rank_text is None or rank_text == "":
        return False
    try:
        r = int(rank_text)
    except (ValueError, TypeError):
        return False
    return 1 <= r <= k


def short_key(text, n=16):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def inchikey14(key):
    if key is None:
        return None
    key = key.strip()
    if not key:
        return None
    return key.split("-")[0] if "-" in key else key[:14]


# --------------------------------------------------------------------------
# TSV read-only scan (test fold: identity/overlap audit only).
# --------------------------------------------------------------------------

WANTED = ("identifier", "smiles", "inchikey", "formula", "parent_mass",
          "adduct", "instrument_type", "collision_energy", "fold",
          "mzs", "intensities")


def scan_tsv_identities(tsv_path, max_rows=0):
    """Read-only identity scan. Returns per-fold dicts of SMILES/InChIKey sets + adduct counts.

    Spectra are parsed for presence bookkeeping only (counts), never used
    for scoring/training here. max_rows=0 means no cap.
    """
    folds = {}
    n_rows = 0
    with open(tsv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            n_rows += 1
            if max_rows and n_rows > max_rows:
                break
            fold = (row.get("fold") or "unknown").strip()
            d = folds.setdefault(fold, {
                "n_rows": 0, "smiles": set(), "keys": set(),
                "adduct_counts": {}, "inst_counts": {},
                "n_spectrum_ok": 0, "n_spectrum_bad": 0,
            })
            d["n_rows"] += 1
            smi = (row.get("smiles") or "").strip()
            if smi:
                d["smiles"].add(smi)
            key = inchikey14(row.get("inchikey"))
            if key:
                d["keys"].add(key)
            ad = (row.get("adduct") or "").strip() or "missing"
            d["adduct_counts"][ad] = d["adduct_counts"].get(ad, 0) + 1
            inst = (row.get("instrument_type") or "").strip() or "missing"
            d["inst_counts"][inst] = d["inst_counts"].get(inst, 0) + 1
            peaks, err = parse_spectrum(row.get("mzs"), row.get("intensities"))
            if peaks is None:
                d["n_spectrum_bad"] += 1
            else:
                d["n_spectrum_ok"] += 1
    return folds, n_rows


def structural_family_proxy(smiles_list, standardize_fn, cap=5000):
    """Crude structural-family proxy groups (NOT exact Murcko scaffolds).

    Group key: (formula, sorted atom-type multiset, n_edges). Requires the
    existing typed-graph standardizer; unparseable SMILES are counted under
    their exclusion reason. Capped at `cap` distinct SMILES with explicit
    truncation status. The exact-scaffold (Murcko/RDKit) gate is INCOMPLETE:
    RDKit is unavailable and no proxy is substituted for it.
    """
    from collections import Counter
    groups = {}
    n_done = 0
    n_truncated = 0
    excl = Counter()
    for smi in smiles_list:
        if n_done >= cap:
            n_truncated += 1
            continue
        rec, reason = standardize_fn(smi)
        n_done += 1
        if rec is None:
            excl[reason] += 1
            continue
        try:
            from tools.ms2_chebi_corpus import formula_key as _fk
            form_sig = _fk(rec["formula"])
            atom_sig = tuple(sorted((a[0], a[1]) for a in rec["atom_types"]))
            key = (form_sig, atom_sig, len(rec["edges"]))
        except (TypeError, IndexError, KeyError):
            key = ("repr", short_key(repr((rec.get("formula"), rec.get("atom_types"),
                                           len(rec.get("edges", ()))))))
        groups.setdefault(key, []).append(smi)
    return groups, {"n_done": n_done, "n_truncated": n_truncated,
                    "exclusions": dict(excl),
                    "method": ("formula + sorted atom-type multiset + edge count via "
                               "existing typed-graph standardizer; NOT Murcko"),
                    "exact_scaffold_gate": ("INCOMPLETE: exact Murcko scaffold requires "
                                            "RDKit (unavailable); no proxy substituted")}


# --------------------------------------------------------------------------
# Main analysis.
# --------------------------------------------------------------------------

def load_predictor_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def has_valid_ranks(r):
    """All four arm rank cells present (non-None, non-empty).

    Arm-specific None/empty ranks (absent target for that arm, unscored
    pool) are NOT valid applicable ranks; queries lacking them are kept in
    all-selected denominators as misses but excluded from eligible
    accuracy denominators.
    """
    for arm in ARMS:
        v = r.get(f"rank_{arm}")
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True


def is_eligible_parseable(r):
    """Complete + present + parseable + valid applicable ranks."""
    return (r.get("status") == "complete"
            and r.get("in_pool") == "True"
            and r.get("target_parseable") == "True"
            and has_valid_ranks(r))


def analyze_topk(rows):
    """Top-k hits per arm on all / present / parseable / capability subgroups.

    Denominators (explicit, never conflated):
      all_selected:             every selected query (absent rank -> miss)
      target_present:           target present in its supplied pool
      eligible_parseable:       complete AND present AND target graph
                                parseable AND valid applicable ranks
                                (all four arm rank cells present)
      capability_any_candidate: rankable (any finite-score candidate)
    Pool recall over all selected is reported separately from ranking
    rates; ranking denominators never silently drop absent targets.
    Eligible accuracy uses ranks-only-applicable denominators.
    """
    out = {}
    for label, sel in (("all_selected", lambda r: True),
                       ("target_present", lambda r: r.get("in_pool") == "True"),
                       ("eligible_parseable", is_eligible_parseable),
                       ("capability_any_candidate", lambda r: r.get("rankable") == "True")):
        sub = [r for r in rows if sel(r)]
        n = len(sub)
        entry = {"n": n, "qids": [r.get("qid") for r in sub]}
        for arm in ARMS:
            col = f"rank_{arm}"
            hits = {}
            for k in KS:
                h = [hit_at_k(r.get(col), k) for r in sub]
                c = sum(h)
                lo, hi = wilson_interval(c, n)
                hits[f"top{k}"] = {"hits": c, "n": n,
                                   "rate": (c / n) if n else None,
                                   "ci95": [lo, hi]}
            entry[arm] = hits
            entry[f"{arm}_hitvec_top1"] = {
                r.get("qid"): hit_at_k(r.get(col), 1) for r in sub}
        out[label] = entry
    # legacy alias: "all" == "all_selected" (kept for downstream readers)
    out["all"] = out["all_selected"]
    return out


def compute_pubchem_status(internal_misses, external_misses, api_calls=0):
    """Pure PubChem-conditional gate from MEASURED misses (internal+external).

    internal_misses: list of miss keys (formula/adduct) from the frozen run.
    external_misses: None (not yet measured) or list of miss keys from the
    external validation. Returns a status dict; trigger fires only on
    measured misses. Never invents candidates or issues calls here.
    """
    if external_misses is None:
        ext_note = "external misses not yet measured"
        total = list(internal_misses)
    else:
        ext_note = f"{len(external_misses)} external pool-miss queries measured"
        total = list(internal_misses) + list(external_misses)
    if total:
        return {"trigger_rule": "expand only formulas/masses with measured pool misses",
                "measured_misses": len(total),
                "internal_misses": len(internal_misses),
                "external_note": ext_note,
                "status": (f"triggered: {len(total)} measured pool misses "
                           f"({len(internal_misses)} internal + "
                           f"{len(total) - len(internal_misses)} external); tiny pinned "
                           f"expansion (<=5 formulas x <=100 candidates) required"),
                "api_calls": api_calls,
                "policy": "cache by formula+DB date, <=5 req/s + backoff, <=5 formulas x <=100 candidates"}
    return {"trigger_rule": "expand only formulas/masses with measured pool misses",
            "measured_misses": 0,
            "internal_misses": len(internal_misses),
            "external_note": ext_note,
            "status": ("not-triggered: no measured pool misses in the frozen "
                       "slice; no PubChem query was issued"),
            "api_calls": api_calls,
            "policy": "would cache by formula+DB date, <=5 req/s + backoff, <=5 formulas x <=100 candidates"}


def analyze_paired(topk):
    """Paired predictor-vs-baseline differences on fixed existing ranks (no tuning).

    Pairs are matched on qid explicitly: only qids present in BOTH arms'
    hit vectors contribute (arm-specific missing ranks never misalign).
    """
    out = {}
    for label in ("all_selected", "all", "target_present",
                  "eligible_parseable", "capability_any_candidate"):
        entry = topk[label]
        nq = entry.get("qids", [])
        for other in ("uniform", "massresid", "prior"):
            pa = entry["predictor_hitvec_top1"]
            pb = entry[f"{other}_hitvec_top1"]
            if isinstance(pa, dict) and isinstance(pb, dict):
                common = [q for q in nq if q in pa and q in pb]
                a = [pa[q] for q in common]
                b = [pb[q] for q in common]
            else:
                a, b, common = list(pa), list(pb), list(nq)
            n = len(common)
            diff, lo, hi, nab, nba, ntie = paired_diff_ci(a, b, n)
            out[f"{label}.predictor_minus_{other}.top1"] = {
                "diff": diff, "ci95": [lo, hi], "n": n,
                "n_predictor_better": nab, f"n_{other}_better": nba,
                "n_tie": ntie}
    return out


def run_quality_analysis(repo_root, out_dir, max_proxy=5000):
    t0 = time.process_time()
    wall0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(repo_root, "data/pinned/MassSpecGym1.5.tsv")
    pin_path = os.path.join(repo_root, "data/pinned/PIN.json")
    rows_path = os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/predictor_rows.csv")
    summary_path = os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/predictor_summary.json")
    fit_path = os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/train_fit_rows.csv")

    with open(pin_path, "r", encoding="utf-8") as handle:
        pin = json.load(handle)
    with open(summary_path, "r", encoding="utf-8") as handle:
        pred_summary = json.load(handle)
    rows = load_predictor_rows(rows_path)
    assert len(rows) == 200, f"frozen 200 expected, got {len(rows)}"
    qids = [r["qid"] for r in rows]
    assert qids == [f"VAL-{i:04d}" for i in range(200)], "qid order drift vs frozen slice"

    # --- top-k + Wilson + paired (fixed existing ranks) ---
    topk = analyze_topk(rows)
    # strip hit vectors from persisted form (keep in memory only for paired)
    paired = analyze_paired(topk)
    for label in topk:
        for arm in ARMS:
            topk[label].pop(f"{arm}_hitvec_top1", None)

    # extended paired diffs for top-3/10/25 from rows directly
    label_sel = {"all_selected": lambda r: True,
                 "all": lambda r: True,
                 "target_present": lambda r: r.get("in_pool") == "True",
                 "eligible_parseable": is_eligible_parseable,
                 "capability_any_candidate": lambda r: r.get("rankable") == "True"}
    for label, sel in label_sel.items():
        sub = [r for r in rows if sel(r)]
        n = len(sub)
        for other in ("uniform", "massresid", "prior"):
            for k in (3, 10, 25):
                a = [hit_at_k(r.get("rank_predictor"), k) for r in sub]
                b = [hit_at_k(r.get(f"rank_{other}"), k) for r in sub]
                diff, lo, hi, nab, nba, ntie = paired_diff_ci(a, b, n)
                paired[f"{label}.predictor_minus_{other}.top{k}"] = {
                    "diff": diff, "ci95": [lo, hi], "n": n,
                    "n_predictor_better": nab, f"n_{other}_better": nba,
                    "n_tie": ntie}

    # --- pool recall (all selected; frozen in_pool flags + recompute) ---
    n_in_pool = sum(1 for r in rows if r.get("in_pool") == "True")
    pool_lo, pool_hi = wilson_interval(n_in_pool, len(rows))

    # --- per-query quality CSV (top-25 added; ranks reused verbatim) ---
    quality_csv = os.path.join(out_dir, "quality_rows.csv")
    with open(quality_csv, "w", encoding="utf-8", newline="") as handle:
        fields = ["qid", "smiles", "adduct", "instrument", "in_pool",
                  "target_parseable", "rankable", "rankable_reason",
                  "n_pool", "n_parseable", "margin_status"]
        for arm in ARMS:
            fields.append(f"rank_{arm}")
            for k in KS:
                fields.append(f"hit_{arm}_top{k}")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ("qid", "smiles", "adduct", "instrument", "in_pool",
                    "target_parseable", "rankable", "rankable_reason",
                    "n_pool", "n_parseable", "margin_status")}
            for arm in ARMS:
                out[f"rank_{arm}"] = r.get(f"rank_{arm}", "")
                for k in KS:
                    out[f"hit_{arm}_top{k}"] = "True" if hit_at_k(r.get(f"rank_{arm}"), k) else "False"
            writer.writerow(out)

    # --- read-only TSV identity/overlap/adduct audit (test: overlap only) ---
    progress("scanning TSV identities (read-only; test fold overlap-audit only)")
    folds, n_rows = scan_tsv_identities(tsv_path)
    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in folds and b in folds:
            sa, sb = folds[a]["smiles"], folds[b]["smiles"]
            ka, kb = folds[a]["keys"], folds[b]["keys"]
            overlap[f"{a}_vs_{b}"] = {
                "smiles_intersection": len(sa & sb),
                "smiles_union": len(sa | sb),
                "inchikey14_intersection": len(ka & kb),
                "inchikey14_union": len(ka | kb),
            }
    # selected-200 adduct/instrument bias vs val overall
    sel_ad = {}
    sel_inst = {}
    for r in rows:
        ad = (r.get("adduct") or "missing").strip() or "missing"
        sel_ad[ad] = sel_ad.get(ad, 0) + 1
        inst = (r.get("instrument") or "missing").strip() or "missing"
        sel_inst[inst] = sel_inst.get(inst, 0) + 1
    unsupported_adduct_val = sum(
        c for ad, c in folds.get("val", {}).get("adduct_counts", {}).items()
        if ad not in SUPPORTED_ADDUCTS)

    # --- structural-family proxy on capped identity set (val targets + fit/calib) ---
    from tools.ms2_msgym_corpus import standardize_smiles as _std

    def _std_cached(smi):
        return _std(smi, "quality-proxy", smi)

    with open(fit_path, "r", encoding="utf-8") as handle:
        fit_rows = list(csv.DictReader(handle))
    fit_smiles = [r["smiles"] for r in fit_rows if r.get("smiles")]
    val_smiles = [r["smiles"] for r in rows if r.get("smiles")]
    proxy_ids = list(dict.fromkeys(val_smiles + fit_smiles))
    groups, proxy_status = structural_family_proxy(proxy_ids, _std_cached, cap=max_proxy)
    proxy_status["n_groups"] = len(groups)
    proxy_status["n_unique_ids"] = len(proxy_ids)
    multi = sum(1 for v in groups.values() if len(v) > 1)
    proxy_status["n_multi_member_groups"] = multi

    # --- exact Murcko scaffolds via project-local RDKit (pinned) ---
    # Identity policy: RDKit MurckoScaffoldSmilesFromSmiles with
    # includeChirality=False (stereo-insensitive connectivity identity of
    # the Bemis-Murcko framework). Acyclic molecules have NO scaffold:
    # they are counted as unparseable-for-scaffold, never grouped under a
    # placeholder. Overlap is reported for source(selected200)/fit/val/test
    # sets with the genuine versioned algorithm recorded below.
    try:
        from tools.ms2_external_validation import murcko_of, rdkit_lib
        _rd = rdkit_lib()
        _rd_version = getattr(_rd, "__version__", "unknown") if _rd else None
    except Exception:
        _rd, _rd_version = None, None
    if _rd is not None:
        def _scaf_set(smiles_list):
            out = {}
            n_fail = 0
            for smi in smiles_list:
                sc, _ = murcko_of(smi)
                if sc is None:
                    n_fail += 1
                    continue
                out.setdefault(sc, []).append(smi)
            return out, n_fail

        sel_scaf, sel_fail = _scaf_set(val_smiles)
        fit_scaf, fit_fail = _scaf_set(fit_smiles)
        test_set = set()
        if "test" in folds:
            test_set = folds["test"]["smiles"]
        test_scaf, test_fail = _scaf_set(sorted(test_set)[:20000])
        exact_scaffold = {
            "gate": "COMPLETE",
            "method": ("rdkit.Chem.Scaffolds.MurckoScaffold "
                       "(MurckoScaffoldSmilesFromSmiles, includeChirality=False; "
                       "stereo-insensitive connectivity identity; acyclic "
                       "molecules have no scaffold and are counted, never grouped)"),
            "rdkit_version": _rd_version,
            "rdkit_scope": ("project-local "
                            "experiments/molecular_completion/20261003_remaining/.rdkit_lib"),
            "n_selected200_scaffolds": len(sel_scaf),
            "n_fit_scaffolds": len(fit_scaf),
            "n_val_scaffolds": len(sel_scaf),
            "n_test_scaffolds": len(test_scaf),
            "selected_vs_fit_intersection": len(set(sel_scaf) & set(fit_scaf)),
            "selected_vs_test_intersection": len(set(sel_scaf) & set(test_scaf)),
            "selected_unparseable_for_scaffold": sel_fail,
            "fit_unparseable_for_scaffold": fit_fail,
        }
    else:
        exact_scaffold = {
            "gate": "INCOMPLETE: RDKit unavailable project-locally; no proxy substituted",
            "method": None,
        }

    # --- PubChem conditional: MEASURED misses decide (internal + external) ---
    internal_misses = [r["qid"] for r in rows if r.get("in_pool") != "True"]
    ext_path = os.path.join(out_dir, "external_summary.json")
    if os.path.exists(ext_path):
        try:
            with open(ext_path, "r", encoding="utf-8") as handle:
                ext_summary = json.load(handle)
            ext_misses = ext_summary.get("pubchem_miss_queries", [])
        except (ValueError, OSError):
            ext_summary, ext_misses = None, None
    else:
        ext_summary, ext_misses = None, None
    pubchem = compute_pubchem_status(internal_misses, ext_misses, api_calls=0)
    pubchem["measured_pool_recall"] = f"{n_in_pool}/{len(rows)}"
    pubchem["external_summary_present"] = ext_summary is not None
    if ext_summary is not None:
        pubchem["external_expansion"] = ext_summary.get("pubchem_conditional", {}).get("status")
        pubchem["external_api_calls"] = ext_summary.get("costs", {}).get("api_calls", 0)

    # --- official MassSpecGym fingerprint baseline: pinned evidence ---
    # Verified 2026-10-03 via api.github.com (pinned main HEAD f259fe37,
    # 2026-05-08) and raw.githubusercontent.com file content, plus a bounded
    # execution attempt with installed torch 2.13 + project-local RDKit
    # against cached massspecgym==1.3.1 sources. No dataset size is claimed:
    # no byte total is pinned, so none is stated.
    official = {
        "identified_candidate": ("retrieval baselines in pluskal-lab/MassSpecGym "
                                 "massspecgym/models/retrieval/: DeepSets "
                                 "(deepsets.py), fingerprint FFN "
                                 "(fingerprint_ffn.py), random (random.py); "
                                 "spectrum/mol transforms in "
                                 "massspecgym/data/transforms.py "
                                 "(SpecTokenizer/SpecBinner, MolFingerprinter)"),
        "pinned_source": {
            "repo": "https://github.com/pluskal-lab/MassSpecGym",
            "license": "MIT (verified via api.github.com repo record)",
            "main_head_sha": "f259fe3780d5bd227fc6ece36ce6f397c2eef716",
            "main_head_date": "2026-05-08T14:46:41Z",
            "verified": "2026-10-03",
            "files": {
                "massspecgym/models/retrieval/deepsets.py": "sha ba405e7f6f8f481a78ad8c96b91874b5f59ea059",
                "massspecgym/models/retrieval/fingerprint_ffn.py": "sha 39e80e3e4f7eb843b8650175a3b2d2d9c76a0f18",
                "massspecgym/models/retrieval/random.py": "sha 066b2b75795a5c5a70fbf027607278d789547f53",
                "massspecgym/data/transforms.py": "sha 1c3d3151a6cdeb9ebc0c843469923fa5ba16af24",
            },
            "releases_checked": "v1.3.1 (2025-03-23), v1.2.2: zero release assets",
            "pretrained_checkpoint": ("none published: both releases carry no "
                                      "assets; no pinned .ckpt weights found"),
        },
        "cached_package_inspection": {
            "package": "massspecgym==1.3.1 (cached /tmp/opencode/msgym, pip metadata)",
            "files": {
                "massspecgym/data/transforms.py": "sha256 eb6e93a31584b7e896582819d3e3438ebcadd2bda5052ef334cef600512b792e (14306 B)",
                "massspecgym/models/retrieval/deepsets.py": "sha256 f16b9de9b066016e91be708e4a27a305f7b68be6108c809072881c26739a7b61 (3518 B)",
                "massspecgym/models/retrieval/base.py": "sha256 9c5c3abd00e96530b2be8f4954c9e5c5d06f8a2fec29307678f27c9416032fee (7734 B)",
                "massspecgym/models/retrieval/fingerprint_ffn.py": "sha256 0f67462399fef93ffd9a0515a3500f4f96b2154166e68ba0cb4ed5f3936b83c0 (1999 B)",
                "massspecgym/models/base.py": "sha256 948461fd1f35f0666676ddd17fae91fcd3bf97bf1484440f4eab659dab6e6639 (6615 B)",
                "massspecgym/utils.py": "sha256 6220caa6af4bdba1682135ca47b69290ef13f5e941e925d4962af43ad1595559 (16730 B)",
                "massspecgym/definitions.py": "sha256 47b381bdbfa0f1a537c1ef7dd45e219731733df0d12892077db0be637571effd (1264 B)",
            },
            "note": "exact cached contents hashed, not altered or stubbed",
        },
        "hf_dataset_checkpoint_check": {
            "listing": "experiments/molecular_completion/20261003_remaining/hf_dataset_listing.json",
            "listing_sha256": "f242dde4dae139ee",
            "n_files": 26,
            "ckpt_files": ["models/DreaMS_embedding_model_MassSpecGym.ckpt",
                           "models/mist_cf_fast_filter.ckpt",
                           "models/mist_cf_msg.ckpt"],
            "retrieval_checkpoint": ("NONE: the three .ckpt files are DreaMS "
                                     "embedding + MIST-CF simulation/annotation "
                                     "filters, not a DeepSets/fingerprint-FFN "
                                     "retrieval checkpoint"),
        },
        "bounded_execution_attempt": {
            "environment": "installed torch 2.13.0+cu130 + project-local RDKit; "
                           "cached massspecgym==1.3.1 + matchms/torchmetrics/"
                           "torch_geometric/pulp/myopic-mces/lightning deps "
                           "present (no new downloads)",
            "error": ("massspecgym/utils.py:351 evaluates "
                      "pulp.listSolvers(onlyAvailable=True)[0] at import time; "
                      "zero LP solver binaries installed -> IndexError: list "
                      "index out of range. The official import chain "
                      "(transforms, deepsets, retrieval base) requires "
                      "massspecgym.utils and fails before any model code runs."),
            "verdict": ("plain official model code CANNOT execute exactly "
                        "with existing dependencies; altering/stubbing it "
                        "would no longer be official and was not done"),
        },
        "requirements_evidence": ("massspecgym/data/transforms.py (pinned sha "
                                  "above) imports torch, matchms "
                                  "(matchms.filtering), rdkit "
                                  "(rdkit.Chem.AllChem), torch_geometric "
                                  "(torch_geometric.data.Batch); MolFingerprinter "
                                  "defaults type='morgan' fp_size=2048 in the "
                                  "pinned file (no 4096-bit claim made)"),
        "prior_dependency_incident": ("a prior pip attempt fetched a duplicate "
                                      "multi-GB CUDA torch tree into scratch, "
                                      "hit disk quota, and was removed; total "
                                      "prior download bytes cannot be "
                                      "reconstructed -> reported as "
                                      "unknown/exceeded, never <1GB"),
        "status": ("blocked (concrete): no published retrieval checkpoint "
                   "(releases: zero assets; HF: simulation/embedding .ckpt "
                   "only; README: train-yourself example only), and the "
                   "official stack cannot import with existing dependencies "
                   "(pulp LP-solver IndexError, no solver binaries, no "
                   "further downloads allowed). Training the official "
                   "DeepSets architecture is out of scope (no checkpoint, "
                   "HF data download, Lightning training). The local Ridge "
                   "hash-fingerprint model is NOT the official baseline and "
                   "is never presented as such"),
        "ran_here": False,
    }

    # --- costs: FULL run incl. fit/calibration + per-1000 normalization ---
    cost = pred_summary.get("cost", {})
    wall_s = float(cost.get("wall_s", 0.0))
    cpu_s = float(cost.get("cpu_s", 0.0))
    rss_kb = int(cost.get("peak_rss_kb", 0))
    fit_s = float(pred_summary.get("train", {}).get("train_time_s", 0.0))
    n_fit = int(pred_summary.get("train", {}).get("n_fit", 0))
    n_calib = int(pred_summary.get("train", {}).get("n_calib", 0))
    per1000 = {"denominator": ("200 frozen val queries; FULL run cost covers "
                               f"{n_fit} fit + {n_calib} calib molecules plus "
                               "retrieval scoring (declared scope in "
                               "predictor_summary.json cost.note)"),
               "wall_s_per_1000": wall_s / 200 * 1000,
               "cpu_s_per_1000": cpu_s / 200 * 1000,
               "fit_time_s_included": fit_s,
               "retrieval_plus_overhead_s_cpu": cpu_s - fit_s}

    cpu_s_own = time.process_time() - t0
    wall_s_own = time.time() - wall0
    rss_own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    summary = {
        "tool": "tools/ms2_quality_audit.py",
        "frozen_slice": {"n": 200, "qids": "VAL-0000..VAL-0199",
                         "ranks_reused_verbatim": True, "no_val_tuning": True,
                         "no_refit": True},
        "topk": topk,
        "paired_predictor_vs_baseline": paired,
        "pool_recall_all": {"hits": n_in_pool, "n": len(rows),
                             "rate": n_in_pool / len(rows),
                             "ci95": [pool_lo, pool_hi]},
        "denominators": {
            "tsv_rows_scanned": n_rows,
            "folds": {f: {"n_rows": d["n_rows"],
                          "n_distinct_smiles": len(d["smiles"]),
                          "n_distinct_inchikey14": len(d["keys"]),
                          "adduct_counts": d["adduct_counts"],
                          "instrument_counts": d["inst_counts"],
                          "n_spectrum_ok": d["n_spectrum_ok"],
                          "n_spectrum_bad": d["n_spectrum_bad"]}
                      for f, d in folds.items()},
            "molecule_overlap": overlap,
            "structural_family_proxy": proxy_status,
            "exact_scaffolds": exact_scaffold,
            "unsupported_adduct_val_rows": unsupported_adduct_val,
            "supported_adducts": list(SUPPORTED_ADDUCTS),
            "selected200_adduct": sel_ad,
            "selected200_instrument": sel_inst,
            "test_fold_use": "identity/scaffold overlap audit only (read-only); never scored or trained on",
        },
        "pubchem_conditional": pubchem,
        "official_baseline": official,
        "costs": {"prior_run": {"wall_s": wall_s, "cpu_s": cpu_s, "peak_rss_kb": rss_kb},
                  "per_1000_queries": per1000,
                  "audit_own": {"wall_s": wall_s_own, "cpu_s": cpu_s_own,
                                "peak_rss_kb": rss_own}},
        "pins": {"tsv_sha256": sha256_file(tsv_path),
                 "predictor_rows_sha256": sha256_file(rows_path),
                 "pin_json": pin},
        "env": {"python": sys.version.split()[0]},
        "code_sha256": sha256_file(os.path.join(repo_root, "tools/ms2_quality_audit.py")),
        "outputs": {"rows": quality_csv},
        "adequacy": ("CIs reported for every rate; sample stays at frozen n=200 for "
                     "comparisons; any expansion would need predeclared stable sampling "
                     "and is NOT performed here"),
    }
    # strict JSON: no NaN/Infinity allowed
    text = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
    summary_path_out = os.path.join(out_dir, "quality_summary.json")
    with open(summary_path_out, "w", encoding="utf-8") as handle:
        handle.write(text)
    summary["outputs"]["summary"] = summary_path_out
    return summary


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/molecular_completion/20261003_remaining")
    ap.add_argument("--max-proxy", type=int, default=5000)
    args = ap.parse_args(argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # when invoked as tools.ms2_quality_audit, abspath dirname is tools/
    if os.path.basename(repo_root) == "tools":
        repo_root = os.path.dirname(repo_root)
    summary = run_quality_analysis(repo_root, args.out_dir, max_proxy=args.max_proxy)
    print(json.dumps({"n": 200,
                      "pool_recall": summary["pool_recall_all"]["rate"],
                      "predictor_top1_all": summary["topk"]["all"]["predictor"]["top1"],
                      "out": summary["outputs"]["summary"]}, allow_nan=False))


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

class WilsonTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_edges_inside(self):
        lo, hi = wilson_interval(0, 200)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.0 < hi < 0.05)
        lo, hi = wilson_interval(200, 200)
        self.assertTrue(0.95 < lo < 1.0)
        self.assertEqual(hi, 1.0)

    def test_contains_point(self):
        lo, hi = wilson_interval(12, 200)
        self.assertTrue(lo <= 12 / 200 <= hi)

    def test_bad_hits(self):
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)


class PairedTests(unittest.TestCase):
    def test_identical(self):
        d, lo, hi, nab, nba, nt = paired_diff_ci([1, 0, 1], [1, 0, 1], 3)
        self.assertEqual((d, nab, nba, nt), (0.0, 0, 0, 3))

    def test_direction(self):
        d, lo, hi, nab, nba, nt = paired_diff_ci([1, 1, 0], [1, 0, 0], 3)
        self.assertAlmostEqual(d, 1 / 3)
        self.assertEqual((nab, nba, nt), (1, 0, 2))

    def test_empty(self):
        d, lo, hi, nab, nba, nt = paired_diff_ci([], [], 0)
        self.assertEqual((nab, nba, nt), (0, 0, 0))
        self.assertIsNone(lo)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            paired_diff_ci([1], [1, 0], 2)


class HitTests(unittest.TestCase):
    def test_none_absent(self):
        self.assertFalse(hit_at_k(None, 25))
        self.assertFalse(hit_at_k("", 25))
        self.assertFalse(hit_at_k("nan", 3))

    def test_bounds(self):
        self.assertTrue(hit_at_k("25", 25))
        self.assertFalse(hit_at_k("26", 25))
        self.assertFalse(hit_at_k("0", 1))


class ProxyTests(unittest.TestCase):
    def test_proxy_groups_and_cap(self):
        from tools.ms2_msgym_corpus import standardize_smiles as _std

        def _s(smi):
            return _std(smi, "t", smi)

        groups, status = structural_family_proxy(["CCO", "CCO", "CCC", "not a smiles["], _s, cap=3)
        self.assertEqual(status["n_done"], 3)
        self.assertEqual(status["n_truncated"], 1)
        self.assertIn("exact_scaffold_gate", status)
        self.assertIn("INCOMPLETE", status["exact_scaffold_gate"])


class PubchemTriggerTests(unittest.TestCase):
    def test_no_misses_not_triggered(self):
        st = compute_pubchem_status([], None)
        self.assertEqual(st["measured_misses"], 0)
        self.assertIn("not-triggered", st["status"])
        self.assertEqual(st["api_calls"], 0)

    def test_internal_miss_triggers(self):
        st = compute_pubchem_status(["VAL-0001", "VAL-0002"], [])
        self.assertEqual(st["measured_misses"], 2)
        self.assertIn("triggered", st["status"])

    def test_external_misses_included(self):
        st = compute_pubchem_status([], ["EXT-003"])
        self.assertEqual(st["measured_misses"], 1)
        self.assertIn("triggered", st["status"])

    def test_external_unmeasured_noted(self):
        st = compute_pubchem_status([], None)
        self.assertIn("not yet measured", st["external_note"])


class DenominatorTests(unittest.TestCase):
    def _rows(self):
        return [
            {"qid": "Q0", "status": "complete", "in_pool": "True",
             "target_parseable": "True",
             "rankable": "True", "rank_uniform": "1", "rank_massresid": "",
             "rank_predictor": "5", "rank_prior": "30"},
            {"qid": "Q1", "status": "complete", "in_pool": "True",
             "target_parseable": "False",
             "rankable": "True", "rank_uniform": "", "rank_massresid": "2",
             "rank_predictor": "", "rank_prior": ""},
            {"qid": "Q2", "status": "complete", "in_pool": "False",
             "target_parseable": "False",
             "rankable": "False", "rank_uniform": "", "rank_massresid": "",
             "rank_predictor": "", "rank_prior": ""},
        ]

    def test_absent_rank_is_miss_but_counted(self):
        rows = self._rows()
        topk = analyze_topk(rows)
        self.assertEqual(topk["all_selected"]["n"], 3)
        self.assertEqual(topk["target_present"]["n"], 2)
        # Q0 lacks a valid massresid rank -> not eligible (ranks must apply)
        self.assertEqual(topk["eligible_parseable"]["n"], 0)
        self.assertEqual(topk["capability_any_candidate"]["n"], 2)
        # absent/empty rank cells are misses, still in denominator
        self.assertEqual(topk["all_selected"]["uniform"]["top1"]["hits"], 1)
        self.assertEqual(topk["all_selected"]["uniform"]["top1"]["n"], 3)

    def test_excluded_identity_missing_not_eligible(self):
        # Review fixture: parseable True but absent from pool with an
        # excluded identity status must NOT count as eligible.
        rows = self._rows() + [
            {"qid": "Q3", "status": "excluded_identity_missing",
             "in_pool": "False", "target_parseable": "True",
             "rankable": "False", "rank_uniform": "", "rank_massresid": "",
             "rank_predictor": "", "rank_prior": ""},
        ]
        topk = analyze_topk(rows)
        self.assertEqual(topk["all_selected"]["n"], 4)
        self.assertEqual(topk["target_present"]["n"], 2)
        self.assertEqual(topk["eligible_parseable"]["n"], 0)
        # all-selected recall retains it as a miss
        self.assertEqual(topk["all_selected"]["uniform"]["top1"]["n"], 4)

    def test_eligible_requires_complete_present_parseable_ranks(self):
        rows = [
            {"qid": "Q0", "status": "complete", "in_pool": "True",
             "target_parseable": "True", "rankable": "True",
             "rank_uniform": "1", "rank_massresid": "2",
             "rank_predictor": "5", "rank_prior": "30"},
        ]
        topk = analyze_topk(rows)
        self.assertEqual(topk["eligible_parseable"]["n"], 1)
        self.assertEqual(topk["eligible_parseable"]["uniform"]["top1"]["hits"], 1)

    def test_paired_matches_same_qid(self):
        rows = [
            {"qid": "Q0", "status": "complete", "in_pool": "True",
             "target_parseable": "True", "rankable": "True",
             "rank_uniform": "5", "rank_massresid": "9",
             "rank_predictor": "1", "rank_prior": "9"},
            {"qid": "Q1", "status": "complete", "in_pool": "True",
             "target_parseable": "True", "rankable": "True",
             "rank_uniform": "1", "rank_massresid": "9",
             "rank_predictor": "5", "rank_prior": "9"},
        ]
        topk = analyze_topk(rows)
        paired = analyze_paired(topk)
        # predictor better on Q0 only, uniform better on Q1 only
        key = "all_selected.predictor_minus_uniform.top1"
        self.assertEqual(paired[key]["n"], 2)
        self.assertEqual(paired[key]["n_predictor_better"], 1)
        self.assertEqual(paired[key]["n_uniform_better"], 1)

    def test_hit_bounds_with_none(self):
        self.assertFalse(hit_at_k(None, 1))
        self.assertFalse(hit_at_k("", 25))
        self.assertTrue(hit_at_k("25", 25))
        self.assertFalse(hit_at_k("26", 25))


class ScafTests(unittest.TestCase):
    def test_exact_murcko_when_available(self):
        try:
            from tools.ms2_external_validation import murcko_of, rdkit_lib
            rd = rdkit_lib()
        except Exception:
            rd = None
        if rd is None:
            self.skipTest("RDKit project-local unavailable")
        sc, reason = murcko_of("CC(C)(O)c1ccco1")
        self.assertEqual(sc, "c1ccoc1")
        sc2, _ = murcko_of("CCO")
        self.assertIsNone(sc2)  # acyclic: no ring scaffold


class StrictJsonTests(unittest.TestCase):
    def test_no_nan(self):
        bad = {"x": float("nan")}
        with self.assertRaises(ValueError):
            json.dumps(bad, allow_nan=False)


if __name__ == "__main__":
    main(sys.argv[1:])
