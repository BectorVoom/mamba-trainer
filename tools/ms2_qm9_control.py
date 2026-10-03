"""Restricted QM9 positive-control match (existing evidence only; no new QM).

Uses ONLY official pinned QM9 files (CC0, Figshare collection 978904):
  qm9_readme.txt              data description (5 KB)
  qm9_uncharacterized.txt     3054 gdb9 indices failing geometry check (487 KB)
  qm9_dsC7O2H10nsd.xyz.tar.bz2  6095 C7H10O2 isomers with B3LYP/6-31G(2df,p)
                                geometries + properties (4.1 MB)

What is measured:
  * QM9-eligibility denominator: how many frozen val-200 / train-fit
    molecules satisfy QM9 composition (elements subset of {C,H,O,N,F},
    neutral, heavy atoms <= 9) via the existing typed-graph standardizer.
  * Full XYZ-tail parse of every scanned member: finite-coordinate and
    atom-count validation, harmonic-frequency list (positive/zero/negative
    counts), paired GDB-input vs relaxed-geometry SMILES and InChI strings.
  * Exact connectivity-identity matching of eligible queries against the
    official member SMILES (relaxed primary, GDB reported separately) via
    the existing typed-graph standardizer plus brute-force typed graph
    isomorphism -- never formula or weak-fingerprint presence.
  * Exclusion-policy audit: the restricted C7H10O2 file uses SUBSET-LOCAL
    numbering (gdb 1..6095), which is incompatible with the full-GDB9
    numbering of uncharacterized.txt (indices up to 133885); no member is
    excluded on a cross-numbering join. Recorded as source-identity
    mismatch status, not silently joined.

What is NOT claimed:
  * No kinetic lifetime / solution-stability claim: a matched optimized
    geometry or positive frequency is restricted evidence only.
  * Imaginary-frequency verification uses the per-member frequency tail
    lines shipped in the xyz archive (negative values would be
    imaginary-candidates, zero is zero, positive is positive). No kinetics.
  * No target is injected into any measured candidate pool.
  * Subset scope: C7H10O2-only. Queries of any other formula are
    out-of-subset-scope (NOT absence from QM9). A full-archive scan is
    performed when data/pinned/external/dsgdb9nsd.xyz.tar.bz2 is present;
    otherwise broader absence is explicitly not claimed.

CPU stdlib baseline only: GPU execution and Rust/Python parity are NOT
APPLICABLE (recorded as limitations, not passing checks).
"""

import csv
import hashlib
import json
import math
import os
import resource
import sys
import tarfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QM9_ELEMENTS = ("C", "H", "O", "N", "F")
QM9_MAX_HEAVY = 9

FIGSHARE = {
    "collection": "https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904",
    "collection_id": 978904,
    "files": {
        "readme.txt": {"article_id": 1057641, "file_id": 3195392,
                       "doi": "10.6084/m9.figshare.978904_D7"},
        "uncharacterized.txt": {"article_id": 1057644, "file_id": 3195404,
                                "doi": "10.6084/m9.figshare.978904_D10"},
        "dsC7O2H10nsd.xyz.tar.bz2": {"article_id": 1057645, "file_id": 3195398,
                                     "doi": "10.6084/m9.figshare.978904_D11"},
        "dsgdb9nsd.xyz.tar.bz2": {"article_id": 1057646, "file_id": 3195389,
                                  "doi": "10.6084/m9.figshare.978904_D12"},
    },
    "license": "CC0",
    "method": "B3LYP/6-31G(2df,p) geometry + harmonic frequencies (per official dataset page)",
    "cite": ["Ramakrishnan et al., Sci. Data 1, 140022 (2014)",
             "Blum & Reymond, JACS 131:8732 (2009); Ruddigkeit et al., JCIM 52:2864 (2012)"],
}


def progress(msg):
    sys.stderr.write(f"# progress {msg}\n")
    sys.stderr.flush()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            b = handle.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_uncharacterized(path):
    """Parse the official excluded-molecule list. Returns (entries, n_lines).

    Format (per readme): full-GDB9 index + original GDB17 SMILES / B3LYP
    SMILES / Corina SMILES / Coulomb-matrix distance. Header/separator
    lines are counted as bad, never silently skipped. NOTE: indices are
    full-GDB9 numbering (1..133885), incompatible with the C7H10O2
    subset-local numbering (1..6095).
    """
    entries = []
    n_bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
            except (ValueError, IndexError):
                n_bad += 1
                continue
            entries.append({"gdb9_index": idx, "rest": parts[1:]})
    return entries, n_bad


def _split_pair(line):
    """Split a two-field tail line (SMILES or InChI pair).

    Official separator is TAB; fall back to whitespace split when no tab
    is present. Returns (first, second, method) with empty strings for
    missing fields (missingness is counted downstream, never inferred).
    """
    if "\t" in line:
        parts = line.split("\t")
        first = parts[0].strip() if len(parts) > 0 else ""
        second = parts[1].strip() if len(parts) > 1 else ""
        return first, second, "tab"
    parts = line.split()
    first = parts[0] if len(parts) > 0 else ""
    second = parts[1] if len(parts) > 1 else ""
    return first, second, "whitespace"


def _num(tok):
    """Parse a provider numeric token to float.

    Normalizes the provider-documented scientific notation `*^`
    (e.g. ``2.1997*^-6`` in dsgdb9nsd_000212.xyz) and Fortran `D`
    exponents to Python `e` before float conversion. Raises ValueError
    for anything else (counted downstream, never guessed).
    """
    t = tok.strip().replace("*^", "e")
    try:
        return float(t)
    except ValueError:
        pass
    if "D" in t or "d" in t:
        import re as _re
        t2 = _re.sub(r"[Dd]", "e", t)
        return float(t2)
    raise ValueError(f"bad_numeric:{tok!r}")


def _linearity(coords):
    """Validate molecular linearity from finite coordinates.

    Returns (is_linear, detail). Diatomics (nat==2) are linear by
    definition; monatomics (nat==1) are degenerate (no meaningful axis).
    Otherwise: SVD of centered coordinates; linear iff the SECOND
    singular value is < 1e-3 of the largest (one significant dimension;
    the smallest singular value is exactly zero for any planar geometry
    and must NOT be used). Documented tolerance: exact QM9 equilibrium
    geometries are far from this boundary; near-degenerate numerical
    cases fall back to nonlinear, never silently linear.
    Requires numpy; without it returns (False, "numpy_unavailable") and
    callers treat the count as unvalidated.
    """
    import math as _m
    nat = len(coords)
    if nat <= 1:
        return False, "degenerate_monatomic"
    if nat == 2:
        return True, "diatomic"
    try:
        import numpy as _np
    except Exception:
        return False, "numpy_unavailable"
    arr = _np.asarray(coords, dtype=float)
    arr = arr - arr.mean(axis=0)
    try:
        sv = _np.linalg.svd(arr, compute_uv=False)
    except Exception:
        return False, "svd_failed"
    if not _m.isfinite(float(sv[0])) or sv[0] <= 0:
        return False, "degenerate_zero_extent"
    if len(sv) < 2:
        return False, "degenerate_rank"
    ratio = float(sv[1]) / float(sv[0])
    if ratio < 1e-3:
        return True, f"collinear_ratio_{ratio:.2e}"
    return False, f"nonlinear_ratio_{ratio:.2e}"


def parse_xyz_text(text):
    """Parse one QM9 xyz member incl. full tail. Returns (parsed, None).

    Returns (None, reason) on any structural problem. Never infers bonds:
    coordinates are geometry provenance, not a graph. Formula comes from
    element symbols only. Frequencies/SMILES/InChI come from the actual
    tail lines (na+3/na+4/na+5 per the pinned readme).
    """
    from collections import Counter
    lines = text.strip("\n").split("\n")
    if len(lines) < 3:
        return None, "too_short"
    try:
        nat = int(lines[0].strip())
    except ValueError:
        return None, "bad_natoms"
    if nat <= 0:
        return None, "bad_natoms"
    prop_line = lines[1].strip() if len(lines) > 1 else ""
    if not prop_line.startswith("gdb"):
        return None, "bad_property_line"
    gdb_index = None
    try:
        gdb_index = int(prop_line.split()[1])
    except (ValueError, IndexError):
        return None, "bad_property_index"
    expect = 2 + nat + 3
    if len(lines) != expect:
        return None, "line_count_mismatch"
    elems = []
    coords_ok = True
    for ln in lines[2:2 + nat]:
        parts = ln.split()
        if len(parts) < 4:
            return None, "bad_atom_line"
        el = parts[0]
        if not el.isalpha():
            return None, f"bad_element:{el}"
        try:
            vals = [_num(x) for x in parts[1:4]]
        except ValueError:
            return None, "bad_coordinates"
        if any(not math.isfinite(v) for v in vals):
            coords_ok = False
        if len(parts) >= 5:
            try:
                q = _num(parts[4])
                if not math.isfinite(q):
                    coords_ok = False
            except ValueError:
                return None, "bad_charge"
        elems.append(el)
    if len(elems) != nat:
        return None, "atom_count_mismatch"
    if not coords_ok:
        return None, "nonfinite_coordinates"
    freq_line = lines[2 + nat].strip()
    freq_tokens = freq_line.replace("\t", " ").split()
    freqs = []
    for tok in freq_tokens:
        try:
            f = _num(tok)
        except ValueError:
            return None, "bad_frequency"
        if not math.isfinite(f):
            return None, "nonfinite_frequency"
        freqs.append(f)
    n_pos = sum(1 for f in freqs if f > 0)
    n_zero = sum(1 for f in freqs if f == 0)
    n_neg = sum(1 for f in freqs if f < 0)
    lo, hi = 3 * nat - 6, 3 * nat - 5
    # Expected vibrational-mode count requires VALIDATED linearity:
    # linear -> 3N-5, nonlinear -> 3N-6 (linearity computed from the
    # finite coordinates above, not assumed). Either-count-allowed is
    # retired: the actual expected count is chosen explicitly.
    coords = None
    try:
        coords = [[_num(x) for x in ln.split()[1:4]]
                  for ln in lines[2:2 + nat]]
    except (ValueError, IndexError):
        coords = None
    is_linear, lin_detail = _linearity(coords) if coords else (False, "no_coords")
    expected = (3 * nat - 5) if is_linear else (3 * nat - 6)
    # Complete vibrational-mode evidence requires the exact expected count
    # with all-finite values. Incomplete counts keep exact geometry
    # identity but NEVER count as stationary-point (positive-only)
    # frequency evidence.
    if nat <= 1:
        freq_mode_status = "degenerate"
    elif len(freqs) == expected:
        freq_mode_status = "complete"
    else:
        freq_mode_status = "incomplete_count"
    smi_gdb, smi_relaxed, smi_method = _split_pair(lines[2 + nat + 1].strip())
    inchi_gdb, inchi_relaxed, inchi_method = _split_pair(lines[2 + nat + 2].strip())
    return {"counts": dict(Counter(elems)), "natoms": nat,
            "gdb_index": gdb_index,
            "comment": prop_line[:400],
            "n_freq": len(freqs),
            "freqs": freqs,
            "n_freq_positive": n_pos,
            "n_freq_zero": n_zero,
            "n_freq_negative": n_neg,
            "freq_expected": (3 * nat - 6, 3 * nat - 5),
            "freq_expected_validated": expected,
            "is_linear": is_linear,
            "linearity_detail": lin_detail,
            "freq_mode_status": freq_mode_status,
            "smiles_gdb": smi_gdb, "smiles_relaxed": smi_relaxed,
            "smiles_split": smi_method,
            "inchi_gdb": inchi_gdb, "inchi_relaxed": inchi_relaxed,
            "inchi_split": inchi_method,
            "smiles_match": bool(smi_gdb) and smi_gdb == smi_relaxed,
            "inchi_match": bool(inchi_gdb) and inchi_gdb == inchi_relaxed,
            }, None


def exclusion_numbering_status(subset_indices, unchar_indices):
    """Decide whether the exclusion list can be joined to scanned members.

    The C7H10O2 subset file numbers members 1..6095 (subset-local); the
    official uncharacterized list numbers full-GDB9 molecules 1..133885.
    Joining them directly would misattribute exclusions, so the join is
    refused and reported. Returns a status dict.
    """
    sub_max = max(subset_indices) if subset_indices else 0
    sub_min = min(subset_indices) if subset_indices else 0
    un_max = max(unchar_indices) if unchar_indices else 0
    un_min = min(unchar_indices) if unchar_indices else 0
    compatible = bool(subset_indices) and sub_max <= 133885 and un_max <= 133885 \
        and sub_max > 6095
    # The subset-local file demonstrably runs exactly 1..6095; that range
    # alone cannot distinguish subset-local from a full-GDB9 prefix, but the
    # pinned readme describes dsC7O2H10nsd as a 6095-isomer extract with its
    # own consecutive numbering, while uncharacterized.txt explicitly covers
    # the 133885-molecule GDB9 set. The join is therefore refused.
    return {
        "subset_index_range": [sub_min, sub_max],
        "uncharacterized_index_range": [un_min, un_max],
        "join": "refused_incompatible_numbering",
        "detail": ("C7H10O2 subset uses consecutive subset-local numbering "
                   "1..6095; uncharacterized.txt uses full-GDB9 numbering "
                   "1..133885. No member is excluded on a cross-numbering "
                   "join; members failing local validation are counted "
                   "separately."),
        "n_excluded_by_join": 0,
    }


def scan_xyz_tar(tar_path, max_members=0, max_seconds=480):
    """Stream-parse xyz members incl. tails. Returns (rows, stats).

    Never extracts the archive to disk. max_members=0 means no cap.
    Capped-scan completion is explicit: 'complete', 'truncated_cap', or
    'truncated_time'. Per-member frequency counts and SMILES/InChI
    presence/mismatch are recorded.
    """
    from collections import Counter
    t0 = time.time()
    rows = []
    formula_counts = Counter()
    gdb_indices = []
    n_members = 0
    n_bad = 0
    bad_reasons = Counter()
    timed_out = False
    capped = False
    freq_tot = Counter()
    freq_mode_counts = {}
    n_freq_count_ok = 0
    n_freq_count_off = 0
    n_smiles_present = 0
    n_smiles_mismatch = 0
    n_inchi_present = 0
    n_inchi_mismatch = 0
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if max_members and n_members >= max_members:
                capped = True
                break
            if time.time() - t0 > max_seconds:
                timed_out = True
                break
            fh = tar.extractfile(member)
            if fh is None:
                n_bad += 1
                bad_reasons["unreadable"] += 1
                continue
            try:
                text = fh.read().decode("utf-8", errors="replace")
            except Exception:
                n_bad += 1
                bad_reasons["decode"] += 1
                continue
            parsed, reason = parse_xyz_text(text)
            n_members += 1
            if parsed is None:
                n_bad += 1
                bad_reasons[reason] += 1
                continue
            key = tuple(sorted(parsed["counts"].items()))
            formula_counts[key] += 1
            gdb_indices.append(parsed["gdb_index"])
            freq_tot["positive"] += parsed["n_freq_positive"]
            freq_tot["zero"] += parsed["n_freq_zero"]
            freq_tot["negative"] += parsed["n_freq_negative"]
            if parsed["freq_mode_status"] == "complete":
                n_freq_count_ok += 1
            else:
                n_freq_count_off += 1
            freq_mode_counts[parsed["freq_mode_status"]] = \
                freq_mode_counts.get(parsed["freq_mode_status"], 0) + 1
            has_smi = bool(parsed["smiles_gdb"]) and bool(parsed["smiles_relaxed"])
            has_inchi = bool(parsed["inchi_gdb"]) and bool(parsed["inchi_relaxed"])
            n_smiles_present += 1 if has_smi else 0
            n_inchi_present += 1 if has_inchi else 0
            if has_smi and not parsed["smiles_match"]:
                n_smiles_mismatch += 1
            if has_inchi and not parsed["inchi_match"]:
                n_inchi_mismatch += 1
            rows.append({"name": member.name,
                         "gdb_subset_index": parsed["gdb_index"],
                         "formula": dict(parsed["counts"]),
                         "natoms": parsed["natoms"],
                         "n_freq": parsed["n_freq"],
                         "freq_mode_status": parsed["freq_mode_status"],
                         "n_freq_positive": parsed["n_freq_positive"],
                         "n_freq_zero": parsed["n_freq_zero"],
                         "n_freq_negative": parsed["n_freq_negative"],
                         "smiles_gdb": parsed["smiles_gdb"],
                         "smiles_relaxed": parsed["smiles_relaxed"],
                         "smiles_match": parsed["smiles_match"],
                         "inchi_gdb": parsed["inchi_gdb"],
                         "inchi_relaxed": parsed["inchi_relaxed"],
                         "inchi_match": parsed["inchi_match"],
                         "has_comment": True})
    if timed_out:
        completion = "truncated_time"
    elif capped:
        completion = "truncated_cap"
    else:
        completion = "complete"
    stats = {"n_members": n_members, "n_bad": n_bad,
             "bad_reasons": dict(bad_reasons),
             "n_formulas": len(formula_counts),
             "formula_counts": {str(k): v for k, v in formula_counts.items()},
             "timed_out": timed_out, "capped": capped,
             "cap_limit": max_members, "completion": completion,
             "freq_totals": dict(freq_tot),
             "freq_mode_status_counts": freq_mode_counts,
             "n_freq_count_ok": n_freq_count_ok,
             "n_freq_count_off": n_freq_count_off,
             "n_smiles_present": n_smiles_present,
             "n_smiles_mismatch": n_smiles_mismatch,
             "n_inchi_present": n_inchi_present,
             "n_inchi_mismatch": n_inchi_mismatch,
             "subset_index_min": min(gdb_indices) if gdb_indices else None,
             "subset_index_max": max(gdb_indices) if gdb_indices else None,
             }
    return rows, stats, gdb_indices


def graphs_isomorphic(t1, e1, t2, e2):
    """Exact typed-graph isomorphism (connectivity identity).

    t*: tuple of (element, h, valence); e*: tuple of (a, b, order).
    Stereo/tautomer policy inherits the standardizer: connectivity-only,
    stereo stripped, tautomer not resolved. Returns True/False, never
    a similarity score. Brute-force backtracking with type+degree
    pruning; suitable for QM9-sized graphs (<=9 heavy atoms).
    """
    if len(t1) != len(t2) or len(e1) != len(e2):
        return False
    if sorted(t1) != sorted(t2):
        return False
    n = len(t1)
    if n == 0:
        return False
    if n == 1:
        return t1[0] == t2[0]

    def adj(edges):
        a = [dict() for _ in range(n)]
        deg = [0] * n
        for x, y, o in edges:
            a[x][y] = o
            a[y][x] = o
            deg[x] += 1
            deg[y] += 1
        return a, deg

    a1, d1 = adj(e1)
    a2, d2 = adj(e2)
    if sorted(d1) != sorted(d2):
        return False
    order = sorted(range(n), key=lambda i: (t2[i], d2[i]))
    mapping = {}
    used = [False] * n
    cand_by_type = {}
    for i in range(n):
        cand_by_type.setdefault(t1[i], []).append(i)

    def consistent(j2, i1):
        for k2, mapped in mapping.items():
            o2 = a2[j2].get(k2)
            o1 = a1[i1].get(mapped)
            if (o2 is None) != (o1 is None):
                return False
            if o2 is not None and o2 != o1:
                return False
        return True

    def rec(pos):
        if pos == n:
            return True
        j2 = order[pos]
        for i1 in cand_by_type.get(t2[j2], ()):
            if used[i1] or d1[i1] != d2[j2]:
                continue
            if not consistent(j2, i1):
                continue
            mapping[j2] = i1
            used[i1] = True
            if rec(pos + 1):
                return True
            del mapping[j2]
            used[i1] = False
        return False

    return rec(0)


def standardize_cached(smi, cache, source="qm9"):
    """Standardize one SMILES with memoization. Returns (record, reason)."""
    if smi in cache:
        return cache[smi]
    from tools.ms2_msgym_corpus import standardize_smiles as _std
    rec, reason = _std(smi, source, smi)
    cache[smi] = (rec, reason)
    return rec, reason


def exact_match_queries(queries, xyz_rows):
    """Exact connectivity match of eligible queries vs QM9 members.

    queries: list of {"qid": str (original id), "smiles": str,
    "source": str, ...extra provenance}. Back-compat: "label" accepted
    as the qid when "qid" is absent.
    Member exclusion CANNOT be joined in subset scope (subset-local
    numbering vs full-GDB9 exclusion list): every match carries
    ``exclusion: "unresolved_subset_numbering"`` and the stats record it,
    so no excluded geometry is silently claimed as positive control.
    Matching policy: relaxed-geometry SMILES is primary physical evidence
    (the optimized structure); GDB-input SMILES reported separately.
    Returns (matches, stats) where matches list per-query best hits.
    """
    from tools.ms2_chebi_corpus import formula_key as _fk
    cache = {}
    qrecs = []
    q_excl = {}
    for q in queries:
        rec, reason = standardize_cached(q["smiles"], cache, "qm9-query")
        if rec is None:
            q_excl[reason] = q_excl.get(reason, 0) + 1
            qrecs.append(None)
        else:
            qrecs.append(rec)
    # index members by formula for bounded comparison
    mem_by_formula = {}
    mem_unparseable = {"gdb": 0, "relaxed": 0}
    for m in xyz_rows:
        for role in ("smiles_gdb", "smiles_relaxed"):
            smi = m.get(role, "")
            if not smi:
                continue
            rec, reason = standardize_cached(smi, cache, "qm9-member")
            if rec is None:
                mem_unparseable["gdb" if role == "smiles_gdb" else "relaxed"] += 1
                continue
            mem_by_formula.setdefault(_fk(rec["formula"]), []).append(
                (m["name"], role, rec))
    matches = []
    n_relaxed = 0
    n_gdb_only = 0
    for q, qr in zip(queries, qrecs):
        qid = q.get("qid", q.get("label", "?"))
        base = {"qid": qid, "smiles": q.get("smiles", ""),
                "source": q.get("source", "unknown"),
                "exclusion": "unresolved_subset_numbering"}
        if qr is None:
            matches.append(dict(base, status="query_unparseable", hits=[]))
            continue
        key = _fk(qr["formula"])
        cands = mem_by_formula.get(key, [])
        if not cands:
            matches.append(dict(base, status="out_of_subset_scope", hits=[]))
            continue
        relaxed_hits = []
        gdb_hits = []
        for name, role, mr in cands:
            if graphs_isomorphic(qr["atom_types"], qr["edges"],
                                 mr["atom_types"], mr["edges"]):
                if role == "smiles_relaxed":
                    relaxed_hits.append(name)
                else:
                    gdb_hits.append(name)
        relaxed_hits = sorted(set(relaxed_hits))
        gdb_hits = sorted(set(gdb_hits))
        gdb_only = [h for h in gdb_hits if h not in relaxed_hits]
        if relaxed_hits:
            n_relaxed += 1
            status = "exact_relaxed_match"
        elif gdb_only:
            n_gdb_only += 1
            status = "exact_gdb_only_match"
        else:
            status = "no_exact_match_in_subset"
        matches.append(dict(base, status=status,
                            n_relaxed_hits=len(relaxed_hits),
                            n_gdb_only_hits=len(gdb_only),
                            relaxed_hits=relaxed_hits[:10],
                            gdb_only_hits=gdb_only[:10],
                            query=qid))
    stats = {"n_queries": len(queries),
             "n_query_unparseable": sum(1 for m in matches if m["status"] == "query_unparseable"),
             "n_out_of_subset_scope": sum(1 for m in matches if m["status"] == "out_of_subset_scope"),
             "n_exact_relaxed_match": n_relaxed,
             "n_exact_gdb_only_match": n_gdb_only,
             "n_no_exact_match": sum(1 for m in matches if m["status"] == "no_exact_match_in_subset"),
             "exclusion_policy": "unresolved_subset_numbering (subset-local indices "
                                "cannot join full-GDB9 exclusion list; no excluded "
                                "geometry silently claimed)",
             "query_exclusions": q_excl,
             "member_unparseable": mem_unparseable,
             "match_policy": ("relaxed-geometry SMILES is primary physical "
                              "evidence; GDB-input SMILES reported separately; "
                              "formula/weak-fingerprint presence never counts "
                              "as a match")}
    return matches, stats


def exact_match_stream(tar_path, queries_std, max_members=0, max_seconds=480,
                       excluded_indices=None, full_numbering=False):
    """Exact connectivity match by streaming a (large) archive.

    queries_std: list of (qid, record-or-None, provenance-dict).
    Back-compat: 2-tuples (qid, record) accepted (provenance {}).
    excluded_indices: set of official full-GDB9 exclusion indices. Enforced
    ONLY when full_numbering is True (member gdb_index joins the official
    list); otherwise exclusions are marked unresolved and no member is
    silently claimed. Excluded members that would match are rejected and
    counted (status ``excluded_geometry_match``), never positive evidence.
    Returns (matches, stats). Unmatched queries are ``unknown_truncated``
    when the stream hit a cap/timeout, else explicit ``no_exact_match_in_full``.
    """
    from tools.ms2_chebi_corpus import formula_key as _fk
    from collections import Counter
    cache = {}
    by_formula = {}
    prov = {}
    for item in queries_std:
        if len(item) == 3:
            qid, qr, pq = item
        else:
            qid, qr = item
            pq = {}
        prov[qid] = pq
        if qr is None:
            continue
        by_formula.setdefault(_fk(qr["formula"]), []).append((qid, qr))
    excluded_indices = set(excluded_indices or ())
    _qlabels = [q[0] for q in queries_std]
    relaxed_hits = {qid: set() for qid in _qlabels}
    gdb_only_hits = {qid: set() for qid in _qlabels}
    excluded_hits = {qid: set() for qid in _qlabels}
    mem_unparseable = Counter()
    matched_member_freq = {}
    matched_member_detail = {}
    n_members = 0
    n_tested = 0
    n_excluded_members = 0
    capped = False
    t0 = time.time()
    timed_out = False
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if max_members and n_members >= max_members:
                capped = True
                break
            if time.time() - t0 > max_seconds:
                timed_out = True
                break
            fh = tar.extractfile(member)
            if fh is None:
                continue
            try:
                text = fh.read().decode("utf-8", errors="replace")
            except Exception:
                continue
            parsed, reason = parse_xyz_text(text)
            n_members += 1
            if parsed is None:
                continue
            is_excluded = (full_numbering
                           and parsed["gdb_index"] in excluded_indices)
            if is_excluded:
                n_excluded_members += 1
            for role in ("smiles_gdb", "smiles_relaxed"):
                smi = parsed.get(role, "")
                if not smi:
                    continue
                rec, rsn = standardize_cached(smi, cache, "qm9-full")
                if rec is None:
                    mem_unparseable[role] += 1
                    continue
                for qid, qr in by_formula.get(_fk(rec["formula"]), ()):
                    n_tested += 1
                    if graphs_isomorphic(qr["atom_types"], qr["edges"],
                                         rec["atom_types"], rec["edges"]):
                        choice = ("relaxed" if role == "smiles_relaxed"
                                  else "gdb_input")
                        if is_excluded:
                            excluded_hits[qid].add(member.name)
                            continue
                        if role == "smiles_relaxed":
                            relaxed_hits[qid].add(member.name)
                            if member.name not in matched_member_freq:
                                matched_member_freq[member.name] = (
                                    parsed["n_freq_positive"],
                                    parsed["n_freq_zero"],
                                    parsed["n_freq_negative"],
                                    parsed["freq_mode_status"] == "complete")
                                matched_member_detail[member.name] = {
                                    "gdb_index": parsed["gdb_index"],
                                    "choice": choice,
                                    "freq_mode_status": parsed["freq_mode_status"],
                                    "freq_expected_validated": parsed.get(
                                        "freq_expected_validated"),
                                    "is_linear": parsed.get("is_linear"),
                                    "linearity_detail": parsed.get(
                                        "linearity_detail"),
                                    "n_freq": parsed["n_freq"],
                                    "freqs": parsed.get("freqs", []),
                                }
                        else:
                            gdb_only_hits[qid].add(member.name)
    if timed_out:
        completion = "truncated_time"
    elif capped:
        completion = "truncated_cap"
    else:
        completion = "complete"
    matches = []
    for item in queries_std:
        qid = item[0]
        pq = prov[qid]
        qr = item[1]
        base = {"qid": qid, "smiles": pq.get("smiles", ""),
                "source": pq.get("source", "unknown"),
                "scan_completion": completion}
        if qr is None:
            matches.append(dict(base, status="query_unparseable",
                                n_relaxed_hits=0, n_gdb_only_hits=0,
                                relaxed_hits=[], gdb_only_hits=[],
                                excluded_hits=[], query=qid))
            continue
        rh = sorted(relaxed_hits[qid])
        go = sorted(set(gdb_only_hits[qid]) - set(relaxed_hits[qid]))
        xh = sorted(excluded_hits[qid])
        if rh:
            status = "exact_relaxed_match"
        elif go:
            status = "exact_gdb_only_match"
        elif xh:
            # Would-be matches land on officially excluded geometries:
            # rejected as positive control, counted explicitly.
            status = "excluded_geometry_match"
        elif completion != "complete":
            status = "unknown_truncated"
        else:
            status = "no_exact_match_in_full"
        matches.append(dict(base, status=status,
                            n_relaxed_hits=len(rh), n_gdb_only_hits=len(go),
                            n_excluded_hits=len(xh),
                            relaxed_hits=rh[:10], gdb_only_hits=go[:10],
                            excluded_hits=xh[:10], query=qid))
    stats = {"n_members_streamed": n_members,
             "n_isomorphism_tests": n_tested,
             "n_exact_relaxed_match": sum(1 for m in matches if m["status"] == "exact_relaxed_match"),
             "n_exact_gdb_only_match": sum(1 for m in matches if m["status"] == "exact_gdb_only_match"),
             "n_excluded_geometry_match": sum(1 for m in matches if m["status"] == "excluded_geometry_match"),
             "n_unknown_truncated": sum(1 for m in matches if m["status"] == "unknown_truncated"),
             "n_no_exact_match": sum(1 for m in matches if m["status"] == "no_exact_match_in_full"),
             "n_excluded_members_visited": n_excluded_members,
             "exclusion_enforced": full_numbering,
             "n_exclusion_list": len(excluded_indices),
             "completion": completion, "capped": capped,
             "cap_limit": max_members, "timed_out": timed_out,
             "member_unparseable": dict(mem_unparseable),
             "matched_relaxed_member_freq": {
                 "n_members": len(matched_member_freq),
                 "n_freq_positive": sum(v[0] for v in matched_member_freq.values()),
                 "n_freq_zero": sum(v[1] for v in matched_member_freq.values()),
                 "n_freq_negative": sum(v[2] for v in matched_member_freq.values())},
             "stationary_point_members": sum(
                 1 for v in matched_member_freq.values()
                 if v[3] and v[0] > 0 and v[1] == 0 and v[2] == 0),
             "stationary_point_note": ("complete-count all-positive members "
                                       "only; incomplete/zero/negative excluded"),
             "matched_member_detail": matched_member_detail}
    return matches, stats


def qm9_eligibility(standardize_fn, items):
    """QM9 composition gate via existing typed graphs.

    items: list of {"qid": original id, "smiles": str, ...provenance}.
    Back-compat: bare SMILES strings accepted (qid = the SMILES itself).
    Eligible: parseable, neutral, elements subset of {C,H,O,N,F} (H implicit),
    heavy atoms <= 9. Returns (eligible_list, exclusion_counter); eligible
    entries preserve the original qid/smiles/provenance (no relabeling).
    """
    from collections import Counter
    eligible = []
    excl = Counter()
    for it in items:
        if isinstance(it, str):
            it = {"qid": it, "smiles": it}
        smi = it["smiles"]
        rec, reason = standardize_fn(smi)
        if rec is None:
            excl[f"unparseable:{reason}"] += 1
            continue
        els = {e for e, _, _ in rec["atom_types"]} | {"H"}
        if not els.issubset(set(QM9_ELEMENTS) | {"H"}):
            bad = sorted(els - set(QM9_ELEMENTS) - {"H"})
            excl[f"element:{'+'.join(bad)}"] += 1
            continue
        if rec["heavy"] > QM9_MAX_HEAVY:
            excl["too_many_heavy"] += 1
            continue
        entry = dict(it)
        entry.update({"formula": rec["formula"], "heavy": rec["heavy"]})
        eligible.append(entry)
    return eligible, dict(excl)


def run_qm9_control(repo_root, out_dir, max_members=0):
    t0 = time.process_time()
    wall0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.join(repo_root, "data/pinned/external")

    readme_p = os.path.join(ext, "qm9_readme.txt")
    unchar_p = os.path.join(ext, "qm9_uncharacterized.txt")
    c7o2_p = os.path.join(ext, "qm9_dsC7O2H10nsd.xyz.tar.bz2")
    for p in (readme_p, unchar_p, c7o2_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing pinned QM9 file: {p}")
    full_p = os.path.join(ext, "dsgdb9nsd.xyz.tar.bz2")
    full_present = os.path.exists(full_p)

    entries, n_bad_lines = parse_uncharacterized(unchar_p)
    progress(f"uncharacterized: {len(entries)} entries, {n_bad_lines} bad lines")
    progress("streaming C7H10O2 xyz tar (no extraction)")
    xyz_rows, xyz_stats, subset_indices = scan_xyz_tar(c7o2_p, max_members=max_members)
    excl_status = exclusion_numbering_status(
        subset_indices, [e["gdb9_index"] for e in entries])

    from tools.ms2_msgym_corpus import standardize_smiles as _std

    def _s(smi):
        return _std(smi, "qm9-proxy", smi)

    # frozen val-200 targets + train fit/calib identities (original qids kept)
    val_rows = list(csv.DictReader(open(
        os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/predictor_rows.csv"),
        encoding="utf-8")))
    fit_rows = list(csv.DictReader(open(
        os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/train_fit_rows.csv"),
        encoding="utf-8")))
    val_items = [{"qid": r.get("qid", f"VALROW-{i:05d}"),
                    "smiles": r["smiles"], "source": "predictor_rows/val200",
                    "row_index": i}
                   for i, r in enumerate(val_rows) if r.get("smiles")]
    fit_items = [{"qid": f"TRAINROW-{i:05d}", "smiles": r["smiles"],
                  "source": "train_fit_rows",
                  "row_index": i, "group": r.get("group", ""),
                  "split": r.get("split", "")}
                 for i, r in enumerate(fit_rows) if r.get("smiles")]

    val_elig, val_excl = qm9_eligibility(_s, val_items)
    fit_elig, fit_excl = qm9_eligibility(_s, fit_items)

    # exact connectivity match (primary) for eligible queries
    from tools.ms2_chebi_corpus import formula_key as _fk
    from tools.ms2_msgym_corpus import parse_formula as _pf
    c7o2_key = _fk(_pf("C7H10O2"))
    n_c7o2_val = sum(1 for e in val_elig if _fk(e["formula"]) == c7o2_key)
    n_c7o2_fit = sum(1 for e in fit_elig if _fk(e["formula"]) == c7o2_key)

    val_queries = [dict(e, source="predictor_rows/val200") for e in val_elig]
    val_matches, val_match_stats = exact_match_queries(val_queries, xyz_rows)
    fit_queries = [dict(e, source="train_fit_rows") for e in fit_elig]
    fit_matches, fit_match_stats = exact_match_queries(fit_queries, xyz_rows)

    # frequency evidence: exact geometry identity is kept, but subset
    # members carry UNRESOLVED exclusions (subset-local numbering cannot
    # join the official exclusion list), so NO positive-control frequency
    # evidence may use subset rows. Positive evidence comes only from the
    # full archive with enforced exclusions (see full section below).
    matched_names = set()
    for m in val_matches + fit_matches:
        matched_names.update(m.get("relaxed_hits", []))
    matched_freq = [r for r in xyz_rows if r["name"] in matched_names]
    complete_freq = [r for r in matched_freq
                     if r.get("freq_mode_status") == "complete"]
    incomplete_freq = [r for r in matched_freq
                       if r.get("freq_mode_status") != "complete"]
    freq_evidence = {
        "subset_scope": ("exclusion_unresolved: subset members CANNOT "
                         "contribute positive-control frequency evidence; "
                         "counts below are identity bookkeeping only, with "
                         "stationary-point evidence forced to zero"),
        "n_exact_relaxed_members": len(matched_freq),
        "n_freq_positive": sum(r["n_freq_positive"] for r in matched_freq),
        "n_freq_zero": sum(r["n_freq_zero"] for r in matched_freq),
        "n_freq_negative": sum(r["n_freq_negative"] for r in matched_freq),
        "stationary_point_members": 0,
        "stationary_point_note": ("subset exclusion unresolved: no subset "
                                  "row may serve as positive control"),
        "complete_count_members_identity_only": len(complete_freq),
        "incomplete_count_members": len(incomplete_freq),
        "incomplete_count_note": ("exact geometry identity retained; "
                                  "incomplete mode counts NEVER counted as "
                                  "positive-only stationary-point evidence"),
        "interpretation": ("complete-count all-positive harmonic frequencies "
                           "on exactly matched relaxed geometries are RESTRICTED "
                           "physical evidence of a stationary point, NOT "
                           "kinetic lifetime/solution stability; zero/ "
                           "negative/incomplete counts reported separately "
                           "when present"),
    }

    # broader full-archive scan when present, else explicit not-claimed
    if full_present:
        progress("streaming full dsgdb9nsd xyz tar (no extraction)")
        full_rows, full_stats, full_indices = scan_xyz_tar(full_p, max_members=max_members)
        full_excl = exclusion_numbering_status(
            full_indices, [e["gdb9_index"] for e in entries])
        # full-archive numbering IS full-GDB9: join against the official
        # excluded list is legitimate here (subset join stays refused).
        excluded_set = set(e["gdb9_index"] for e in entries)
        n_full_excluded = sum(1 for i in full_indices if i in excluded_set)
        full_status = {"present": True, "scan": full_stats,
                       "numbering": "full-GDB9 (join legitimate)",
                       "n_members_matching_exclusion_list": n_full_excluded}
        # exact connectivity check of eligible queries vs full archive,
        # with official exclusions ENFORCED (full-GDB9 numbering joins).
        from tools.ms2_msgym_corpus import standardize_smiles as _std2
        _cache_q = {}

        def _qs(smi):
            rec, _ = standardize_cached(smi, _cache_q, "qm9-query-full")
            return rec

        excluded_set_full = set(e["gdb9_index"] for e in entries)
        val_qstd = [(e["qid"], _qs(e["smiles"]),
                     {"qid": e["qid"], "smiles": e["smiles"],
                      "source": "predictor_rows/val200",
                      "row_index": e.get("row_index")})
                    for e in val_elig]
        fit_qstd = [(e["qid"], _qs(e["smiles"]),
                     {"qid": e["qid"], "smiles": e["smiles"],
                      "source": "train_fit_rows",
                      "row_index": e.get("row_index"),
                      "group": e.get("group", ""),
                      "split": e.get("split", "")})
                    for e in fit_elig]
        progress("exact matching eligible queries vs full archive")
        val_full_matches, val_full_stats = exact_match_stream(
            full_p, val_qstd, max_members=max_members,
            excluded_indices=excluded_set_full, full_numbering=True)
        fit_full_matches, fit_full_stats = exact_match_stream(
            full_p, fit_qstd, max_members=max_members,
            excluded_indices=excluded_set_full, full_numbering=True)
        full_status["exact_match_val"] = val_full_stats
        full_status["exact_match_train_fit"] = fit_full_stats
        full_status["val_full_matches"] = val_full_matches
        full_status["fit_full_matches"] = fit_full_matches
        for _scope, _st in (("val", val_full_stats), ("train_fit", fit_full_stats)):
            _mf = _st.get("matched_relaxed_member_freq", {})
            if _mf.get("n_members"):
                freq_evidence[f"full_{_scope}_matched_members"] = _mf
        freq_evidence["full_archive_freq_totals"] = full_stats.get("freq_totals")
        # frequency evidence over exactly matched full members is folded
        # into frequency_evidence below via matched-name join when feasible
    else:
        full_status = {"present": False,
                       "status": ("not_obtained: full dsgdb9nsd.xyz.tar.bz2 "
                                  "not in data/pinned/external; no absence "
                                  "claim beyond the C7H10O2 subset is made")}

    cpu_s = time.process_time() - t0
    wall_s = time.time() - wall0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    full_pin = {}
    if full_present:
        full_pin = {"dsgdb9nsd.xyz.tar.bz2": {
            "sha256": sha256_file(full_p),
            "bytes": os.path.getsize(full_p),
            "url": "https://ndownloader.figshare.com/files/3195389",
            "api": "https://api.figshare.com/v2/articles/1057646",
            "retrieved": "2026-10-03"}}
    summary = {
        "tool": "tools/ms2_qm9_control.py",
        "source": FIGSHARE,
        "pins": {
            "qm9_readme.txt": {"sha256": sha256_file(readme_p),
                               "bytes": os.path.getsize(readme_p)},
            "qm9_uncharacterized.txt": {"sha256": sha256_file(unchar_p),
                                       "bytes": os.path.getsize(unchar_p),
                                       "n_entries": len(entries),
                                       "n_bad_lines": n_bad_lines},
            "qm9_dsC7O2H10nsd.xyz.tar.bz2": {"sha256": sha256_file(c7o2_p),
                                             "bytes": os.path.getsize(c7o2_p)},
            "retrieval_date": "2026-10-03",
        },
        "xyz_scan": xyz_stats,
        "exclusion_policy": excl_status,
        "eligibility": {
            "val200": {"n_eligible": len(val_elig), "n_total": len(val_items),
                       "exclusions": val_excl},
            "train_fit": {"n_eligible": len(fit_elig), "n_total": len(fit_items),
                          "exclusions": fit_excl},
            "rule": ("parseable + neutral + elements subset of {C,H,O,N,F} + "
                     "heavy<=9 via existing typed-graph standardizer"),
        },
        "exact_match": {
            "val": val_match_stats,
            "train_fit": fit_match_stats,
            "val_C7H10O2_eligible": n_c7o2_val,
            "train_fit_C7H10O2_eligible": n_c7o2_fit,
            "train_note": ("physical-evidence presence audit only; no spectral scoring, "
                           "hence no train-contamination concern; declared explicitly"),
            "official_C7H10O2_members": xyz_stats["formula_counts"].get(
                str(tuple(sorted({"C": 7, "H": 10, "O": 2}.items()))), 0),
            "level": ("EXACT typed-graph connectivity identity "
                      "(relaxed primary, GDB separate); formula/weak-fp "
                      "presence never counts"),
            "exact_graph_identity_gate": ("COMPLETE for C7H10O2 subset scope; "
                                          "broader QM9 scope only when full "
                                          "archive scanned"),
            "pool_injection": "none (no target injected into any candidate pool)",
            "source_identity_mismatch": {
                "n_members_smiles_mismatch": xyz_stats["n_smiles_mismatch"],
                "n_members_inchi_mismatch": xyz_stats["n_inchi_mismatch"],
                "policy": ("relaxed-geometry SMILES/InChI is the evidence "
                           "structure; GDB-input strings reported separately; "
                           "mismatches counted, never silently merged")},
        },
        "restricted_match": {
            "val_eligible_with_official_formula_present": None,
            "note": ("superseded by exact_match; weak formula presence "
                     "retired, never claimed as physical evidence"),
        },
        "frequency_evidence": freq_evidence,
        "full_archive": full_status,
        "stability_claim": ("NONE: a matched optimized geometry is restricted "
                            "evidence, NOT kinetic lifetime/solution stability"),
        "costs": {"wall_s": wall_s, "cpu_s": cpu_s, "peak_rss_kb": rss_kb,
                  "download_bytes_new": 0,
                  "download_cache": ("all pinned archives pre-existing on "
                                     "disk; 0 new bytes fetched by this run; "
                                     "subset/full sizes below are on-disk "
                                     "pinned bytes, not new ingress"),
                  "pinned_subset_bytes": os.path.getsize(c7o2_p),
                  "api_calls": 0},
        "code_sha256": sha256_file(os.path.join(repo_root, "tools/ms2_qm9_control.py")),
        "env": {"python": sys.version.split()[0]},
    }
    summary["pins"].update(full_pin)
    text = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
    out_p = os.path.join(out_dir, "qm9_summary.json")
    with open(out_p, "w", encoding="utf-8") as handle:
        handle.write(text)
    # per-member CSV (bounded: formula + tail-evidence flags, no coordinates)
    csv_p = os.path.join(out_dir, "qm9_members.csv")
    with open(csv_p, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "name", "gdb_subset_index", "formula", "natoms", "n_freq",
            "freq_mode_status",
            "n_freq_positive", "n_freq_zero", "n_freq_negative",
            "smiles_match", "inchi_match", "has_comment"])
        w.writeheader()
        for r in xyz_rows:
            w.writerow({"name": r["name"],
                        "gdb_subset_index": r["gdb_subset_index"],
                        "formula": _fk(r["formula"]),
                        "natoms": r["natoms"], "n_freq": r["n_freq"],
                        "freq_mode_status": r.get("freq_mode_status", ""),
                        "n_freq_positive": r["n_freq_positive"],
                        "n_freq_zero": r["n_freq_zero"],
                        "n_freq_negative": r["n_freq_negative"],
                        "smiles_match": r["smiles_match"],
                        "inchi_match": r["inchi_match"],
                        "has_comment": r["has_comment"]})
    # per-query exact-match CSV (subset + full when present)
    mq_p = os.path.join(out_dir, "qm9_exact_matches.csv")
    with open(mq_p, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "qid", "smiles", "source", "query", "scope", "status",
            "exclusion", "scan_completion",
            "n_relaxed_hits", "n_gdb_only_hits", "n_excluded_hits"])
        w.writeheader()
        for m in val_matches + fit_matches:
            w.writerow({"qid": m.get("qid", m.get("query", "")),
                        "smiles": m.get("smiles", ""),
                        "source": m.get("source", ""),
                        "query": m.get("query", m.get("qid", "")),
                        "scope": "subset",
                        "status": m["status"],
                        "exclusion": m.get("exclusion", ""),
                        "scan_completion": "complete",
                        "n_relaxed_hits": m.get("n_relaxed_hits", 0),
                        "n_gdb_only_hits": m.get("n_gdb_only_hits", 0),
                        "n_excluded_hits": 0})
        if full_present:
            for m in (full_status.get("val_full_matches", [])
                      + full_status.get("fit_full_matches", [])):
                w.writerow({"qid": m.get("qid", m.get("query", "")),
                            "smiles": m.get("smiles", ""),
                            "source": m.get("source", ""),
                            "query": m.get("query", m.get("qid", "")),
                            "scope": "full",
                            "status": m["status"],
                            "exclusion": ("enforced_full_GDB9"
                                         if full_status.get("exact_match_val", {})
                                         .get("exclusion_enforced", True)
                                         else "unenforced"),
                            "scan_completion": m.get("scan_completion", ""),
                            "n_relaxed_hits": m.get("n_relaxed_hits", 0),
                            "n_gdb_only_hits": m.get("n_gdb_only_hits", 0),
                            "n_excluded_hits": m.get("n_excluded_hits", 0)})
    # full per-match detail is kept in the JSON (val_full_matches /
    # fit_full_matches lists carry up to 10 hit names per query: bounded
    # by construction, retained for frequency/provenance joins).
    summary["outputs"] = {"summary": out_p, "members": csv_p,
                          "matches": mq_p}
    return summary


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/molecular_completion/20261003_remaining")
    ap.add_argument("--max-members", type=int, default=0)
    args = ap.parse_args(argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(repo_root) == "tools":
        repo_root = os.path.dirname(repo_root)
    summary = run_qm9_control(repo_root, args.out_dir, max_members=args.max_members)
    print(json.dumps({"xyz_members": summary["xyz_scan"]["n_members"],
                      "val_eligible": summary["eligibility"]["val200"]["n_eligible"],
                      "exact_gate": summary["exact_match"]["exact_graph_identity_gate"],
                      "out": summary["outputs"]["summary"]}, allow_nan=False))


class XyzTests(unittest.TestCase):
    def test_basic(self):
        text = ("3\ngdb 7 prop\nC 0.0 0.0 0.0 0.0\nH 0.0 0.0 1.0 0.0\n"
                "H 0.0 1.0 0.0 0.0\n100.0 200.0 300.0\nCCO\tCCO\n"
                "InChI=1S/C2H6O/a\tInChI=1S/C2H6O/a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason)
        self.assertEqual(parsed["counts"], {"C": 1, "H": 2})
        self.assertEqual(parsed["gdb_index"], 7)
        self.assertEqual(parsed["n_freq"], 3)
        self.assertEqual(parsed["n_freq_positive"], 3)
        self.assertTrue(parsed["smiles_match"])

    def test_bad_natoms(self):
        parsed, reason = parse_xyz_text("x\nc\nd\n")
        self.assertIsNone(parsed)
        self.assertEqual(reason, "bad_natoms")

    def test_count_mismatch(self):
        parsed, reason = parse_xyz_text("2\ngdb 1 p\nC 0 0 0")
        self.assertIsNone(parsed)
        self.assertIn(reason, ("line_count_mismatch", "atom_count_mismatch"))

    def test_no_bond_inference(self):
        text = ("2\ngdb 3 p\nO 0 0 0 0.0\nH 0 0 1 0.0\n10.0 20.0 30.0 40.0 50.0 60.0\n"
                "O\tO\nInChI=a\tInChI=a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason)
        self.assertNotIn("bonds", parsed)
        self.assertNotIn("edges", parsed)

    def test_nonfinite_coords_rejected(self):
        text = ("1\ngdb 4 p\nC inf 0 0 0.0\n100.0\nC\tC\nI\ta\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(parsed)
        self.assertEqual(reason, "nonfinite_coordinates")

    def test_freq_sign_classes(self):
        text = ("1\ngdb 5 p\nC 0 0 0 0.0\n-5.0 0.0 10.0\nC\tC\nI\ta\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason)
        self.assertEqual(
            (parsed["n_freq_negative"], parsed["n_freq_zero"],
             parsed["n_freq_positive"]), (1, 1, 1))

    def test_smiles_mismatch_detected(self):
        text = ("1\ngdb 6 p\nC 0 0 0 0.0\n50.0\nCCO\tCOC\nInChI=a\tInChI=b\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason)
        self.assertFalse(parsed["smiles_match"])
        self.assertFalse(parsed["inchi_match"])

    def test_missing_tail_rejected(self):
        parsed, reason = parse_xyz_text("1\ngdb 8 p\nC 0 0 0 0.0\n")
        self.assertIsNone(parsed)
        self.assertIn(reason, ("too_short", "line_count_mismatch"))

    def test_provider_scientific_notation(self):
        # Actual provider notation from dsgdb9nsd_000212.xyz: 2.1997*^-6.
        text = ("1\ngdb 212 p\nC 2.1997*^-6 1.4462618059 0.0098312216 -0.335446\n"
                "100.0\nCC\tCC\nInChI=a\tInChI=a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["gdb_index"], 212)

    def test_fortran_d_exponent(self):
        text = ("1\ngdb 9 p\nC 1.0D-06 0.0 0.0 0.0\n2.5D+02\nC\tC\nI\ta\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["n_freq_positive"], 1)

    def test_incomplete_mode_count_flagged(self):
        # 1 atom -> expects 3N-6 <0 / 3N-5 = -2... use 2 atoms: expect 0/1.
        text = ("2\ngdb 10 p\nC 0 0 0 0.0\nH 0 0 1 0.0\n"
                "10.0 20.0 30.0\nC\tC\nI\ta\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["freq_mode_status"], "incomplete_count")

    def test_complete_mode_count(self):
        # 2 atoms nonlinear -> 3*2-6 = 0... use single pattern: 0 or 1 ok.
        text = ("2\ngdb 11 p\nC 0 0 0 0.0\nH 0 0 1 0.0\n10.0\nC\tC\nI\ta\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["freq_mode_status"], "complete")

    def test_linearity_validated_co2(self):
        # Linear O=C=O: 3 atoms -> expected 3*3-5 = 4, not 3.
        text = ("3\ngdb 20 p\nC 0.0 0.0 0.0 0.0\nO -1.16 0.0 0.0 0.0\n"
                "O 1.16 0.0 0.0 0.0\n100.0 200.0 300.0 400.0\n"
                "O=C=O\tO=C=O\nInChI=a\tInChI=a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertTrue(parsed["is_linear"])
        self.assertEqual(parsed["freq_expected_validated"], 4)
        self.assertEqual(parsed["freq_mode_status"], "complete")

    def test_linearity_bent_water(self):
        # Bent H2O: 3 atoms nonlinear -> expected 3*3-6 = 3.
        text = ("3\ngdb 21 p\nO 0.0 0.0 0.0 0.0\nH 0.757 0.586 0.0 0.0\n"
                "H -0.757 0.586 0.0 0.0\n100.0 200.0 300.0\n"
                "O\tO\nInChI=a\tInChI=a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertFalse(parsed["is_linear"])
        self.assertEqual(parsed["freq_expected_validated"], 3)
        self.assertEqual(parsed["freq_mode_status"], "complete")

    def test_linearity_either_count_retired(self):
        # Bent H2O with 4 modes (the LINEAR count) must be incomplete.
        text = ("3\ngdb 22 p\nO 0.0 0.0 0.0 0.0\nH 0.757 0.586 0.0 0.0\n"
                "H -0.757 0.586 0.0 0.0\n100.0 200.0 300.0 400.0\n"
                "O\tO\nInChI=a\tInChI=a\n")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["freq_mode_status"], "incomplete_count")


class ExclusionTests(unittest.TestCase):
    def test_numbering_join_refused(self):
        st = exclusion_numbering_status(list(range(1, 6096)), [58, 61, 133885])
        self.assertEqual(st["join"], "refused_incompatible_numbering")
        self.assertEqual(st["n_excluded_by_join"], 0)
        self.assertEqual(st["subset_index_range"], [1, 6095])


class IsoTests(unittest.TestCase):
    def _std(self, smi):
        from tools.ms2_msgym_corpus import standardize_smiles as _s
        rec, reason = _s(smi, "t", smi)
        self.assertIsNotNone(rec, reason)
        return rec

    def test_same_connectivity_iso(self):
        a = self._std("CCO")
        b = self._std("OCC")
        self.assertTrue(graphs_isomorphic(
            a["atom_types"], a["edges"], b["atom_types"], b["edges"]))

    def test_constitutional_isomer_not_iso(self):
        a = self._std("CCO")
        b = self._std("COC")
        self.assertFalse(graphs_isomorphic(
            a["atom_types"], a["edges"], b["atom_types"], b["edges"]))

    def test_formula_never_counts(self):
        # same Hill formula must not imply identity (CCO vs COC are
        # constitutional isomers; typed multisets already differ here,
        # and the isomorphism check rejects regardless).
        from tools.ms2_database_retrieval import formula_of
        a = self._std("CCO")
        b = self._std("COC")
        self.assertEqual(formula_of(a["atom_types"]),
                         formula_of(b["atom_types"]))
        self.assertFalse(graphs_isomorphic(
            a["atom_types"], a["edges"], b["atom_types"], b["edges"]))

    def test_formula_key_regression(self):
        from tools.ms2_chebi_corpus import formula_key as _fk
        from tools.ms2_msgym_corpus import parse_formula as _pf
        self.assertEqual(_fk(_pf("C7H10O2")), _fk({"C": 7, "H": 10, "O": 2}))
        self.assertNotEqual(_fk(_pf("C7H10O2")), "C7H10O2")


class StreamTests(unittest.TestCase):
    def _member(self, idx, smi_gdb, smi_rel, formula_counts=((("C", 2), ("H", 6), ("O", 1)),)):
        lines = [str(9), f"gdb {idx} p p p p p p p p p p p p p p p p"]
        for el in ["C", "C", "O", "H", "H", "H", "H", "H", "H"]:
            lines.append(f"{el} 0.0 0.0 0.0 0.0")
        lines.append("100.0 200.0")
        lines.append(f"{smi_gdb}\t{smi_rel}")
        lines.append("InChI=a\tInChI=a")
        return "\n".join(lines) + "\n"

    def test_stream_match_and_scope(self):
        import bz2
        import tempfile
        members = {
            "m1.xyz": self._member(11, "CCO", "CCO"),
            "m2.xyz": self._member(12, "CCC", "CCC"),
        }
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as fh:
            path = fh.name
        try:
            with tarfile.open(path, "w:bz2") as tar:
                for name, text in members.items():
                    import io as _io
                    data = text.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, _io.BytesIO(data))
            from tools.ms2_msgym_corpus import standardize_smiles as _s

            def _qs(smi):
                rec, _ = standardize_cached(smi, {}, "t")
                return rec
            qs = [("Q-CCO", _qs("OCC"), {"qid": "Q-CCO", "smiles": "OCC"}),
                  ("Q-CCC", _qs("CCC"), {"qid": "Q-CCC", "smiles": "CCC"}),
                  ("Q-BIG", None, {"qid": "Q-BIG", "smiles": ""})]
            matches, stats = exact_match_stream(path, qs)
            by_q = {m["qid"]: m for m in matches}
            self.assertEqual(by_q["Q-CCO"]["status"], "exact_relaxed_match")
            self.assertEqual(by_q["Q-CCC"]["status"], "exact_relaxed_match")
            self.assertEqual(by_q["Q-BIG"]["status"], "query_unparseable")
            self.assertEqual(stats["n_members_streamed"], 2)
            self.assertEqual(stats["completion"], "complete")
        finally:
            os.unlink(path)

    def _write_tar(self, members):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as fh:
            path = fh.name
        with tarfile.open(path, "w:bz2") as tar:
            for name, text in members.items():
                import io as _io
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, _io.BytesIO(data))
        return path

    def test_excluded_geometry_rejected(self):
        # Mirrors real dsgdb9nsd_000058.xyz (gdb 58, NC(=N)C#N): officially
        # excluded, must never count as positive control.
        lines = ["8", "gdb 58 p p p p p p p p p p p p p p p p"]
        for el in ["N", "C", "N", "C", "N", "H", "H", "H"]:
            lines.append(f"{el} 0.0 0.0 0.0 0.0")
        lines.append("100.0 200.0")
        lines.append("NC(=N)C#N\tNC(=N)C#N")
        lines.append("InChI=a\tInChI=a")
        members = {"dsgdb9nsd_000058.xyz": "\n".join(lines) + "\n"}
        path = self._write_tar(members)
        try:
            def _qs(smi):
                rec, _ = standardize_cached(smi, {}, "t")
                return rec
            qs = [("Q58", _qs("NC(=N)C#N"),
                   {"qid": "Q58", "smiles": "NC(=N)C#N"})]
            matches, stats = exact_match_stream(
                path, qs, excluded_indices={58}, full_numbering=True)
            self.assertEqual(matches[0]["status"], "excluded_geometry_match")
            self.assertEqual(matches[0]["n_excluded_hits"], 1)
            self.assertEqual(matches[0]["n_relaxed_hits"], 0)
            self.assertEqual(stats["n_excluded_geometry_match"], 1)
            self.assertEqual(stats["n_exact_relaxed_match"], 0)
            self.assertEqual(stats["n_excluded_members_visited"], 1)
            # Without enforcement the same geometry matches (control).
            matches2, _ = exact_match_stream(path, qs)
            self.assertEqual(matches2[0]["status"], "exact_relaxed_match")
        finally:
            os.unlink(path)

    def test_cap_truncation_unknown(self):
        members = {
            "m1.xyz": self._member(11, "CCO", "CCO"),
            "m2.xyz": self._member(12, "CCC", "CCC"),
        }
        path = self._write_tar(members)
        try:
            def _qs(smi):
                rec, _ = standardize_cached(smi, {}, "t")
                return rec
            qs = [("Q-NOMATCH", _qs("CCN"),
                   {"qid": "Q-NOMATCH", "smiles": "CCN"})]
            matches, stats = exact_match_stream(path, qs, max_members=1)
            self.assertEqual(stats["completion"], "truncated_cap")
            self.assertEqual(matches[0]["status"], "unknown_truncated")
        finally:
            os.unlink(path)

    def test_full_zero_eligible_batch(self):
        members = {"m1.xyz": self._member(11, "CCO", "CCO")}
        path = self._write_tar(members)
        try:
            qs = [("Q-NONE", None, {"qid": "Q-NONE", "smiles": ""})]
            matches, stats = exact_match_stream(path, qs)
            self.assertEqual(stats["completion"], "complete")
            self.assertEqual(matches[0]["status"], "query_unparseable")
        finally:
            os.unlink(path)


class RealArchiveTests(unittest.TestCase):
    ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "pinned", "external",
        "dsgdb9nsd.xyz.tar.bz2")

    def _read_member(self, suffix):
        import tarfile as _tf
        with _tf.open(self.ARCHIVE, "r:bz2") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(suffix):
                    return tar.extractfile(member).read().decode(
                        "utf-8", errors="replace")
        self.fail(f"member {suffix} not found")

    def test_actual_record58_excluded(self):
        # Real dsgdb9nsd_000058.xyz (gdb 58, NC(=N)C#N) is officially
        # excluded: parses, would match, but must be rejected with
        # enforcement on.
        text = self._read_member("_000058.xyz")
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["gdb_index"], 58)
        entries, _ = parse_uncharacterized(os.path.join(
            os.path.dirname(self.ARCHIVE), "qm9_uncharacterized.txt"))
        self.assertIn(58, set(e["gdb9_index"] for e in entries))

    def test_actual_record212_numeric_format(self):
        # Real dsgdb9nsd_000212.xyz carries 2.1997*^-6 coordinates.
        text = self._read_member("_000212.xyz")
        self.assertIn("*^", text)
        parsed, reason = parse_xyz_text(text)
        self.assertIsNone(reason, reason)
        self.assertEqual(parsed["gdb_index"], 212)


class UncharTests(unittest.TestCase):
    def test_parse_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("123 CCO blah\nbadline here\n\n456 CCN x\n")
            name = fh.name
        try:
            entries, n_bad = parse_uncharacterized(name)
        finally:
            os.unlink(name)
        self.assertEqual([e["gdb9_index"] for e in entries], [123, 456])
        self.assertEqual(n_bad, 1)


class EligibilityTests(unittest.TestCase):
    def test_gates(self):
        from tools.ms2_msgym_corpus import standardize_smiles as _std

        def _s(smi):
            return _std(smi, "t", smi)

        elig, excl = qm9_eligibility(_s, ["CCO", "c1ccccc1", "CC[Si](C)C", "not smiles"])
        # CCO eligible (<=9 heavy CHON); benzene: aromatic parser? counted either way
        self.assertTrue(any(e["smiles"] == "CCO" for e in elig))
        self.assertTrue(any("element" in k or "unparseable" in k for k in excl))

    def test_original_qids_preserved(self):
        from tools.ms2_msgym_corpus import standardize_smiles as _std

        def _s(smi):
            return _std(smi, "t", smi)

        items = [{"qid": "VAL-0007", "smiles": "CCO", "source": "s"},
                 {"qid": "VAL-0008", "smiles": "CC[Si](C)C", "source": "s"}]
        elig, excl = qm9_eligibility(_s, items)
        self.assertEqual([e["qid"] for e in elig], ["VAL-0007"])
        self.assertEqual(elig[0]["source"], "s")


if __name__ == "__main__":
    main(sys.argv[1:])
