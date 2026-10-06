"""Real-corpus ingestion for the database-first retrieval harness.

Extends tools/ms2_database_retrieval.py (imported, not modified) with:

- a restricted, stdlib-only SMILES parser that accepts a documented subset
  and rejects everything else with an explicit reason (aromaticity,
  charges, non-domain elements, bracket anomalies, syntax errors). Rejected
  inputs are a measured denominator, never silently reinterpreted;
- a Hill-formula parser used to cross-check the graph-derived composition;
- an incremental scanner for the large MassSpecGym candidate JSONs so a
  capped prefix download yields complete per-query entries;
- a pinned, capped MassSpecGym val-fold driver that joins supplied
  candidate pools to measured-spectrum metadata and runs the same four
  staged filters as the synthetic harness.

Run:   python3 tools/ms2_msgym_corpus.py --prefix data/pinned/msgym_candidates_formula_prefix.json \\
           --tsv data/pinned/MassSpecGym1.5.tsv --max-queries 150
Test:  python3 -m unittest tools.ms2_msgym_corpus

Scope notes (see docs/MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md):
- Target-parent reconstruction only. Peak masses do not identify fragment
  connectivity, so no fragment-graph claims are made.
- Whole-target requests: the precursor measurement is redundant with the
  target evidence, so the stage-4 precursor arm is reported not_evaluated.
- Aromatic records are outside the current closed-molecule typed-graph
  domain (no kekulization is performed; silent kekulization would invent
  bond orders). They are counted as exclusions.
- Top-k is reported only when the target is present in the supplied pool.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_database_retrieval import (
    DOMAIN_ELEMENTS,
    DatabaseIndex,
    WorkCounters,
    edge_fingerprint,
    formula_key,
    formula_of,
    joint_contains,
    mass_verdict,
    neutral_mass,
    rank_baselines,
    run_query,
    standardize_record,
    contains_exact,
)

VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1, "S": 2}
AROMATIC_VALENCE = {"C": 3, "N": 3, "O": 2, "S": 2}
DOMAIN_SET = {"C", "N", "O", "F", "Cl", "Br", "I", "S"}
AROMATIC_ATOMS = set("cnops")
ORGANIC_ATOMS = {
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
}


class SmilesUnsupported(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _read_atom(smi, i):
    """Parse one atom token at position i. Returns (atom_dict, next_i)."""
    n = len(smi)
    c = smi[i]
    if c == "[":
        j = smi.index("]", i)
        body = smi[i + 1:j]
        # Strip stereochemistry markers (connectivity identity only).
        body = body.replace("@", "")
        k = 0
        while k < len(body) and not body[k].isalpha():
            k += 1
        # A leading number is an isotope label ([13C], [2H]): normalize-and-
        # forget would silently compute the unlabeled mass, so reject.
        if any(ch.isdigit() for ch in body[:k]):
            raise SmilesUnsupported("unsupported_isotope")
        aromatic = False
        if k < len(body) and body[k].islower():
            aromatic = True
            elem = body[k].upper()
            k += 1
        elif k < len(body) and body[k].isupper():
            elem = body[k]
            k += 1
            if k < len(body) and body[k].islower():
                elem += body[k]
                k += 1
        else:
            raise SmilesUnsupported(f"bracket_feature:{body!r}")
        rest = body[k:]
        hydrogens = 0
        charge = 0
        p = 0
        while p < len(rest):
            if rest[p] == "H":
                p += 1
                num = ""
                while p < len(rest) and rest[p].isdigit():
                    num += rest[p]
                    p += 1
                hydrogens = int(num) if num else 1
            elif rest[p] in "+-":
                sign = 1 if rest[p] == "+" else -1
                p += 1
                num = ""
                while p < len(rest) and rest[p].isdigit():
                    num += rest[p]
                    p += 1
                charge += sign * (int(num) if num else 1)
            else:
                raise SmilesUnsupported(f"bracket_feature:{rest!r}")
        return ({"element": elem, "aromatic": aromatic, "bracket": True,
                 "hydrogens": hydrogens, "charge": charge,
                 "valence": AROMATIC_VALENCE.get(elem, VALENCE.get(elem, 0)) if aromatic
                 else VALENCE.get(elem, 0)}, j + 1)
    if c.islower():
        return ({"element": c.upper(), "aromatic": True, "bracket": False,
                 "hydrogens": 0, "charge": 0,
                 "valence": AROMATIC_VALENCE.get(c.upper(), 0)}, i + 1)
    if c.isupper():
        if c in ("Cl",) or (c == "C" and i + 1 < n and smi[i + 1] == "l"):
            return ({"element": "Cl", "aromatic": False, "bracket": False,
                     "hydrogens": 0, "charge": 0, "valence": 1}, i + 2)
        if c == "B" and i + 1 < n and smi[i + 1] == "r":
            return ({"element": "Br", "aromatic": False, "bracket": False,
                     "hydrogens": 0, "charge": 0, "valence": 1}, i + 2)
        if c == "I":
            return ({"element": "I", "aromatic": False, "bracket": False,
                     "hydrogens": 0, "charge": 0, "valence": 1}, i + 1)
        if c == "S":
            return ({"element": "S", "aromatic": False, "bracket": False,
                     "hydrogens": 0, "charge": 0, "valence": 2}, i + 1)
        return ({"element": c, "aromatic": False, "bracket": False,
                 "hydrogens": 0, "charge": 0,
                 "valence": VALENCE.get(c, 0)}, i + 1)
    raise SmilesUnsupported(f"unexpected_char:{c!r}")


def parse_smiles(smi):
    """Parse SMILES into (atoms, bonds).

    atoms: list of dicts with element/aromatic/bracket/hydrogens/charge/
    valence. bonds: list of (a, b, order) with order in (1, 2, 3, 4 =
    aromatic). Raises SmilesUnsupported for anything outside the subset.
    """
    from tools.ms2_database_retrieval import AROM_ORDER
    if not smi:
        raise SmilesUnsupported("empty_smiles")
    if "." in smi:
        raise SmilesUnsupported("disconnected_or_salt")
    atoms, bonds = [], []
    stack = []
    ring = {}
    prev = None
    pending_bond = None
    i, n = 0, len(smi)

    def add_bond(a, b, order):
        if a is None or b is None:
            raise SmilesUnsupported("dangling_bond")
        bonds.append((a, b, order))

    def close_ring(num, order):
        # Default ring-bond order resolves at close time, when both endpoint
        # aromaticity flags are known (fixes aromatic closures).
        if num in ring:
            other, stored = ring.pop(num)
            if order is not None:
                resolved = order
            elif stored is not None:
                resolved = stored
            elif atoms[prev]["aromatic"] and atoms[other]["aromatic"]:
                resolved = AROM_ORDER
            else:
                resolved = 1
            add_bond(prev, other, resolved)
        else:
            ring[num] = (prev, order)

    while i < n:
        c = smi[i]
        if c == "(":
            stack.append((prev, pending_bond))
            pending_bond = None
            i += 1
        elif c == ")":
            if not stack:
                raise SmilesUnsupported("unbalanced_paren")
            prev, pending_bond = stack.pop()
            i += 1
        elif c in "-=#:/\\":
            if c == "=":
                pending_bond = 2
            elif c == "#":
                pending_bond = 3
            elif c == ":":
                pending_bond = AROM_ORDER
            else:
                pending_bond = 1
            i += 1
        elif c == "%":
            digits = smi[i + 1:i + 3]
            if len(digits) != 2 or not digits.isdigit():
                raise SmilesUnsupported("bad_ring_closure")
            num = int(digits)
            order = pending_bond
            pending_bond = None
            close_ring(num, order)
            i += 3
        elif c.isdigit():
            num = int(c)
            order = pending_bond
            pending_bond = None
            close_ring(num, order)
            i += 1
        else:
            atom, i = _read_atom(smi, i)
            idx = len(atoms)
            atoms.append(atom)
            if prev is not None:
                if pending_bond is not None:
                    order = pending_bond
                elif atom["aromatic"] and atoms[prev]["aromatic"]:
                    order = AROM_ORDER
                else:
                    order = 1
                add_bond(prev, idx, order)
            pending_bond = None
            prev = idx
    if stack:
        raise SmilesUnsupported("unbalanced_paren")
    if ring:
        raise SmilesUnsupported("unclosed_ring")
    if not atoms:
        raise SmilesUnsupported("empty_smiles")
    return atoms, bonds


def _aromatic_type(el, bracket_h, n_arom, n_other, vsum, std_valence,
                   has_exo_multiple):
    """Resolve (h, valence) for an aromatic atom or raise SmilesUnsupported.

    bracket_h: stated H count, or None for bare atoms. Bare `n` with two
    aromatic-only connections follows the RDKit convention (pyridine,
    H=0); this is sound for RDKit-canonical corpora and recorded as a
    corpus dependence. Anything structurally ambiguous is rejected, not
    guessed.

    Atoms with an exocyclic multiple bond (e.g. a pyrone carbonyl carbon,
    which RDKit still flags aromatic) cannot use aromatic valence 3: they
    fall back to aliphatic typing with the standard valence, which the
    uniform residual rule then validates. `vsum` uses ORDER_VALUE and
    `std_valence` is the aliphatic valence (C4 N3 O2 S2).
    """
    total = n_arom + n_other
    if has_exo_multiple and bracket_h is None and el in ("C", "N", "O", "S"):
        h = std_valence - vsum
        if h < 0:
            raise SmilesUnsupported("hypervalent")
        return h, std_valence
    if el == "C":
        if bracket_h is not None:
            if bracket_h != 3 - total:
                raise SmilesUnsupported("bracket_valence")
            return bracket_h, 3
        if total < 2:
            raise SmilesUnsupported("ambiguous_aromatic")
        h = 3 - total
        if h < 0:
            raise SmilesUnsupported("hypervalent")
        return h, 3
    if el == "N":
        if bracket_h is not None:
            return bracket_h, 3 if bracket_h > 0 else 2
        if n_arom == 2 and n_other == 0:
            return 0, 2
        if n_arom >= 2 and total == 3:
            return 0, 3
        raise SmilesUnsupported("ambiguous_aromatic")
    if el in ("O", "S"):
        if bracket_h is not None:
            return bracket_h, 2
        if total == 2:
            return 0, 2
        raise SmilesUnsupported("ambiguous_aromatic")
    raise SmilesUnsupported(f"ambiguous_aromatic:{el}")


def typed_from_parsed(atoms, bonds):
    """Shared typed-graph builder for SMILES and SDF front ends.

    atoms: list of dicts with element/aromatic/bracket/hydrogens/charge/
    valence. bonds: list of (a, b, order) with order in (1, 2, 3, 4).
    Returns (atom_types, edges) or (None, exclusion_reason). Hydrogens are
    parent-relative counts derived here, never invented downstream.
    Valence sums use ORDER_VALUE (aromatic code counts one unit).
    """
    from tools.ms2_database_retrieval import AROM_ORDER, ORDER_VALUE

    for a, b, o in bonds:
        if o == AROM_ORDER:
            # Aromatic bonds need aromatic-flagged endpoints (SMILES
            # lowercase). V2000 molfiles carry no such flags, so SDF bond
            # type 4 stays rejected; likewise explicit ':' on aliphatics.
            if not (atoms[a]["aromatic"] and atoms[b]["aromatic"]):
                return None, "aromatic_bond"
            continue
        if o not in (1, 2, 3):
            return None, f"unsupported_bond_type:{o}"
    for a in atoms:
        if a["charge"] != 0:
            return None, "unsupported_charge"
        if a["element"] not in DOMAIN_SET:
            return None, f"unsupported_element:{a['element']}"
    n = len(atoms)
    vsum = [0] * n
    narom = [0] * n
    nother = [0] * n
    exomult = [False] * n
    for a, b, o in bonds:
        try:
            value = ORDER_VALUE[o]
        except KeyError:
            return None, f"unsupported_bond_type:{o}"
        vsum[a] += value
        vsum[b] += value
        if o == AROM_ORDER:
            narom[a] += 1
            narom[b] += 1
        else:
            nother[a] += 1
            nother[b] += 1
            if value >= 2:
                exomult[a] = True
                exomult[b] = True
    typed = []
    try:
        for idx, a in enumerate(atoms):
            el = a["element"]
            if a["aromatic"]:
                h, val = _aromatic_type(
                    el, a["hydrogens"] if a["bracket"] else None,
                    narom[idx], nother[idx], vsum[idx],
                    VALENCE.get(el, 0), exomult[idx])
            elif a["bracket"]:
                h = a["hydrogens"]
                val = a["valence"]
                if vsum[idx] + h != val:
                    raise SmilesUnsupported("bracket_valence")
            else:
                val = a["valence"]
                h = val - vsum[idx]
                if h < 0:
                    raise SmilesUnsupported("hypervalent")
            typed.append((el, h, val))
    except SmilesUnsupported as exc:
        return None, exc.reason
    edges = tuple((a, b, o) for a, b, o in bonds)
    return tuple(typed), edges


def smiles_to_typed(smi):
    """Convert SMILES to (atom_types, edges) or return (None, reason)."""
    try:
        atoms, bonds = parse_smiles(smi)
    except SmilesUnsupported as exc:
        return None, exc.reason
    except (ValueError, IndexError) as exc:
        return None, f"syntax_error:{exc}"
    return typed_from_parsed(atoms, bonds)


def parse_formula(text):
    """Parse a Hill formula string (e.g. C16H17NO4) into a counts dict."""
    import re

    counts = {e: 0 for e in DOMAIN_ELEMENTS}
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        if el == "H":
            counts["H"] += int(num) if num else 1
        elif el in counts:
            counts[el] += int(num) if num else 1
        else:
            counts[el] = counts.get(el, 0) + (int(num) if num else 1)
    return counts


def standardize_smiles(smi, source, mid, expected_formula=None):
    """Parse + standardize one SMILES. Returns (record, exclusion_reason)."""
    typed, result = smiles_to_typed(smi)
    if typed is None:
        return None, result
    edges = result
    raw = {"id": mid, "source": source, "family": "msgym",
           "atom_types": typed, "edges": edges, "charge": 0,
           "original": {"smiles": smi}}
    record, reason = standardize_record(raw)
    if record is None:
        return None, reason
    if expected_formula is not None:
        if formula_key(record["formula"]) != formula_key(expected_formula):
            return None, "formula_mismatch"
    return record, None


# --------------------------------------------------------------------------
# Incremental candidate-JSON scanner (capped prefix -> complete entries).
# --------------------------------------------------------------------------

def iter_json_entries(path, max_entries):
    """Yield (query_smiles, candidates) for complete entries in file order.

    The file is a single large JSON object mapping query SMILES to arrays
    of candidate SMILES. Only complete entries are returned; a trailing
    partial entry from a Range-request prefix is dropped.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    entries = []
    i, n = 0, len(text)
    try:
        while len(entries) < max_entries and i < n:
            # Find next string token (object key).
            while i < n and text[i] != '"':
                if text[i] == "}":
                    return iter(entries)
                i += 1
            if i >= n:
                break
            key, i = _scan_string(text, i)
            while i < n and text[i] in " \t\r\n":
                i += 1
            if i >= n or text[i] != ":":
                break
            i += 1
            while i < n and text[i] in " \t\r\n":
                i += 1
            if i >= n or text[i] != "[":
                break
            value_text, i = _scan_array(text, i)
            try:
                candidates = json.loads(value_text)
                query = json.loads(key)
            except json.JSONDecodeError:
                continue
            entries.append((query, candidates))
    except SmilesUnsupported:
        pass  # truncated prefix (key or array): keep complete entries only
    return iter(entries)


def _scan_string(text, i):
    assert text[i] == '"'
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return text[i:j + 1], j + 1
        j += 1
    raise SmilesUnsupported("truncated_string")


def _scan_array(text, i):
    assert text[i] == "["
    depth = 0
    j = i
    n = len(text)
    in_str = False
    while j < n:
        c = text[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1], j + 1
        j += 1
    raise SmilesUnsupported("truncated_array")


# --------------------------------------------------------------------------
# TSV loading (val fold, capped, deduplicated by structure string).
# --------------------------------------------------------------------------

WANTED_COLUMNS = ("identifier", "smiles", "inchikey", "formula",
                  "precursor_formula", "parent_mass", "precursor_mz",
                  "adduct", "instrument_type", "collision_energy", "fold")


def load_tsv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in WANTED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"TSV missing columns: {missing}")
        for row in reader:
            yield {c: row[c] for c in WANTED_COLUMNS}


def index_val_by_smiles(path):
    """Map smiles -> list of val-fold rows (dedup key: raw SMILES string)."""
    groups = {}
    n_rows = 0
    for row in load_tsv_rows(path):
        n_rows += 1
        if row["fold"] != "val":
            continue
        groups.setdefault(row["smiles"], []).append(row)
    return groups, n_rows


# --------------------------------------------------------------------------
# Query construction from a true structure (synthetic open subgraphs).
# --------------------------------------------------------------------------

def extract_patterns(atom_types, edges):
    """Single-atom, single-bond, and first overlapping bond-pair patterns."""
    if not edges:
        return ((atom_types[0],), ()), None
    a, b, o = edges[0]
    single = (((atom_types[a], atom_types[b])), ((0, 1, o),))
    pair = None
    for c, d, p in edges[1:]:
        if len({a, b, c, d}) == 3:
            at = [atom_types[a], atom_types[b]]
            third = c if c not in (a, b) else d
            at.append(atom_types[third])
            # Re-index: keep shared atom first for a compact pattern.
            shared = a if a in (c, d) else b
            others = [x for x in (a, b) if x != shared] + [third]
            order = {tuple(sorted((x, y))): q for x, y, q in edges}
            mapping = {shared: 0, others[0]: 1, others[1]: 2}
            pe = tuple(sorted(
                (mapping[x], mapping[y], q)
                for (x, y), q in order.items()
                if x in (shared,) + tuple(others) and y in (shared,) + tuple(others)
                and {x, y} <= {shared, others[0], others[1]}
                and ((x == shared and y in others) or (y == shared and x in others))))
            pair = ((tuple([atom_types[shared]] + [atom_types[x] for x in others])), pe)
            break
    return single, pair


def run_msgym_experiment(prefix_path, tsv_path, max_queries, candidate_cap=None):
    """Capped val-fold run over supplied formula candidate pools."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    val_groups, n_tsv_rows = index_val_by_smiles(tsv_path)
    entries = [(q, c) for q, c in iter_json_entries(prefix_path, max_queries * 20)]
    # File-order slice: first entries that join to a val row.
    joined = [(q, c) for q, c in entries if q in val_groups][:max_queries]
    pool_stats = Counter()
    target_stats = Counter()
    queries = []  # (name, target_record, patterns_list, row)
    for qi, (qsmiles, cands) in enumerate(joined):
        row = val_groups[qsmiles][0]
        try:
            expected = parse_formula(row["formula"])
        except Exception:
            target_stats["formula_unparseable"] += 1
            continue
        target, reason = standardize_smiles(
            qsmiles, "msgym-val", f"MSGYM-Q{qi:04d}", expected)
        if target is None:
            target_stats[reason.split(":")[0]] += 1
            continue
        target_stats["in_domain"] += 1
        # Mass agreement: graph-derived theoretical neutral mass vs the
        # provider's parent_mass, which is MEASURED (precursor m/z minus
        # proton mass; sub-ppm Orbitrap errors observed). Agreement within
        # 10 ppm validates the parser's H-count logic against an independent
        # column; the separate formula cross-check already passed.
        try:
            from decimal import Decimal

            observed_mu = int(Decimal(row["parent_mass"]) * 1_000_000)
            ppm = abs(observed_mu - target["mass"]) / target["mass"] * 1e6
        except Exception:
            ppm = None
        mass_ok = ppm is not None and ppm <= 10.0
        pool_stats["mass_agree" if mass_ok else "mass_mismatch"] += 1
        pool = []
        capped = cands if candidate_cap is None else cands[:candidate_cap]
        for ci, csmiles in enumerate(capped):
            pool_stats["n_supplied"] += 1
            rec, rreason = standardize_smiles(
                csmiles, "msgym-pool", f"MSGYM-Q{qi:04d}-C{ci:04d}")
            if rec is None:
                pool_stats[f"pool_excl:{rreason.split(':')[0]}"] += 1
                continue
            pool_stats["pool_in_domain"] += 1
            pool.append(rec)
        single, pair = extract_patterns(target["atom_types"], target["edges"])
        pats_mass_only = []
        pats_single = [single] if single else []
        pats_two = [single, pair] if pair else pats_single
        queries.append((f"MSGYM-Q{qi:04d}", target, pool, row,
                        pats_mass_only, pats_single, pats_two))
    rows = []
    import sys as _sys

    for qi, (name, target, pool, row, p0, p1, p2) in enumerate(queries):
        # Supplied pools are used verbatim; the target is never injected.
        sub = DatabaseIndex(pool)
        qsmiles = row["smiles"]
        # Pool membership by exact (RDKit-canonical) SMILES string. This is
        # exact up to the provider's canonicalization; the fingerprint-based
        # flag below is a weaker upper bound (collisions possible).
        strict_in_pool = any(
            r["original"].get("smiles") == qsmiles for r in pool)
        loose_in_pool = strict_in_pool or any(
            formula_key(r["formula"]) == formula_key(target["formula"])
            and edge_fingerprint(r["atom_types"], r["edges"])
            == edge_fingerprint(target["atom_types"], target["edges"])
            for r in pool)
        try:
            from decimal import Decimal

            obs_mu = int(Decimal(row["parent_mass"]) * 1_000_000)
            ppm_err = abs(obs_mu - target["mass"]) / target["mass"] * 1e6
        except Exception:
            ppm_err = None
        for qname, pats in (("mass_only", p0), ("single_bond", p1),
                            ("bonded_pair", p2)):
            # Bonded patterns only: single-atom patterns are combinatorially
            # promiscuous on drug-sized targets; their behavior is covered by
            # the synthetic harness. 100k embedding nodes per arm, then an
            # explicit search_budget_exhausted status (never a silent cut).
            counters = WorkCounters(limit=100_000)
            res = run_query(sub, target, pats, "unknown", None,
                            counters=counters)
            present = {stage: any(
                r["original"].get("smiles") == qsmiles for r in res[stage])
                for stage in ("s1", "s2", "s3")}
            # Rank the surviving pool; the target-equivalent member (by
            # string) is the top-k reference. Absent target => keys gated.
            tgt_rid = next(
                (r["id"] for r in res["s3"]
                 if r["original"].get("smiles") == qsmiles), "__absent__")
            rank, _ = rank_baselines(res["s3"], tgt_rid, sub)
            rows.append({
                "query": f"{name}:{qname}", "target": target["id"],
                "in_supplied_pool": strict_in_pool,
                "in_pool_loose_fp": loose_in_pool,
                "recall_s1": present["s1"], "recall_s2": present["s2"],
                "recall_s3": present["s3"],
                "n_supplied": len(pool),
                "n_s1": len(res["s1"]), "n_s2": len(res["s2"]),
                "n_s3": len(res["s3"]),
                "zero_status": res["zero_status"],
                "top1_uniform": rank.get("uniform@1"),
                "top1_fingerprint": rank.get("fingerprint@1"),
                "top3_fingerprint": rank.get("fingerprint@3"),
                "heavy": target["heavy"],
                "n_pool": len(pool),
                "adduct": row["adduct"],
                "instrument": row["instrument_type"],
                "ppm_err": round(ppm_err, 3) if ppm_err is not None else None,
                "truncated": res["truncated"],
                "embedding_nodes": counters.embedding_nodes,
            })
        if (qi + 1) % 5 == 0 or qi + 1 == len(queries):
            _sys.stderr.write(f"# progress {qi + 1}/{len(queries)}\n")
            _sys.stderr.flush()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return {
        "rows": rows,
        "n_tsv_rows": n_tsv_rows,
        "n_val_structures": len(val_groups),
        "n_entries_scanned": len(entries),
        "n_queries": len(queries),
        "pool_stats": dict(pool_stats),
        "target_stats": dict(target_stats),
        "wall_s": wall,
        "cpu_s": cpu,
    }


def main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--max-queries", type=int, default=150)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    result = run_msgym_experiment(args.prefix, args.tsv, args.max_queries)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
    summary = {k: v for k, v in result.items() if k != "rows"}
    sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
    for r in result["rows"][:10]:
        sys.stderr.write(str(r) + "\n")


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

class SmilesTests(unittest.TestCase):
    def test_ethanol(self):
        typed, edges = smiles_to_typed("CCO")
        f = formula_of(typed)
        self.assertEqual((f["C"], f["H"], f["O"]), (2, 6, 1))
        rec, reason = standardize_record({"id": "t", "atom_types": typed,
                                          "edges": edges, "charge": 0})
        self.assertIsNotNone(rec, reason)

    def test_branch_and_double_bond(self):
        typed, _ = smiles_to_typed("CC(=O)O")
        self.assertEqual(formula_of(typed)["H"], 4)  # acetic acid C2H4O2

    def test_ring_closure(self):
        typed, edges = smiles_to_typed("C1CCOC1")  # tetrahydrofuran
        self.assertEqual(len(typed), 5)
        self.assertEqual(len(edges), 5)

    def test_percent_ring(self):
        typed, _ = smiles_to_typed("C%10CCCCC%10")  # cyclohexane via %10
        self.assertEqual(len(typed), 6)

    def test_benzene_aromatic(self):
        typed, edges = smiles_to_typed("c1ccccc1")
        self.assertEqual(formula_of(typed)["H"], 6)  # C6H6
        self.assertTrue(all(o == 4 for _, _, o in edges))
        rec, reason = standardize_record({"id": "t", "atom_types": typed,
                                          "edges": edges, "charge": 0})
        self.assertIsNotNone(rec, reason)

    def test_pyridine_pyrrole_thiophene(self):
        for smi, h, n in (("n1ccccc1", 5, 1), ("c1cc[nH]c1", 5, 1),
                          ("c1ccsc1", 4, 0)):
            typed, _ = smiles_to_typed(smi)
            f = formula_of(typed)
            self.assertEqual((f["H"], f["N"]), (h, n), smi)

    def test_aromatic_chloride(self):
        typed, _ = smiles_to_typed("Clc1ccccc1")
        self.assertEqual(formula_of(typed),
                         {"C": 6, "H": 5, "N": 0, "O": 0, "F": 0,
                          "S": 0, "Cl": 1, "Br": 0, "I": 0})

    def test_sulfone_hypervalent(self):
        _, reason = smiles_to_typed("CCS(=O)(=O)C")
        self.assertEqual(reason, "hypervalent")

    def test_bare_n_ambiguous(self):
        _, reason = smiles_to_typed("nC")
        self.assertEqual(reason, "ambiguous_aromatic")

    def test_pyrone_carbonyl_fallback(self):
        # RDKit flags pyrone carbonyl carbons aromatic; with an exocyclic
        # double bond they take aliphatic typing (uniform residual rule).
        rec, reason = standardize_smiles(
            "O=c1oc2ccccc2c(O)c1C", "t", "T", parse_formula("C10H8O3"))
        self.assertIsNotNone(rec, reason)
        self.assertEqual(rec["formula"]["H"], 8)

    def test_aromatic_kekule_boundary(self):
        # Representation boundary (documented): aromatic-order patterns do
        # not match kekule-ordered targets and vice versa, even for the
        # same molecule. Within one canonical corpus this never triggers.
        from tools.ms2_database_retrieval import contains_exact

        kt, ke = smiles_to_typed("C1=CC=CC=C1")
        at, ae = smiles_to_typed("c1ccccc1")
        apat = ((at[0], at[1]), ((0, 1, ae[0][2]),))
        self.assertFalse(contains_exact(kt, ke, *apat))
        self.assertTrue(contains_exact(at, ae, *apat))

    def test_charge_excluded(self):
        _, reason = smiles_to_typed("CC[N+](=O)[O-]")
        self.assertEqual(reason, "unsupported_charge")

    def test_halogen_alkyl_now_supported(self):
        # Monovalent halogens joined the domain; silicon stays out.
        typed, _ = smiles_to_typed("CCCl")
        self.assertEqual(formula_of(typed)["Cl"], 1)
        _, reason = smiles_to_typed("CC[Si]")
        self.assertEqual(reason, "unsupported_element:Si")

    def test_disconnected_excluded(self):
        _, reason = smiles_to_typed("CC.[Na+]")
        self.assertEqual(reason, "disconnected_or_salt")

    def test_bracket_stereo_ok(self):
        typed, _ = smiles_to_typed("C[C@H](N)O")
        self.assertEqual(formula_of(typed)["H"], 7)  # C2H7NO

    def test_kekule_benzene_accepted_as_domain(self):
        typed, edges = smiles_to_typed("C1=CC=CC=C1")
        rec, reason = standardize_record({"id": "t", "atom_types": typed,
                                          "edges": edges, "charge": 0})
        self.assertIsNotNone(rec, reason)

    def test_formula_parser(self):
        self.assertEqual(parse_formula("C16H17NO4")["C"], 16)
        self.assertEqual(parse_formula("C16H17NO4")["H"], 17)
        self.assertEqual(parse_formula("C16H17NO4")["N"], 1)

    def test_formula_crosscheck(self):
        typed, _ = smiles_to_typed("CCO")
        rec, reason = standardize_smiles("CCO", "t", "t", parse_formula("C2H6O"))
        self.assertIsNotNone(rec, reason)
        rec2, reason2 = standardize_smiles("CCO", "t", "t", parse_formula("C2H6O2"))
        self.assertIsNone(rec2)
        self.assertEqual(reason2, "formula_mismatch")

    def test_extract_patterns_roundtrip(self):
        from tools.ms2_database_retrieval import joint_contains, WorkCounters

        typed, edges = smiles_to_typed("CC(=O)O")
        single, pair = extract_patterns(typed, edges)
        self.assertTrue(joint_contains(typed, edges, [single], WorkCounters()))
        if pair is not None:
            self.assertTrue(
                joint_contains(typed, edges, [single, pair], WorkCounters()))
        import tempfile, os

        doc = '{"CCO": ["CCO", "COC"], "CCC": ["CCC"]}'
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write(doc)
            path = fh.name
        try:
            out = list(iter_json_entries(path, 10))
            self.assertEqual(out[0][0], "CCO")
            self.assertEqual(out[0][1], ["CCO", "COC"])
            self.assertEqual(out[1][0], "CCC")
        finally:
            os.unlink(path)

    def test_json_scanner_drops_partial(self):
        import tempfile, os

        doc = '{"CCO": ["CCO", "COC"], "CCC": ["CC'
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write(doc)
            path = fh.name
        try:
            out = list(iter_json_entries(path, 10))
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0][0], "CCO")
        finally:
            os.unlink(path)

    def test_json_scanner_drops_truncated_key(self):
        # Review regression: a prefix ending mid-key crashed instead of
        # returning the complete entries before it.
        import tempfile, os

        doc = '{"CCO": ["CCO"], "C'
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write(doc)
            path = fh.name
        try:
            out = list(iter_json_entries(path, 10))
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0][0], "CCO")
        finally:
            os.unlink(path)

    def test_isotope_bracket_rejected(self):
        # Review regression: [13C] silently normalized to 12C with the
        # unlabeled mass. Isotope labels must be rejected, not dropped.
        rec, reason = standardize_smiles("[13CH4]", "t", "T")
        self.assertIsNone(rec)
        self.assertEqual(reason, "unsupported_isotope")


if __name__ == "__main__":
    main(sys.argv[1:])
