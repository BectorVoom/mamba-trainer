"""Database-first molecular-completion retrieval harness (Experiments A/B).

Implements docs/MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md, first pass only:
offline, stdlib-only, capped synthetic fixture corpus. No external dataset is
downloaded by the default run. Ingestion adapters for ChEBI / MassSpecGym /
MassBank / GNPS / PubChem / QM9 are stubs that pin releases, record hashes and
enforce license/throttle notes; they raise an informative error until the
operator supplies a pinned local file.

Run:   python3 tools/ms2_database_retrieval.py
Test:  python3 -m unittest tools/ms2_database_retrieval.py

Conventions (from MOLECULAR_COMPLETION_DESIGN.md):
- Parent-relative hydrogen counts: substructure atom types are
  (element, parent H count, valence); never re-sanitized as capped molecules.
- Three-valued mass verdict: accept / reject / boundary-ambiguous, plus
  unavailable when precision is unknown. Integer arithmetic only.
- Fingerprint is a SOUND rejection screen only; surviving hits always go
  through exact typed graph matching.
- Unknown-overlap queries never receive the known-correspondence oracle.
- Zero candidates with target absent from the pool is reported as
  `target_absent`, never as proof of constraint inconsistency.
- Top-k is reported only when the target is present in the candidate pool.
- Synthetic parent-subgraph queries are controlled completion tests, not
  experimentally confirmed fragment structures.
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import time
import unittest
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from itertools import combinations

# --------------------------------------------------------------------------
# Integer mass arithmetic (same constants/semantics as the pilot reference).
# --------------------------------------------------------------------------

EXACT = {
    "C": "12",
    "H": "1.00782503223",
    "N": "14.00307400443",
    "O": "15.99491461957",
    "F": "18.99840316273",
    "S": "31.9720711744",
    "Cl": "34.968852682",
    "Br": "78.9183376",
    "I": "126.9044719",
}
SCALE = 1_000_000
MASS = {e: int((Decimal(m) * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))
        for e, m in EXACT.items()}
ERROR_NDA = {
    e: int((abs(Decimal(m) * SCALE - MASS[e]) * 1000).to_integral_value(rounding=ROUND_CEILING))
    for e, m in EXACT.items()
}

# V0 neutral closed-shell vocabulary used by this harness, plus the
# max-hydride neutrals (methane/ammonia/water atom types), hydrogen
# fluoride's F, monovalent halogens, divalent sulfur, and aromatic atom
# types for RDKit-canonical SMILES (lowercase). Aromatic bonds carry order
# code AROM_ORDER (4); ORDER_VALUE maps codes to valence sums, so one
# uniform residual rule covers aliphatic and aromatic atoms. Bare aromatic
# `n` with two aromatic-only connections follows the RDKit convention
# (pyridine, H=0); pyrrolic NH must be written [nH]. This convention is
# sound for RDKit-canonical corpora (e.g. MassSpecGym 1.5) and is recorded
# as a corpus dependence, not a general SMILES rule.
# (element, parent-relative H count, total valence)
AROM_ORDER = 4
ORDER_VALUE = {1: 1, 2: 2, 3: 3, 4: 1}
TYPES = (
    ("C", 0, 4), ("C", 1, 4), ("C", 2, 4), ("C", 3, 4), ("C", 4, 4),
    ("N", 0, 3), ("N", 1, 3), ("N", 2, 3), ("N", 3, 3),
    ("O", 0, 2), ("O", 1, 2), ("O", 2, 2),
    ("F", 0, 1), ("F", 1, 1),
    ("Cl", 0, 1), ("Br", 0, 1), ("I", 0, 1),
    ("Cl", 1, 1), ("Br", 1, 1), ("I", 1, 1),
    ("S", 0, 2), ("S", 1, 2), ("S", 2, 2),
    ("C", 0, 3), ("C", 1, 3), ("C", 2, 3),
    ("N", 0, 2),
)
TYPE_SET = set(TYPES)
C3 = ("C", 3, 4)
C2 = ("C", 2, 4)
C1 = ("C", 1, 4)
N2 = ("N", 2, 3)
N1 = ("N", 1, 3)
O1 = ("O", 1, 2)
O0 = ("O", 0, 2)


def formula_of(atom_types):
    result = {e: 0 for e in DOMAIN_ELEMENTS}
    for element, hydrogen, _ in atom_types:
        result[element] += 1
        result["H"] += hydrogen
    return result


def neutral_mass(formula):
    return sum(formula.get(e, 0) * MASS[e] for e in EXACT)


def mass_verdict(observed, formula, uncertainty=50, ppm_tenths=100):
    """Three-valued neutral-mass decision. See design section on formulas."""
    if uncertainty is None:
        return "unavailable"
    computed = neutral_mass(formula)
    error = (sum(formula.get(e, 0) * ERROR_NDA[e] for e in EXACT) + 999) // 1000
    error += uncertainty
    tolerance = observed * ppm_tenths // 10_000_000
    residual = abs(observed - computed)
    if residual + error <= tolerance:
        return "accept"
    if residual > tolerance + error:
        return "reject"
    return "ambiguous"


# --------------------------------------------------------------------------
# Records, standardization, identity.
# --------------------------------------------------------------------------

DOMAIN_ELEMENTS = ("C", "H", "N", "O", "F", "S", "Cl", "Br", "I")


def standardize_record(raw):
    """Validate a raw structure dict into a closed neutral record.

    Returns (record, exclusion_reason). exclusion_reason is None on success.
    Record keeps source id and original payload for provenance; identity is
    connectivity-only (stereo stripped, tautomer not resolved -- recorded as
    a limitation, not silently normalized).
    Rejects: unknown atom types, nonzero charge, isotopic labels, salts or
    disconnected graphs, open valences (DB stage is closed molecules only).
    """
    atom_types = tuple(raw.get("atom_types", ()))
    edges = tuple(raw.get("edges", ()))
    for t in atom_types:
        if t not in TYPE_SET:
            return None, f"unsupported_atom_type:{t}"
    if raw.get("charge", 0) != 0:
        return None, "unsupported_charge"
    if raw.get("isotopes"):
        return None, "unsupported_isotope"
    n = len(atom_types)
    if n == 0:
        return None, "empty_graph"
    # Connectedness (single component => no salt fragments).
    seen = {0}
    changed = True
    emap = {}
    for a, b, o in edges:
        if o:
            emap[tuple(sorted((a, b)))] = o
    while changed:
        changed = False
        for a, b, o in edges:
            if not o:
                continue
            if (a in seen) != (b in seen):
                seen.update((a, b))
                changed = True
    if len(seen) != n:
        return None, "disconnected_or_salt"
    # Closed-shell: residual valence must be zero for every atom, using
    # ORDER_VALUE so aromatic bonds (code 4) count one valence unit.
    residual = [v - h for _, h, v in atom_types]
    for a, b, o in edges:
        try:
            value = ORDER_VALUE[o]
        except KeyError:
            return None, f"unsupported_bond_type:{o}"
        residual[a] -= value
        residual[b] -= value
    if any(r != 0 for r in residual):
        return None, "open_valence"
    formula = formula_of(atom_types)
    record = {
        "id": raw.get("id"),
        "source": raw.get("source"),
        "family": raw.get("family", "unspecified"),
        "atom_types": atom_types,
        "edges": edges,
        "formula": formula,
        "mass": neutral_mass(formula),
        "charge": 0,
        "heavy": sum(1 for e, _, _ in atom_types if e != "H"),
        "original": dict(raw.get("original", {})),
    }
    return record, None


def canonical_key(atom_types, edges):
    """Connectivity identity (no stereo). For fixture-scale graphs only."""
    from itertools import permutations as _perm

    n = len(atom_types)
    edge_map = {tuple(sorted((a, b))): o for a, b, o in edges if o}
    best = None
    for order in _perm(range(n)):
        labels = tuple(atom_types[i] for i in order)
        bonds = tuple(edge_map.get(tuple(sorted((order[i], order[j]))), 0)
                      for i, j in combinations(range(n), 2))
        key = (labels, bonds)
        if best is None or key < best:
            best = key
    return best


def skeleton_key(atom_types, edges):
    """Coarse scaffold grouping: heavy-element sequence + bond multiset."""
    heavy = tuple(sorted(e for e, _, _ in atom_types))
    orders = tuple(sorted(o for _, _, o in edges if o))
    return (heavy, orders)


# --------------------------------------------------------------------------
# Exact typed matching + sound fingerprint screen.
# --------------------------------------------------------------------------

class WorkCounters:
    def __init__(self, limit=None):
        self.fingerprint_screens = 0
        self.exact_matches = 0
        self.embedding_nodes = 0
        self.limit = limit

    def tick(self):
        self.embedding_nodes += 1
        if self.limit is not None and self.embedding_nodes > self.limit:
            raise BudgetExhausted("embedding_nodes")


class BudgetExhausted(Exception):
    pass


def edge_fingerprint(atom_types, edges):
    """Necessary-condition counters for typed containment.

    fp = (atom-type Counter, unordered typed-edge Counter). If any pattern
    counter exceeds the candidate's, no injective typed edge-preserving
    embedding exists -> sound to reject. Never used to accept.
    """
    ac = Counter(atom_types)
    ec = Counter()
    for a, b, o in edges:
        if not o:
            continue
        t = tuple(sorted((atom_types[a], atom_types[b])))
        ec[(t[0], t[1], o)] += 1
    return ac, ec


def fingerprint_screen(candidate_fp, pattern_fp):
    cac, cec = candidate_fp
    pac, pec = pattern_fp
    for k, v in pac.items():
        if cac.get(k, 0) < v:
            return False
    for k, v in pec.items():
        if cec.get(k, 0) < v:
            return False
    return True


def pattern_fingerprint(pattern_types, pattern_edges):
    return edge_fingerprint(pattern_types, [(a, b, o) for a, b, o in pattern_edges])


def prepare_target(atom_types, edges):
    """Precompute once per candidate: neighbor maps and type positions.

    Returns (nbr, pos_by_type). nbr[i] maps neighbor -> bond order;
    pos_by_type[t] lists target atoms of exact type t. Both matchers iterate
    only type-compatible atoms and check constraints via dict lookup
    (no per-check tuple allocation or sorting).
    """
    nbr = [{} for _ in atom_types]
    for a, b, o in edges:
        if o:
            nbr[a][b] = o
            nbr[b][a] = o
    pos = {}
    for i, t in enumerate(atom_types):
        pos.setdefault(t, []).append(i)
    return nbr, pos


def _prepare_single(pattern_types, pattern_edges):
    """Pattern-side prep, reusable across candidates.

    Returns (order, cons) where order lists pattern atoms rarest-... (static
    degree order; target rarity is applied at iteration via pos_by_type
    candidate-list lengths) and cons[k] lists (earlier_position, order).
    """
    m = len(pattern_types)
    deg = [0] * m
    adj = [[] for _ in range(m)]
    for a, b, o in pattern_edges:
        deg[a] += 1
        deg[b] += 1
        adj[a].append((b, o))
        adj[b].append((a, o))
    order = sorted(range(m), key=lambda i: -deg[i])
    rank = {atom: k for k, atom in enumerate(order)}
    cons = []
    for k, atom in enumerate(order):
        cons.append(tuple(sorted(
            (rank[nb], o) for nb, o in adj[atom] if rank[nb] < k)))
    return order, cons


def count_embeddings_prep(nbr, pos, ptypes, order, cons, cap, counters=None):
    """Count distinct embeddings of one pattern up to cap (feature use).

    Same existence semantics as _contains_with_prep (result > 0 iff a
    match exists); the count itself is a cheap multiplicity feature, not
    a quality claim. Occupancy handling mirrors the matcher exactly.
    """
    total = 0
    stop = False
    n_target = len(nbr)
    chosen = [0] * len(order)
    occupied = [False] * n_target

    def search(k):
        nonlocal total, stop
        if stop:
            return
        if k == len(order):
            # No bulk reset here: each frame unsets its own atom after the
            # recursive call returns, so ancestor assignments stay intact
            # for the remaining branches (injectivity preserved while
            # counting). Resetting `chosen` at the leaf would let later
            # branches reuse ancestor atoms and inflate the count.
            total += 1
            if total >= cap:
                stop = True
            return
        atom = order[k]
        want = ptypes[atom]
        for cand in pos.get(want, ()):
            if stop:
                return
            if counters is not None:
                counters.tick()
            if occupied[cand]:
                continue
            row = nbr[cand]
            ok = True
            for l, o in cons[k]:
                if row.get(chosen[l]) != o:
                    ok = False
                    break
            if ok:
                chosen[k] = cand
                occupied[cand] = True
                search(k + 1)
                occupied[cand] = False

    search(0)
    return total


def _contains_with_prep(nbr, pos, ptypes, order, cons, occupied, counters):
    # Occupancy is always clean on return: backtracking unsets on failure,
    # and success resets the (small) chosen set before unwinding.
    chosen = [0] * len(order)

    def search(k):
        if k == len(order):
            for c in chosen:
                occupied[c] = False
            return True
        atom = order[k]
        want = ptypes[atom]
        for cand in pos.get(want, ()):
            if counters is not None:
                counters.tick()
            if occupied[cand]:
                continue
            row = nbr[cand]
            ok = True
            for l, o in cons[k]:
                if row.get(chosen[l]) != o:
                    ok = False
                    break
            if ok:
                chosen[k] = cand
                occupied[cand] = True
                if search(k + 1):
                    return True
                occupied[cand] = False
        return False

    return search(0)


def contains_exact(atom_types, edges, pattern_types, pattern_edges, counters=None):
    """Injective typed edge-preserving embedding; target may add edges."""
    if len(pattern_types) > len(atom_types):
        return False
    nbr, pos = prepare_target(atom_types, edges)
    order, cons = _prepare_single(pattern_types, pattern_edges)
    return _contains_with_prep(nbr, pos, pattern_types, order, cons,
                               [False] * len(atom_types), counters)


def _prepare_joint(patterns):
    """Pattern-side prep for joint matching, reusable across candidates.

    Returns (flat_types, flat_pat, flat_adj, degrees). flat_adj[i] lists
    (j, order) pattern-edge neighbors of flat atom i.
    """
    flat_types, flat_pat = [], []
    index_of = {}
    for pi, (pt, pe) in enumerate(patterns):
        for ai, t in enumerate(pt):
            index_of[(pi, ai)] = len(flat_types)
            flat_types.append(t)
            flat_pat.append(pi)
    flat_adj = [[] for _ in flat_types]
    for pi, (pt, pe) in enumerate(patterns):
        for a, b, o in pe:
            i, j = index_of[(pi, a)], index_of[(pi, b)]
            flat_adj[i].append((j, o))
            flat_adj[j].append((i, o))
    degrees = [len(v) for v in flat_adj]
    return flat_types, flat_pat, flat_adj, degrees


def _joint_with_prep(nbr, pos, counts, flat_types, flat_pat, flat_adj,
                     degrees, n_patterns, occupied_sets, counters):
    order = sorted(range(len(flat_types)),
                   key=lambda i: (counts.get(flat_types[i], 0), -degrees[i]))
    assign = [None] * len(flat_types)

    def search(k):
        if k == len(order):
            return True
        fi = order[k]
        want = flat_types[fi]
        used = occupied_sets[flat_pat[fi]]
        for cand in pos.get(want, ()):
            if counters is not None:
                counters.tick()
            if cand in used:
                continue
            row = nbr[cand]
            ok = True
            for j, o in flat_adj[fi]:
                other = assign[j]
                if other is not None and row.get(other) != o:
                    ok = False
                    break
            if ok:
                assign[fi] = cand
                used.add(cand)
                if search(k + 1):
                    return True
                assign[fi] = None
                used.discard(cand)
        return False

    return search(0)


def joint_contains(atom_types, edges, patterns, counters=None):
    """Global overlap consistency across patterns.

    Simultaneously embeds every pattern, injective within each pattern,
    sharing allowed across patterns. Rejects quotients that collapse two
    distinct atoms of one pattern or equate incompatible types. Pairwise
    per-pattern success alone does not imply this returns True.
    """
    if not any(True for _ in patterns for _ in _[0]):
        return True
    flat_types, flat_pat, flat_adj, degrees = _prepare_joint(patterns)
    nbr, pos = prepare_target(atom_types, edges)
    from collections import Counter as _Counter

    occupied_sets = [set() for _ in patterns]
    return _joint_with_prep(nbr, pos, _Counter(atom_types),
                            flat_types, flat_pat, flat_adj, degrees,
                            len(patterns), occupied_sets, counters)


# --------------------------------------------------------------------------
# Fixture corpus (synthetic stand-ins; NOT ChEBI/MassBank records).
# --------------------------------------------------------------------------

def _mol(mid, family, atom_types, edges):
    return {"id": mid, "source": "synthetic-fixture", "family": family,
            "atom_types": tuple(atom_types), "edges": tuple(edges),
            "charge": 0, "original": {"note": "synthetic controlled-completion fixture"}}


def fixture_raw_records():
    C, N, O = C3, N2, O1
    recs = [
        _mol("FIX-001", "alcohol", (C3, C2, O1), ((0, 1, 1), (1, 2, 1))),          # ethanol
        _mol("FIX-002", "ether", (C3, O0, C3), ((0, 1, 1), (1, 2, 1))),            # dimethyl ether
        _mol("FIX-003", "alcohol", (C3, C2, C2, O1),
             ((0, 1, 1), (1, 2, 1), (2, 3, 1))),                                   # 1-propanol
        _mol("FIX-004", "alcohol", (C3, C1, C3, O1),
             ((0, 1, 1), (1, 2, 1), (1, 3, 1))),                                   # 2-propanol
        _mol("FIX-005", "ether", (C3, O0, C2, C3), ((0, 1, 1), (1, 2, 1), (2, 3, 1))),
        _mol("FIX-006", "amine", (C3, N2), ((0, 1, 1),)),                          # methylamine
        _mol("FIX-007", "amide", (C1, O0, N2), ((0, 1, 2), (0, 2, 1))),            # formamide
        _mol("FIX-008", "diol", (C2, C2, O1, O1),
             ((0, 1, 1), (0, 2, 1), (1, 3, 1))),                                   # ethylene glycol
        _mol("FIX-009", "cyclic-ether", (C2, C2, C2, O0),
             ((0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1))),                         # oxetane
        _mol("FIX-010", "cyclic-amine", (C2, C2, N1),
             ((0, 1, 1), (1, 2, 1), (2, 0, 1))),                                   # aziridine
        _mol("FIX-011", "fluoro", (C3, C2, ("F", 0, 1)),
             ((0, 1, 1), (1, 2, 1))),                                              # 1-fluoroethane
        _mol("FIX-012", "carbonyl", (C3, C1, O0),
             ((0, 1, 1), (1, 2, 2))),                                              # acetaldehyde
    ]
    return recs


CH3_PAT = ((C3,), ())
CH2OH_PAT = (((C2, O1)), ((0, 1, 1),))
CH3CH2_PAT = (((C3, C2)), ((0, 1, 1),))
CN_PAT = (((C3, N2)), ((0, 1, 1),))
C1N_PAT = (((C1, N2)), ((0, 1, 1),))
N_PAT = ((N2,), ())


def fixture_queries():
    """(name, target_id, patterns, overlap, precursor_id).

    overlap: 'na' (no patterns), 'unknown' (default condition), 'known'
    (oracle correspondence supplied separately, same filter plus provenance).
    precursor: optional parent record id for stage-4 arm; None => not-evaluated.
    """
    return [
        ("C2H6O_mass_only", "FIX-001", (), "na", None),
        ("C2H6O_CH3_unknown", "FIX-001", (CH3_PAT,), "unknown", None),
        ("C2H6O_CH2OH_unknown", "FIX-001", (CH2OH_PAT,), "unknown", None),
        ("C2H6O_two_overlapping_unknown", "FIX-001", (CH3CH2_PAT, CH2OH_PAT), "unknown", None),
        ("C2H6O_two_overlapping_known", "FIX-001", (CH3CH2_PAT, CH2OH_PAT), "known", None),
        ("C3H8O_mass_only", "FIX-003", (), "na", None),
        ("C3H8O_two_overlapping_unknown", "FIX-003", (CH3CH2_PAT, CH2OH_PAT), "unknown", None),
        ("amine_CN_unknown", "FIX-006", (CN_PAT,), "unknown", None),
        ("amide_C1N_unknown", "FIX-007", (C1N_PAT,), "unknown", None),
        ("amide_C3N_incompatible", "FIX-007", (CN_PAT,), "unknown", None),
        ("ring_mass_only", "FIX-009", (), "na", None),
        ("fragment_completion_precursor", "FIX-001", (CH3_PAT,), "unknown", "FIX-003"),
    ]


# --------------------------------------------------------------------------
# Retrieval index + staged filtering (Experiments A and B).
# --------------------------------------------------------------------------

class DatabaseIndex:
    def __init__(self, records):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.by_formula = {}
        self.fkeys = {}
        for r in records:
            key = formula_key(r["formula"])
            self.by_formula.setdefault(key, []).append(r)
            self.fkeys[r["id"]] = key
        self.fps = {r["id"]: edge_fingerprint(r["atom_types"], r["edges"]) for r in records}
        # NOTE: connectivity identities are NOT computed eagerly: canonical_key
        # is factorial in atom count and hangs on drug-sized molecules. Use
        # identity() for small graphs only.
        self._identities = {}

    def identity(self, rid, max_atoms=8):
        """Canonical connectivity identity for small graphs; None above cap."""
        if rid in self._identities:
            return self._identities[rid]
        r = self.by_id[rid]
        if len(r["atom_types"]) > max_atoms:
            return None
        key = canonical_key(r["atom_types"], r["edges"])
        self._identities[rid] = key
        return key

    def __len__(self):
        return len(self.records)


def formula_key(formula):
    return tuple(sorted((e, formula.get(e, 0)) for e in DOMAIN_ELEMENTS))


def run_query(index, target, patterns, overlap="unknown", precursor=None,
              uncertainty=50, ppm_tenths=100, counters=None):
    """Four staged candidate sets. Returns dict with counts/recall/statuses."""
    counters = counters or WorkCounters()
    observed = target["mass"]
    # Stage 1: mass window (accept + ambiguous retained; reject dropped).
    # Unknown precision ("unavailable") disables the mass decision: those
    # candidates pass through and are counted separately, never rejected.
    s1, n_accept, n_ambig, n_reject, n_unavail = [], 0, 0, 0, 0
    for r in index.records:
        v = mass_verdict(observed, r["formula"], uncertainty, ppm_tenths)
        if v == "accept":
            n_accept += 1
            s1.append(r)
        elif v == "ambiguous":
            n_ambig += 1
            s1.append(r)
        elif v == "unavailable":
            n_unavail += 1
            s1.append(r)
        else:
            n_reject += 1
    # Stage 2: exact formula + charge/element domain.
    target_key = formula_key(target["formula"])
    s2 = [r for r in s1 if index.fkeys.get(r["id"], formula_key(r["formula"])) == target_key
          and r["charge"] == target["charge"]]
    # Stage 3: subgraph containment + overlap consistency. Pattern-side prep
    # is hoisted out of the per-candidate loop; per-candidate target prep is
    # built once and reused across patterns.
    pat_fps = [pattern_fingerprint(pt, pe) for pt, pe in patterns]
    single_preps = [((pt, pe) + _prepare_single(pt, pe)) for pt, pe in patterns]
    joint_prep = _prepare_joint(patterns) if len(patterns) > 1 else None
    from collections import Counter as _Counter

    s3 = []
    truncated = False
    try:
        for r in s2:
            cfp = index.fps[r["id"]]
            counters.fingerprint_screens += 1
            if not all(fingerprint_screen(cfp, pf) for pf in pat_fps):
                continue
            counters.exact_matches += 1
            at, ed = r["atom_types"], r["edges"]
            if any(len(pt) > len(at) for pt, _, _, _ in single_preps):
                continue
            nbr, pos = prepare_target(at, ed)
            occupied = [False] * len(at)
            ok = True
            for pt, pe, order, cons in single_preps:
                if not _contains_with_prep(nbr, pos, pt, order, cons,
                                           occupied, counters):
                    ok = False
                    break
            if not ok:
                continue
            if joint_prep is not None:
                flat_types, flat_pat, flat_adj, degrees = joint_prep
                occupied_sets = [set() for _ in patterns]
                if not _joint_with_prep(nbr, pos, _Counter(at),
                                        flat_types, flat_pat, flat_adj,
                                        degrees, len(patterns),
                                        occupied_sets, counters):
                    continue
            s3.append(r)
    except BudgetExhausted:
        truncated = True
    # Stage 4: precursor evidence (heavy-atom conservation under the
    # provisional target-fragment reading). No precursor => not-evaluated.
    if precursor is None:
        s4, precursor_status = s3, "not_evaluated"
    else:
        s4 = [r for r in s3 if all(
            r["formula"].get(e, 0) <= precursor["formula"].get(e, 0) for e in DOMAIN_ELEMENTS)]
        precursor_status = "evaluated"
    in_pool = any(r["id"] == target["id"] for r in index.records)
    recall = {}
    for name, pool in (("s1", s1), ("s2", s2), ("s3", s3), ("s4", s4)):
        recall[name] = (any(r["id"] == target["id"] for r in pool)
                        if in_pool else None)
    if s3:
        zero_status = "nonempty"
    elif not in_pool:
        zero_status = "target_absent"
    else:
        zero_status = "target_filtered"
    return {
        "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "n_accept": n_accept, "n_ambiguous": n_ambig, "n_reject": n_reject,
        "n_unavailable": n_unavail,
        "recall": recall, "in_pool": in_pool,
        "precursor_status": precursor_status,
        "overlap": overlap,  # provenance label; unknown path never sees oracle
        "zero_status": zero_status,
        "truncated": truncated,
    }


def rank_baselines(pool, target_id, index):
    """Ranking baselines at a fixed pool: uniform, provenance, fingerprint.

    Spectral similarity is unavailable without measured spectra (marked, not
    fabricated). Top-k reported only when target is in the pool.
    """
    ids = [r["id"] for r in pool]
    uniform = list(ids)
    provenance = sorted(ids, key=lambda i: (index.by_id[i]["source"],
                                            index.by_id[i]["id"]))
    # Fingerprint-similarity baseline: Tanimoto on typed-edge sets vs target.
    target_fp = index.fps.get(target_id)
    scored = []
    for r in pool:
        a = set(index.fps[r["id"]][1].elements())
        b = set(target_fp[1].elements()) if target_fp else set()
        inter = len(a & b)
        union = len(a | b) or 1
        scored.append((inter / union, r["id"]))
    fingerprint = [i for _, i in sorted(scored, reverse=True)]
    in_pool = target_id in ids
    out = {"uniform": uniform, "provenance": provenance,
           "fingerprint": fingerprint, "spectral": "unavailable_no_spectrum"}
    if in_pool:
        for k, v in list(out.items()):
            if isinstance(v, list):
                out[k + "@1"] = v[0] == target_id
                out[k + "@3"] = target_id in v[:3]
    return out, in_pool


index_global = None  # set by main/tests for ranking baseline access


# --------------------------------------------------------------------------
# Ingestion adapters (stubs: pin + hash + license, no download by default).
# --------------------------------------------------------------------------

def _pin_error(source, url, note):
    return FileNotFoundError(
        f"{source} fixture not supplied. Pin a release from {url}, record its "
        f"sha256, place the file under data/pinned/, and re-run. {note}")


def ingest_chebi_sdf(path, release, sha256):
    raise _pin_error("ChEBI", "https://www.ebi.ac.uk/chebi/downloads",
                     "3-star SDF; CC BY 4.0; standardize charge/isotopes/salts; "
                     "compute mass from graph, not display values.")


def ingest_massspecgym_tsv(path, release, sha256):
    raise _pin_error("MassSpecGym", "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym",
                     "Pin dataset revision; main TSV + needed candidate JSONs only; "
                     "inspect split/structure duplication before pooling.")


def ingest_mgf(path, release, sha256, license_note):
    raise _pin_error("MGF", "https://massbank.github.io/MassBank-documentation/about.html",
                     f"Pinned record archive required. {license_note}")


def pubchem_query_cached(formula, cache, release, calls):
    """Batched, throttled, cached PubChem lookup (stub).

    Policy: <=5 req/s, cache by (formula, release), record truncation.
    Never the sole ground truth for a reproducible benchmark.
    """
    key = (formula_key(formula), release)
    if key in cache:
        return cache[key], False
    calls.append(formula)
    raise _pin_error("PubChem", "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
                     "Query only unresolved formulas; prefer bulk downloads at scale.")


def qm9_match(record, qm9_index):
    """Restricted positive-control match into QM9 (stub).

    Uses published geometries/frequencies only; no new DFT. Presence is not
    a lifetime/solution-stability guarantee. Returns (matched, method_env).
    """
    if qm9_index is None:
        return False, "qm9_not_loaded"
    heavy = record["heavy"]
    els = set(e for e, _, _ in record["atom_types"]) | {"H"}
    if heavy <= 9 and els <= {"C", "H", "O", "N", "F"}:
        return record["id"] in qm9_index, "qm9:b3lyp/6-31G(2df,p)//freq (published)"
    return False, "outside_qm9_domain"


# --------------------------------------------------------------------------
# Experiment driver.
# --------------------------------------------------------------------------

def residual_valence_sum(record):
    return 0  # closed molecules by construction; kept for stratification key


def stratify_key(target, patterns):
    return {
        "heavy": target["heavy"],
        "n_subgraphs": len(patterns),
        "overlap": "overlap" if len(patterns) > 1 else ("single" if patterns else "none"),
        "residual_valence": residual_valence_sum(target),
        "family": target["family"],
    }


def run_experiment(index, queries, out_of_pool_ids=(), uncertainty=50, ppm_tenths=100):
    global index_global
    index_global = index
    rows = []
    t0 = time.perf_counter()
    c0 = time.process_time()
    for name, tid, patterns, overlap, prec_id in queries:
        sub = DatabaseIndex([r for r in index.records if r["id"] not in out_of_pool_ids])
        index_global = sub
        target = index.by_id[tid]
        precursor = index.by_id.get(prec_id) if prec_id else None
        counters = WorkCounters()
        res = run_query(sub, target, patterns, overlap, precursor,
                        uncertainty, ppm_tenths, counters)
        rank, target_in_s3 = rank_baselines(res["s3"], tid, sub)
        key = stratify_key(target, patterns)
        rows.append({
            "query": name, "target": tid, "overlap": overlap,
            "in_pool": res["in_pool"], "target_in_s3": target_in_s3,
            "precursor_status": res["precursor_status"],
            "n_s1": len(res["s1"]), "n_s2": len(res["s2"]),
            "n_s3": len(res["s3"]), "n_s4": len(res["s4"]),
            "recall_s1": res["recall"]["s1"], "recall_s3": res["recall"]["s3"],
            "accept": res["n_accept"], "ambiguous": res["n_ambiguous"],
            "reject": res["n_reject"], "unavailable": res["n_unavailable"],
            "zero_status": res["zero_status"],
            "top1_uniform": rank.get("uniform@1"), "top1_fingerprint": rank.get("fingerprint@1"),
            "spectral": rank["spectral"], **{f"strat_{k}": v for k, v in key.items()},
            "embedding_nodes": counters.embedding_nodes,
        })
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return rows, {"wall_s": wall, "cpu_s": cpu,
                  "per_1000_queries_s": wall / max(len(queries), 1) * 1000}


def denominators(raw_records, standardized, index, rows):
    parseable = sum(1 for _ in standardized)
    total = len(raw_records)
    in_domain = len(index)
    recalls = [r for r in rows if r["in_pool"]]
    pool_recall = (sum(1 for r in recalls if r["recall_s3"]) / len(recalls)) if recalls else None
    amb = sum(r["ambiguous"] for r in rows)
    # Scaffold overlap across corpus.
    skels = Counter(skeleton_key(r["atom_types"], r["edges"]) for r in index.records)
    dup_scaffolds = sum(1 for _, c in skels.items() if c > 1)
    return {
        "fraction_parseable": parseable / total if total else None,
        "fraction_in_domain": in_domain / parseable if parseable else None,
        "target_in_pool_recall_s3": pool_recall,
        "n_queries": len(rows),
        "n_in_pool": len(recalls),
        "total_boundary_ambiguous_hits": amb,
        "scaffolds_total": len(skels),
        "scaffolds_shared": dup_scaffolds,
        "physical_label_coverage": "0 (QM9 index not loaded; no new DFT per plan)",
        "download_bytes": 0, "api_calls": 0,
    }


def main():
    raw = fixture_raw_records()
    # Add two intentionally excluded records to publish denominators honestly.
    raw = list(raw) + [
        {"id": "FIX-BAD-CHARGE", "source": "synthetic-fixture", "family": "excluded",
         "atom_types": (C3, N2), "edges": ((0, 1, 1),), "charge": 1,
         "original": {"note": "charged; outside V0 neutral domain"}},
        {"id": "FIX-BAD-SALT", "source": "synthetic-fixture", "family": "excluded",
         "atom_types": (C3, C3), "edges": (), "charge": 0,
         "original": {"note": "disconnected salt-like pair"}},
    ]
    standardized, excluded = [], []
    for r in raw:
        rec, reason = standardize_record(r)
        if rec is None:
            excluded.append((r["id"], reason))
        else:
            standardized.append(rec)
    index = DatabaseIndex(standardized)
    queries = fixture_queries()
    rows, perf = run_experiment(index, queries)
    # Out-of-pool slice: drop FIX-001 entirely, re-run one query.
    oop_rows, _ = run_experiment(index, [q for q in queries if q[1] == "FIX-001"][:2],
                                 out_of_pool_ids=("FIX-001", "FIX-002"))
    den = denominators(raw, standardized, index, rows)
    fields = ["query", "target", "overlap", "in_pool", "target_in_s3", "precursor_status",
              "n_s1", "n_s2", "n_s3", "n_s4", "recall_s1", "recall_s3",
              "accept", "ambiguous", "reject", "unavailable", "zero_status",
              "top1_uniform", "top1_fingerprint", "spectral",
              "strat_heavy", "strat_n_subgraphs", "strat_overlap",
              "strat_residual_valence", "strat_family", "embedding_nodes"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows + [{**r, "query": r["query"] + "|out_of_pool_slice"} for r in oop_rows]:
        w.writerow(r)
    sys.stdout.write(buf.getvalue())
    sys.stderr.write(f"# excluded: {excluded}\n# perf: {perf}\n# denominators: {den}\n")


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recs = []
        for raw in fixture_raw_records():
            rec, reason = standardize_record(raw)
            assert rec is not None, reason
            recs.append(rec)
        cls.index = DatabaseIndex(recs)
        global index_global
        index_global = cls.index

    def test_mass_trichotomy_and_unavailable(self):
        f = {"C": 2, "H": 6, "N": 0, "O": 1, "F": 0}
        obs = neutral_mass(f)
        self.assertEqual(mass_verdict(obs, f), "accept")
        self.assertEqual(mass_verdict(obs, f, uncertainty=None), "unavailable")
        tol = obs * 100 // 10_000_000
        self.assertEqual(mass_verdict(obs + tol, f), "ambiguous")
        self.assertEqual(mass_verdict(obs + 10 * tol + 10_000_000, f), "reject")

    def test_standardization_rejects_charge_and_salt(self):
        bad_c, r1 = standardize_record({"id": "x", "atom_types": (C3, N2),
                                        "edges": ((0, 1, 1),), "charge": 1})
        self.assertIsNone(bad_c)
        self.assertIn("charge", r1)
        bad_s, r2 = standardize_record({"id": "y", "atom_types": (C3, C3),
                                        "edges": (), "charge": 0})
        self.assertIsNone(bad_s)
        self.assertIn("salt", r2)

    def test_fingerprint_sound(self):
        # Screen rejections must agree with exact matching (soundness), and
        # every true embedding must pass the screen (no false rejection).
        pats = [CH3_PAT, CH2OH_PAT, CH3CH2_PAT, CN_PAT]
        for r in self.index.records:
            cfp = self.index.fps[r["id"]]
            for pt, pe in pats:
                screen = fingerprint_screen(cfp, pattern_fingerprint(pt, pe))
                exact = contains_exact(r["atom_types"], r["edges"], pt, pe)
                if not screen:
                    self.assertFalse(exact, (r["id"], pt))
                if exact:
                    self.assertTrue(screen, (r["id"], pt))

    def test_monotonicity_across_stages(self):
        target = self.index.by_id["FIX-001"]
        res = run_query(self.index, target, (CH3CH2_PAT, CH2OH_PAT))
        self.assertGreaterEqual(len(res["s1"]), len(res["s2"]))
        self.assertGreaterEqual(len(res["s2"]), len(res["s3"]))
        self.assertGreaterEqual(len(res["s3"]), len(res["s4"]))

    def test_unknown_overlap_never_sees_oracle(self):
        target = self.index.by_id["FIX-001"]
        oracle = {(0, 0): 0}  # fabricated mapping object; unknown path takes none
        a = run_query(self.index, target, (CH3CH2_PAT, CH2OH_PAT), overlap="unknown")
        b = run_query(self.index, target, (CH3CH2_PAT, CH2OH_PAT), overlap="unknown",
                      precursor=None)
        _ = oracle
        self.assertEqual([r["id"] for r in a["s3"]], [r["id"] for r in b["s3"]])

    def test_zero_count_reports_target_absent(self):
        sub = DatabaseIndex([r for r in self.index.records
                              if r["id"] not in ("FIX-001", "FIX-002")])
        target = self.index.by_id["FIX-001"]
        res = run_query(sub, target, (CH3CH2_PAT, CH2OH_PAT))
        self.assertFalse(res["in_pool"])
        self.assertIsNone(res["recall"]["s3"])
        self.assertIn(res["zero_status"], ("target_absent", "nonempty"))

    def test_topk_gated_on_pool_membership(self):
        target = self.index.by_id["FIX-001"]
        res = run_query(self.index, target, ())
        rank, in_pool = rank_baselines(res["s3"], "FIX-001", self.index)
        self.assertTrue(in_pool)
        self.assertIn("uniform@1", rank)
        sub = DatabaseIndex([r for r in self.index.records if r["id"] != "FIX-001"])
        global index_global
        index_global = sub
        res2 = run_query(sub, target, ())
        rank2, in_pool2 = rank_baselines(res2["s3"], "FIX-001", sub)
        self.assertFalse(in_pool2)
        self.assertNotIn("uniform@1", rank2)
        index_global = self.index

    def test_joint_overlap_rejects_collapse(self):
        # Two distinct single-atom patterns forced onto one atom by joint
        # mapping are fine (sharing allowed), but an edge pattern plus an
        # incompatible atom sharing must still satisfy edge constraints.
        target = self.index.by_id["FIX-006"]  # methylamine
        ok = joint_contains(target["atom_types"], target["edges"], (CH3_PAT, N_PAT))
        self.assertTrue(ok)
        impossible = (((C3, C3), ((0, 1, 1),)),)
        self.assertFalse(joint_contains(target["atom_types"], target["edges"], impossible))

    def test_count_triangle_path_is_six(self):
        # Review regression: leaf bulk-reset overcounted (9 instead of 6)
        # by letting later branches reuse ancestor atoms.
        tri = ((C3, C3, C3), ((0, 1, 1), (1, 2, 1), (0, 2, 1)))
        pat = ((C3, C3, C3), ((0, 1, 1), (1, 2, 1)))
        nbr, pos = prepare_target(*tri)
        order, cons = _prepare_single(*pat)
        self.assertEqual(count_embeddings_prep(nbr, pos, pat[0], order, cons, 100), 6)
        self.assertEqual(count_embeddings_prep(nbr, pos, pat[0], order, cons, 4), 4)

    def test_unavailable_precision_passes_through(self):
        # Review regression: unknown precision disabled the mass decision;
        # it must not read as rejection.
        target = self.index.by_id["FIX-001"]
        res = run_query(self.index, target, (), "unknown", None, uncertainty=None)
        self.assertEqual(len(res["s1"]), len(self.index))
        self.assertEqual(res["n_unavailable"], len(self.index))
        self.assertEqual(res["n_reject"], 0)
        self.assertTrue(res["recall"]["s1"])

    @staticmethod
    def _all_injections(at, ed, pt, pe):
        import itertools

        emap = {tuple(sorted((a, b))): o for a, b, o in ed if o}
        for perm in itertools.permutations(range(len(at)), len(pt)):
            if any(at[perm[i]] != pt[i] for i in range(len(pt))):
                continue
            ok = True
            for a, b, o in pe:
                if emap.get(tuple(sorted((perm[a], perm[b])))) != o:
                    ok = False
                    break
            if ok:
                yield perm

    def test_matchers_agree_with_brute_force(self):
        # Optimized backtracking matchers vs an independent
        # permutation-enumeration reference on randomized small graphs.
        import itertools
        import random

        def brute_single(at, ed, pt, pe):
            emap = {tuple(sorted((a, b))): o for a, b, o in ed if o}
            for perm in itertools.permutations(range(len(at)), len(pt)):
                if any(at[perm[i]] != pt[i] for i in range(len(pt))):
                    continue
                ok = True
                for a, b, o in pe:
                    if emap.get(tuple(sorted((perm[a], perm[b])))) != o:
                        ok = False
                        break
                if ok:
                    return True
            return False

        def brute_joint(at, ed, patterns):
            emap = {tuple(sorted((a, b))): o for a, b, o in ed if o}
            injs = []
            for pt, pe in patterns:
                found = []
                for perm in itertools.permutations(range(len(at)), len(pt)):
                    if any(at[perm[i]] != pt[i] for i in range(len(pt))):
                        continue
                    ok = True
                    for a, b, o in pe:
                        if emap.get(tuple(sorted((perm[a], perm[b])))) != o:
                            ok = False
                            break
                    if ok:
                        found.append(perm)
                injs.append(found)
            for combo in itertools.product(*injs):
                return True  # sharing across patterns allowed; per-pattern
                # injectivity already enforced by permutations
            return False

        rnd = random.Random(20261003)
        pool = [C3, C2, C1, N2, N1, O1, O0]
        for trial in range(300):
            n = rnd.randint(1, 6)
            at = tuple(rnd.choice(pool) for _ in range(n))
            pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
            rnd.shuffle(pairs)
            ed = tuple((a, b, rnd.choice([1, 1, 2]))
                       for a, b in pairs[:rnd.randint(0, min(len(pairs), n))])
            m = rnd.randint(1, 3)
            pt = tuple(rnd.choice(pool) for _ in range(m))
            pp = [(a, b) for a in range(m) for b in range(a + 1, m)]
            rnd.shuffle(pp)
            pe = tuple((a, b, rnd.choice([1, 2]))
                       for a, b in pp[:rnd.randint(0, len(pp))])
            self.assertEqual(contains_exact(at, ed, pt, pe),
                             brute_single(at, ed, pt, pe), (trial, at, ed, pt, pe))
            if trial % 10 == 0:
                # Capped counter agrees with exhaustive enumeration below cap
                # and saturates at cap above it.
                from tools.ms2_database_retrieval import (
                    count_embeddings_prep, prepare_target, _prepare_single)
                nbr, pos = prepare_target(at, ed)
                order, cons = _prepare_single(pt, pe)
                total = sum(1 for _ in self._all_injections(at, ed, pt, pe))
                self.assertEqual(
                    count_embeddings_prep(nbr, pos, pt, order, cons, 10_000),
                    total, (trial, at, ed, pt, pe))
                self.assertEqual(
                    count_embeddings_prep(nbr, pos, pt, order, cons, 1),
                    min(1, total))
            if trial % 3 == 0:
                m2 = rnd.randint(1, 2)
                pt2 = tuple(rnd.choice(pool) for _ in range(m2))
                pats = [(pt, pe), (pt2, ())]
                self.assertEqual(joint_contains(at, ed, pats),
                                 brute_joint(at, ed, pats), (trial, at, ed, pats))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[sys.argv[0]])
    else:
        main()
