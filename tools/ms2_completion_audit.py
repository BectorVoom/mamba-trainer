#!/usr/bin/env python3
"""Bounded molecular-completion ambiguity audit (independent Python mirror).

Run:      python3 tools/ms2_completion_audit.py
Fixtures: experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json

This mirrors the bounded Rust audit in `src/models/ms2/completion.rs`
semantics-for-semantics but uses its own enumeration (type-multiset +
edge-assignment) and its own identity (permutation canonical key). Both
sides must produce the same per-fixture status, mass status, accepted
formula set, unique-graph count, recovery decision and zero-certification
bit; work counters intentionally differ (different enumeration strategies)
so they are not part of parity. The older C/O pilot
(tools/ms2_completion_ambiguity.py) is untouched and remains a third,
C/O-only independent check on its own fixed queries.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from itertools import combinations_with_replacement, permutations
import hashlib
import json
from pathlib import Path
import sys

MAX_TRACE_ATOMS = 6
EXACT = {"C": "12", "H": "1.00782503223", "N": "14.00307400443",
         "O": "15.99491461957"}
SCALE = 1_000_000
MASS = {e: int((Decimal(m) * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))
        for e, m in EXACT.items()}
ERROR_NDA = {
    e: int((abs(Decimal(m) * SCALE - MASS[e]) * 1000).to_integral_value(rounding=ROUND_CEILING))
    for e, m in EXACT.items()
}
# Element, parent-relative hydrogen count, valence.
ALL_TYPES = (("C", 0, 4), ("C", 1, 4), ("C", 2, 4), ("C", 3, 4),
             ("N", 0, 3), ("N", 1, 3), ("N", 2, 3),
             ("O", 0, 2), ("O", 1, 2))
TYPE_INDEX = {t: i + 1 for i, t in enumerate(ALL_TYPES)}
ELEM_OF = {t[0]: idx for idx, t in enumerate(("C", "H", "N", "O"))}


class Verdicts:
    ACCEPT = "accept"
    REJECT = "reject"
    AMBIG = "ambiguous"


def formula_mass(c):
    return sum(c[e] * MASS[e] for e in EXACT)


def verdict(observed, comp, uncertainty, ppm_tenths):
    if uncertainty is None:
        return "unavailable"
    error = (sum(comp[e] * ERROR_NDA[e] for e in EXACT) + 999) // 1000 + uncertainty
    tol = observed * ppm_tenths // 10_000_000
    r = abs(observed - formula_mass(comp))
    if r + error <= tol:
        return Verdicts.ACCEPT
    if r > tol + error:
        return Verdicts.REJECT
    return Verdicts.AMBIG


def atom_formula(types):
    out = {"C": 0, "H": 0, "N": 0, "O": 0}
    for element, hydrogens, _ in types:
        out[element] += 1
        out["H"] += hydrogens
    return out


def ring_count(n, bonds):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    for a, b, o in bonds:
        if o == 0:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            components -= 1
    return len([(a, b, o) for a, b, o in bonds if o]) - n + components


def connected(n, bonds):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    for a, b, o in bonds:
        if o == 0:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            components -= 1
    return components == 1


def canonical(types, bonds):
    n = len(types)
    edge_map = {}
    for a, b, o in bonds:
        if o:
            edge_map[tuple(sorted((a, b)))] = o
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best = None
    for perm in permutations(range(n)):
        labels = tuple(types[i] for i in perm)
        orders = tuple(edge_map.get(tuple(sorted((perm[i], perm[j]))), 0) for i, j in pairs)
        key = (labels, orders)
        if best is None or key < best:
            best = key
    return best


def contains_phantom(types, bonds, pattern):
    p_types, p_bonds = pattern
    if len(p_types) > len(types):
        return False
    pset = {}
    for a, b, o in bonds:
        if o:
            pset[tuple(sorted((a, b)))] = o
    order = list(range(len(p_types)))
    chosen = [None] * len(p_types)

    def go(depth, used):
        if depth == len(p_types):
            return True
        for cand in range(len(types)):
            if cand in used or types[cand] != p_types[depth]:
                continue
            ok = True
            for a, b, o in p_bonds:
                if a == depth and b < depth:
                    other = b
                elif b == depth and a < depth:
                    other = a
                else:
                    continue
                if pset.get(tuple(sorted((cand, chosen[other]))), 0) != o:
                    ok = False
                    break
            if ok:
                chosen[depth] = cand
                used.add(cand)
                if go(depth + 1, used):
                    return True
                used.discard(cand)
        return False

    return go(0, set())


def matches_all(types, bonds, patterns, correspondence):
    for pat in patterns:
        if not contains_phantom(types, bonds, (pat[0], pat[1])):
            return False
    if correspondence is not None:
        embeds = []
        for pat in patterns:
            maps = []
            for cand_map in all_embeddings(types, bonds, pat[0], pat[1]):
                maps.append(cand_map)
            if not maps:
                return False
            embeds.append(maps)
        # equivalence classes over atom occurrences (union-find)
        parents = {}

        def find(key):
            root = key
            while parents[root] != root:
                root = parents[root]
            while parents[key] != root:
                parents[key], key = root, parents[key]
            return root

        for s in range(len(patterns)):
            for a in range(len(patterns[s][0])):
                parents.setdefault((s, a), (s, a))
        for (s1, a1), (s2, a2) in correspondence:
            r1, r2 = find((s1, a1)), find((s2, a2))
            if r1 != r2:
                parents[r1] = r2
        for combo in _product(embeds):
            ok = True
            for (s1, a1), (s2, a2) in correspondence:
                if combo[s1][a1] != combo[s2][a2]:
                    ok = False
                    break
            if not ok:
                continue
            # fully-known semantics: each target atom belongs to exactly one
            # class (distinct classes cannot share a target atom).
            target_class = {}
            bad = False
            for s, m in enumerate(combo):
                for a, t in enumerate(m):
                    r = find((s, a))
                    prev = target_class.setdefault(t, r)
                    if prev != r:
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                return True
        return False
    return True


def _product(lists):
    if not lists:
        yield ()
    else:
        firsts, rest = lists[0], lists[1:]
        for f in firsts:
            for r in _product(rest):
                yield (f,) + r


def all_embeddings(types, bonds, p_types, p_bonds):
    pset = {}
    for a, b, o in bonds:
        if o:
            pset[tuple(sorted((a, b)))] = o
    chosen = [None] * len(p_types)

    def go(depth, used):
        if depth == len(p_types):
            yield tuple(chosen)
            return
        for cand in range(len(types)):
            if cand in used or types[cand] != p_types[depth]:
                continue
            ok = True
            for a, b, o in p_bonds:
                if a == depth and b < depth:
                    other = b
                elif b == depth and a < depth:
                    other = a
                else:
                    continue
                if pset.get(tuple(sorted((cand, chosen[other]))), 0) != o:
                    ok = False
                    break
            if ok:
                chosen[depth] = cand
                used.add(cand)
                yield from go(depth + 1, used)
                used.discard(cand)

    yield from go(0, set())


def parse_pattern(obj):
    types = tuple(ALL_TYPES[i - 1] for i in obj["atoms"])
    bonds = tuple(tuple(b) for b in obj["bonds"])
    return types, bonds


def enumerate_graphs(formula, domain, patterns, correspondence):
    n = formula["C"] + formula["N"] + formula["O"]
    if n < domain[0] or n > domain[1]:
        return []
    types_pool = []
    for t in ALL_TYPES:
        if t[0] in domain[2]:
            types_pool.append(t)
    out = set()
    for mult in combinations_with_replacement(types_pool, n):
        if atom_formula(mult) != formula:
            continue
        caps = [t[2] - t[1] for t in mult]
        if sum(caps) % 2 != 0:
            continue
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        orders = [0] * len(pairs)
        remaining = list(caps)

        def edges_at(idx):
            if idx == len(pairs):
                if any(rem != 0 for rem in remaining):
                    return
                bonds = [(a, b, orders[i]) for i, (a, b) in enumerate(pairs)]
                positive = [(a, b, o) for (a, b, o) in bonds if o]
                if not connected(n, bonds):
                    return
                if ring_count(n, bonds) > domain[3]:
                    return
                if matches_all(mult, bonds, patterns, correspondence):
                    out.add(canonical(mult, bonds))
                return
            a, b = pairs[idx]
            for o in range(min(3, remaining[a], remaining[b]) + 1):
                orders[idx] = o
                remaining[a] -= o
                remaining[b] -= o
                edges_at(idx + 1)
                remaining[a] += o
                remaining[b] += o

        edges_at(0)
    return sorted(out)


def _part(symbol, count):
    if not count:
        return ""
    return symbol if count == 1 else f"{symbol}{count}"


def formula_text(comp):
    return (_part("C", comp.get("C", 0)) + _part("H", comp.get("H", 0))
            + _part("N", comp.get("N", 0)) + _part("O", comp.get("O", 0)))


def run_fixture(fx):
    mass = fx["mass"]
    domain_tuple = (fx["domain"]["min_heavy"], fx["domain"]["max_heavy"],
                    fx["domain"]["elements"], fx["domain"]["max_ring_closures"])
    obs = fx["observed_mass_uda"]
    ppm = mass["ppm_tenths"]
    unc = mass["uncertainty_uda"]

    # validate correspondence (mirror of Rust Correspondence::validate)
    corr = fx.get("correspondence")
    patterns = [parse_pattern(p) for p in fx.get("patterns", [])]
    if corr is not None:
        ok = True
        for (s1, a1), (s2, a2) in corr:
            if s1 >= len(patterns) or s2 >= len(patterns):
                ok = False
                break
            if a1 >= len(patterns[s1][0]) or a2 >= len(patterns[s2][0]):
                ok = False
                break
            if patterns[s1][0][a1] != patterns[s2][0][a2]:
                ok = False
                break
        # union-find collapse check mirrors Rust Correspondence::validate
        parent = {}
        for s, p in enumerate(patterns):
            for a in range(len(p[0])):
                parent[(s, a)] = (s, a)

        def findc(key):
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != root:
                parent[key], key = root, parent[key]
            return root

        for (s1, a1), (s2, a2) in corr:
            r1, r2 = findc((s1, a1)), findc((s2, a2))
            if r1 != r2:
                parent[r1] = r2
        by_root = {}
        for s, p in enumerate(patterns):
            for a in range(len(p[0])):
                by_root.setdefault(findc((s, a)), set()).add((s, a))
        for members in by_root.values():
            per_sub = {}
            for s, a in members:
                per_sub.setdefault(s, set()).add(a)
            if any(len(v) > 1 for v in per_sub.values()):
                ok = False
                break
        if not ok:
            return {"status": "unsupported_input", "mass_status": "not_evaluated",
                    "unique_graphs": 0, "accepted_formulas": [],
                    "ambiguous_formulas": [], "certifies_zero": False,
                    "recovery": None, "termination_reasons": []}

    accepted = []
    ambiguous = []
    accepted_any = False
    ambiguous_any = False
    for c in range(7):
        for n in range(7):
            for o in range(7):
                heavy = c + n + o
                if heavy < fx["domain"]["min_heavy"] or heavy > fx["domain"]["max_heavy"]:
                    continue
                allowed = set(fx["domain"]["elements"])
                if c and "C" not in allowed or n and "N" not in allowed or o and "O" not in allowed:
                    continue
                h = n % 2
                while h <= 2 * c + n + 2:
                    comp = {"C": c, "H": h, "N": n, "O": o}
                    if unc is not None:
                        v = verdict(obs, comp, unc, ppm)
                        if v == Verdicts.ACCEPT:
                            accepted.append(comp)
                            accepted_any = True
                        elif v == Verdicts.AMBIG:
                            ambiguous.append(comp)
                            ambiguous_any = True
                    h += 2

    if unc is None:
        return {"status": "mass_evidence_unresolved", "mass_status": "unavailable",
                "unique_graphs": 0, "accepted_formulas": [],
                "ambiguous_formulas": [], "certifies_zero": False,
                "recovery": None, "termination_reasons": []}
    mstatus = "ambiguous" if ambiguous_any else ("accepted" if accepted_any else "rejected")

    graphs = set()
    for comp in accepted:
        for key in enumerate_graphs(comp, domain_tuple[0:1] + (domain_tuple[1],) + tuple(domain_tuple[2:]), patterns, corr):
            graphs.add(key)

    # reference canonical membership
    recovery = None
    if "reference" in fx and fx["reference"]:
        ref = fx["reference"]
        rt = tuple(ALL_TYPES[i - 1] for i in ref["atoms"])
        rb = tuple(tuple(b) for b in ref["bonds"])
        recovery = canonical(rt, rb) in graphs

    allowed_zero = True
    if ambiguous_any:
        status = "mass_evidence_unresolved"
        mstatus = "ambiguous"
        allowed_zero = False
    else:
        status = "complete"
        mstatus = "ambiguous" if ambiguous_any else ("accepted" if accepted_any else "rejected")
    return {"status": status, "mass_status": mstatus,
            "unique_graphs": len(graphs),
            "accepted_formulas": [formula_text(c) for c in accepted],
            "ambiguous_formulas": [formula_text(c) for c in ambiguous],
            "certifies_zero": allowed_zero and not graphs,
            "recovery": recovery, "termination_reasons": [],
            "identity_keys": sorted(graphs, key=repr)}


def main():
    root = Path(__file__).resolve().parent.parent
    fx_path = root / "experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json"
    doc = json.loads(fx_path.read_bytes().decode())
    rows = []
    ok = True
    for fx in doc["fixtures"]:
        if fx.get("full_bf_out_of_scope"):
            rows.append({"name": fx["name"], "got": {"status": "bf_out_of_scope_python_mirror", "mass_status": None,
                                                      "unique_graphs": None, "accepted_formulas": [], "ambiguous_formulas": [],
                                                      "certifies_zero": False, "recovery": None, "termination_reasons": []},
                        "expected": fx["expected"], "match": None, "note": "independent brute-force mirror limited; Rust authoritative"})
            continue
        got = run_fixture(fx)
        # Identity-set check when the fixture pins it.
        graphs_match = True
        if fx.get("expected_graphs"):
            mine = set()
            for g in fx["expected_graphs"]:
                rt = tuple(ALL_TYPES[i - 1] for i in g["atoms"])
                rb = tuple(tuple(b) for b in g["bonds"])
                mine.add(canonical(rt, rb))
            keys_got = got.get("identity_keys")
            graphs_match = keys_got is not None and set(keys_got) == mine
        exp = fx["expected"]
        match = graphs_match and (got["status"] == exp["status"] and got["mass_status"] == exp["mass_status"]
                 and got["unique_graphs"] == exp["unique_graphs"]
                 and got["accepted_formulas"] == exp["accepted_formulas"]
                 and got["ambiguous_formulas"] == exp["ambiguous_formulas"]
                 and got["certifies_zero"] == exp["certifies_zero"]
                 and (exp.get("recovery") is None or got["recovery"] == exp["recovery"]))
        rows.append({"name": fx["name"], "got": got, "expected": exp, "match": match})
        if match is not None:
            ok = ok and match
    out = {
        "protocol": "completion-bounded-v1",
        "python_mirror": True,
        "fixtures_sha256": hashlib.sha256(fx_path.read_bytes()).hexdigest(),
        "ok": ok,
        "fixtures": rows,
    }
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
