"""Bounded, independent molecular-completion ambiguity pilot.

Run: python3 tools/ms2_completion_ambiguity.py
Test: python3 -m unittest tools/ms2_completion_ambiguity.py

This standalone reference intentionally uses only Python's standard library so the
pilot is runnable on main before the in-progress MS2 Rust work is committed.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from itertools import combinations, combinations_with_replacement, permutations
import unittest

EXACT = {"C": "12", "H": "1.00782503223", "O": "15.99491461957"}
SCALE = 1_000_000
MASS = {e: int((Decimal(m) * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))
        for e, m in EXACT.items()}
ERROR_NDA = {
    e: int((abs(Decimal(m) * SCALE - MASS[e]) * 1000).to_integral_value(rounding=ROUND_CEILING))
    for e, m in EXACT.items()
}
# Element, parent-relative hydrogen count, total valence (V0 neutral C/O types).
TYPES = (("C", 0, 4), ("C", 1, 4), ("C", 2, 4), ("C", 3, 4),
         ("O", 0, 2), ("O", 1, 2))
MAX_ATOMS = 4
LIMIT = 100_000


class BudgetExceeded(Exception):
    pass


class Counters:
    def __init__(self):
        self.formula_visits = 0
        self.edge_assignments = 0
        self.embedding_nodes = 0
        self.canonical_permutations = 0

    def count(self, field):
        value = getattr(self, field) + 1
        setattr(self, field, value)
        if value > LIMIT:
            raise BudgetExceeded(field)


def mass(formula):
    return sum(formula.get(e, 0) * MASS[e] for e in EXACT)


def verdict(observed, formula, uncertainty=50, ppm_tenths=100):
    """Three-valued neutral mass decision with integer arithmetic bounds."""
    if uncertainty is None:
        return "unavailable"
    computed = mass(formula)
    error = (sum(formula.get(e, 0) * ERROR_NDA[e] for e in EXACT) + 999) // 1000
    error += uncertainty
    tolerance = observed * ppm_tenths // 10_000_000
    residual = abs(observed - computed)
    if residual + error <= tolerance:
        return "accept"
    if residual > tolerance + error:
        return "reject"
    return "ambiguous"


def formulas(observed, counters):
    accepted = []
    unresolved = 0
    for c in range(MAX_ATOMS + 1):
        for o in range(MAX_ATOMS - c + 1):
            if not 2 <= c + o <= MAX_ATOMS:
                continue
            for h in range(4 * MAX_ATOMS + 1):
                counters.count("formula_visits")
                candidate = {"C": c, "H": h, "O": o}
                decision = verdict(observed, candidate)
                if decision == "accept":
                    accepted.append(candidate)
                elif decision == "ambiguous":
                    unresolved += 1
    return accepted, unresolved


def atom_formula(atom_types):
    result = {"C": 0, "H": 0, "O": 0}
    for element, hydrogen, _ in atom_types:
        result[element] += 1
        result["H"] += hydrogen
    return result


def connected(n, edges):
    seen = {0}
    changed = True
    while changed:
        changed = False
        for a, b, order in edges:
            if not order:
                continue
            if (a in seen) != (b in seen):
                seen.update((a, b))
                changed = True
    return len(seen) == n


def canonical(atom_types, edges, counters):
    n = len(atom_types)
    edge_map = {(a, b): order for a, b, order in edges if order}
    best = None
    for order in permutations(range(n)):
        counters.count("canonical_permutations")
        labels = tuple(atom_types[i] for i in order)
        bonds = tuple(edge_map.get(tuple(sorted((order[i], order[j]))), 0)
                      for i, j in combinations(range(n), 2))
        key = (labels, bonds)
        if best is None or key < best:
            best = key
    return best


def contains(atom_types, edges, pattern_types, pattern_edges, counters):
    """Injective, typed edge-preserving embedding; target may have extra edges."""
    if len(pattern_types) > len(atom_types):
        return False
    edge_map = {(a, b): order for a, b, order in edges if order}
    chosen = []

    def search(depth):
        if depth == len(pattern_types):
            return True
        for candidate, atom in enumerate(atom_types):
            counters.count("embedding_nodes")
            if atom != pattern_types[depth] or candidate in chosen:
                continue
            valid = True
            for a, b, order in pattern_edges:
                previous = b if a == depth and b < depth else a if b == depth and a < depth else None
                if previous is None:
                    continue
                pair = tuple(sorted((candidate, chosen[previous])))
                if edge_map.get(pair) != order:
                    valid = False
                    break
            if valid:
                chosen.append(candidate)
                if search(depth + 1):
                    return True
                chosen.pop()
        return False

    return search(0)


def generate(formula, patterns, identities, counters):
    n = formula["C"] + formula["O"]
    pairs = tuple(combinations(range(n), 2))
    for atom_types in combinations_with_replacement(TYPES, n):
        if atom_formula(atom_types) != formula:
            continue
        capacities = tuple(valence - hydrogen for _, hydrogen, valence in atom_types)
        if sum(capacities) % 2:
            continue
        edge_orders = [0] * len(pairs)
        remaining = list(capacities)

        def edges_at(index, nonzero):
            if index == len(pairs):
                if any(remaining) or nonzero < n - 1 or nonzero > n:
                    return
                edges = tuple((a, b, edge_orders[k]) for k, (a, b) in enumerate(pairs)
                              if edge_orders[k])
                if not connected(n, edges):
                    return
                if all(contains(atom_types, edges, *pattern, counters) for pattern in patterns):
                    identities.add(canonical(atom_types, edges, counters))
                    if len(identities) > LIMIT:
                        raise BudgetExceeded("unique graphs")
                return
            a, b = pairs[index]
            for bond in range(min(3, remaining[a], remaining[b]) + 1):
                counters.count("edge_assignments")
                edge_orders[index] = bond
                remaining[a] -= bond
                remaining[b] -= bond
                edges_at(index + 1, nonzero + bool(bond))
                remaining[a] += bond
                remaining[b] += bond

        edges_at(0, 0)


def query(observed, patterns=()):
    counters = Counters()
    identities = set()
    status = "complete"
    accepted = []
    unresolved = 0
    try:
        accepted, unresolved = formulas(observed, counters)
        if unresolved:
            status = "mass_evidence_unresolved"
        for candidate in accepted:
            generate(candidate, patterns, identities, counters)
    except BudgetExceeded:
        status = "search_budget_exhausted"
    return status, len(accepted), len(identities), counters


CH3 = (TYPES[3],)
CH2_OH = ((TYPES[2], TYPES[5]), ((0, 1, 1),))
CH3_CH2 = ((TYPES[3], TYPES[2]), ((0, 1, 1),))
QUERIES = (
    ("C2H6O_mass_only", {"C": 2, "H": 6, "O": 1}, ()),
    ("C2H6O_CH3", {"C": 2, "H": 6, "O": 1}, ((CH3, ()),)),
    ("C2H6O_CH2OH", {"C": 2, "H": 6, "O": 1}, (CH2_OH,)),
    ("C2H6O_two_overlapping", {"C": 2, "H": 6, "O": 1}, (CH3_CH2, CH2_OH)),
    ("C3H8O_mass_only", {"C": 3, "H": 8, "O": 1}, ()),
)


class PilotTests(unittest.TestCase):
    def test_known_constitutional_isomers(self):
        for _, target, patterns, expected in (
            (*QUERIES[0], 2), (*QUERIES[2], 1), (*QUERIES[3], 1),
            (*QUERIES[4], 3),
        ):
            status, formula_count, graph_count, _ = query(mass(target), patterns)
            self.assertEqual((status, formula_count, graph_count),
                             ("complete", 1, expected))

    def test_mass_boundary_and_unknown_precision(self):
        f = {"C": 2, "H": 6, "O": 1}
        observed = mass(f)
        self.assertEqual(verdict(observed, f), "accept")
        self.assertEqual(verdict(observed, f, uncertainty=None), "unavailable")
        tolerance = observed * 100 // 10_000_000
        self.assertEqual(verdict(observed + tolerance, f), "ambiguous")

    def test_incompatible_open_pattern(self):
        target = {"C": 2, "H": 6, "O": 1}
        impossible = ((TYPES[0], TYPES[5]), ((0, 1, 1),))
        status, _, graphs, _ = query(mass(target), (impossible,))
        self.assertEqual((status, graphs), ("complete", 0))


if __name__ == "__main__":
    print("query,status,formulas,graphs,formula_visits,edge_assignments,embedding_nodes,canonical_permutations")
    for name, target, patterns in QUERIES:
        status, formula_count, graph_count, counters = query(mass(target), patterns)
        print(f"{name},{status},{formula_count},{graph_count},"
              f"{counters.formula_visits},{counters.edge_assignments},"
              f"{counters.embedding_nodes},{counters.canonical_permutations}")
