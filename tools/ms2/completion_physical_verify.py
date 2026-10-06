"""Optional geometry-and-strain screen for molecular-completion candidates.

Protocol ``physical-verification-v1``. A bounded, auditable check at
force-field level: structural alerts on the typed graph, one 3D build and
relaxation per constitution (or per requested stereoisomer), and geometry
diagnostics on the kept conformer, including re-perception of every
requested stereo element from the final coordinates.

This is evidence at force-field level only. It is not thermodynamic
stability, kinetic persistence or synthesizability, and the output carries
fixed statements that make that impossible to misread. Candidates are never
reordered, filtered or re-ranked here, and never labelled physically
verified by the generator: the only success status is
``force_field_optimization_converged``, which means the bounded procedure
ran to completion, not that the molecule is stable.

Pipeline per molecule (constitution, or one stereoisomer), in order:

1. Structural alerts on the typed graph, before RDKit sees it (pure Python
   over atom type ids and bonds; each alert has an id, the atom indices
   involved, the rule's parameters, and a one-line meaning). Every alert
   is a warning about strain or reactivity, never a verdict. Alerts are
   computed first and reported with every status, including converged
   optimisations. The single exception is ``bridgehead_double_bond``,
   which needs ring membership from the smallest set of smallest rings
   and therefore reads SSSR ring info off a stereo-free RDKit copy of the
   molecule (stated in the alert's parameters); it still runs before any
   embedding or optimisation.
2. Conversion with ``completion_stereo_check.to_rdkit`` (the one typed
   graph to RDKit conversion; hydrogen counts fixed, no hydrogen
   invention), with the stereo assignment applied when one is given.
3. Embedding with explicit hydrogens (ETKDGv3, fixed seed), then a single
   retry with random coordinates when nothing embeds. The record counts
   requested conformers (``conformers_requested``), not internal ETKDG
   iterations.
4. Optimisation with MMFF94 when it has parameters for the whole
   molecule, else UFF, else ``unsupported``. The lowest-energy converged
   conformer is kept.
5. Geometry diagnostics on the kept conformer, each recorded with its
   numbers: finite coordinates, bond lengths against covalent radii,
   close contacts (only pairs with a shortest path of 4+ bonds; 1-2, 1-3
   and 1-4 pairs are excluded), angle deviations at sp3/sp2 carbons, and
   re-perception of every assigned stereo element from coordinates.
   ``coordinates_not_finite`` and ``stereo_not_preserved`` fail the
   molecule (``calculation_failed``); bond-length, contact and angle
   outliers and undetermined stereo are recorded loudly but do not fail
   it. This split is a decision of this tool, stated here and in the
   output.

Requires RDKit (tested with 2026.03.3). No quantum-chemistry program is
used or needed. Set the seed for determinism (default 20261005); the
optimiser runs single-threaded.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from completion_stereo_check import (
    LostAssignmentError,
    candidate_block_to_rdkit_block,
    to_rdkit,
)
from ms2_reference import ATOM_TYPES

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import rdkit

RDLogger.DisableLog("rdApp.*")

PROTOCOL = "physical-verification-v1"
DEFAULT_SEED = 20261005

STABILITY = {
    "status": "not_evaluated",
    "reason": (
        "a converged force-field optimisation is not evidence of "
        "thermodynamic stability, kinetic persistence or synthesizability"
    ),
}
ELECTRONIC_STRUCTURE = {
    "status": "not_evaluated",
    "reason": "no quantum-chemistry program is installed",
}
ENERGY_COMPARABILITY = (
    "force-field energies compare conformers of one molecule under one "
    "method only; they do not rank different molecules"
)

CONVERGED = "force_field_optimization_converged"
FAILED = "calculation_failed"
UNSUPPORTED = "unsupported"
ERROR = "error"

HALOGENS = {"F", "Cl", "Br", "I"}
# Lowest routine valence per element for the hypervalency flag: sulphur is
# routinely divalent, phosphorus trivalent; anything above is expanded
# valence that a force field may describe poorly.
BASE_VALENCE = {"S": 2, "P": 3}

# Small-ring cutoffs (all in atoms).
TRANS_DOUBLE_MAX_RING = 8  # trans double bond flagged in rings below this size
TRIPLE_MAX_RING = 8  # triple bond flagged in rings below this size
CUMULENE_MAX_RING = 9  # C=C=C-type atom flagged in rings below this size
BRIDGEHEAD_MAX_RING = 8  # bridgehead double bond flagged when the largest
# SSSR ring holding the bond is below this size
SMALL_RING = 4  # fused/cage/three-membered analysis uses rings of this size or less

BOND_RATIO_LOW, BOND_RATIO_HIGH = 0.8, 1.25
CONTACT_FACTOR = 0.7
ANGLE_OUTLIER_DEG = 25.0
SP3_IDEAL_DEG = 109.5
SP2_IDEAL_DEG = 120.0
# Near-degenerate stereo perception: a tetrahedral signed volume under this
# (cubic angstrom) carries no readable assignment, as does a double-bond
# dihedral within this margin (degrees) of 90.
VOLUME_TOLERANCE_A3 = 0.05
DIHEDRAL_MARGIN_DEG = 10.0


@dataclass
class VerifyConfig:
    conformers: int = 10
    max_iters: int = 2000
    seed: int = DEFAULT_SEED
    timeout_s: float = 60.0
    num_threads: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "VerifyConfig":
        return cls(
            conformers=int(d.get("conformers", 10)),
            max_iters=int(d.get("max_iters", 2000)),
            seed=int(d.get("seed", DEFAULT_SEED)),
            timeout_s=float(d.get("timeout_s", 60.0)),
            num_threads=int(d.get("num_threads", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "conformers": self.conformers,
            "max_iters": self.max_iters,
            "seed": self.seed,
            "timeout_s": self.timeout_s,
            "num_threads": self.num_threads,
        }


# ---------------------------------------------------------------------------
# Pure-Python graph helpers (no RDKit; used by the structural alerts).
# ---------------------------------------------------------------------------


def _adjacency(n: int, bonds: list) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b, _ in bonds:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _bond_orders(bonds: list) -> dict[tuple[int, int], int]:
    return {(min(a, b), max(a, b)): o for a, b, o in bonds}


def _elements(atoms: list[int]) -> list[str]:
    return [ATOM_TYPES[t][0] for t in atoms]


def _hydrogens(atoms: list[int]) -> list[int]:
    return [ATOM_TYPES[t][1] for t in atoms]


def _bfs_length(adj: dict[int, set[int]], src: int, dst: int,
               skip: tuple[int, int] | None = None) -> int | None:
    """Shortest-path length in bonds, optionally ignoring one edge."""
    if src == dst:
        return 0
    seen = {src}
    frontier = [src]
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if skip is not None and {u, v} == {skip[0], skip[1]}:
                    continue
                if v == dst:
                    return depth
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return None


def smallest_ring_through_bond(adj: dict[int, set[int]], a: int, b: int) -> int | None:
    """Smallest ring holding bond (a, b), in atoms (None when acyclic there)."""
    d = _bfs_length(adj, a, b, skip=(a, b))
    return None if d is None else d + 1


def _find_triangles(adj: dict[int, set[int]], n: int) -> set[tuple[int, int, int]]:
    tris = set()
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[a] & adj[b]:
                if c > b:
                    tris.add((a, b, c))
    return tris


def _find_squares(adj: dict[int, set[int]], n: int) -> set[frozenset]:
    """Chordless four-cycles as frozensets of normalised edges."""
    sqs = set()
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            for x in adj[a]:
                if x == b or x in adj[b]:
                    continue
                for y in adj[b]:
                    if y == a or y == x or y in adj[a]:
                        continue
                    if y in adj[x]:
                        sqs.add(
                            frozenset(
                                (min(p, q), max(p, q))
                                for p, q in ((a, b), (b, y), (y, x), (x, a))
                            )
                        )
    return sqs


def _small_rings(adj: dict[int, set[int]], n: int) -> list[dict]:
    """All 3- and 4-membered rings: {'atoms', 'edges'} (edges normalised)."""
    rings = []
    for a, b, c in _find_triangles(adj, n):
        edges = frozenset(
            (min(p, q), max(p, q)) for p, q in ((a, b), (b, c), (a, c))
        )
        rings.append({"atoms": (a, b, c), "edges": edges})
    for edges in _find_squares(adj, n):
        atoms = sorted({i for e in edges for i in e})
        rings.append({"atoms": tuple(atoms), "edges": edges})
    return rings


def _max_disjoint_paths(n: int, adj: dict[int, set[int]], s: int, t: int,
                       limit: int = 3) -> int:
    """Maximum number of internally node-disjoint s-t paths (max flow).

    Unit capacities everywhere (intermediate nodes admit one path each;
    endpoints admit ``limit``), edges admit one unit each, and the search
    stops once ``limit`` paths are established. Bounded: at most ``limit``
    BFS augmentations over a 2n-node graph.
    """
    cap: dict[tuple[int, int], int] = {}

    def add(u: int, v: int, c: int) -> None:
        cap[(u, v)] = cap.get((u, v), 0) + c

    for v in range(n):
        c = limit if v in (s, t) else 1
        add(v, v + n, c)
    for u in range(n):
        for v in adj[u]:
            add(u + n, v, 1)
    flow = 0
    while flow < limit:
        parent: dict[int, int] = {s + n: -1}
        queue = [s + n]
        while queue and (t) not in parent:
            u = queue.pop(0)
            for (x, y), c in cap.items():
                if x == u and c > 0 and y not in parent:
                    parent[y] = u
                    queue.append(y)
        if t not in parent:
            return flow
        v = t
        while v != s + n:
            u = parent[v]
            cap[(u, v)] -= 1
            cap[(v, u)] = cap.get((v, u), 0) + 1
            v = u
        flow += 1
    return flow


# ---------------------------------------------------------------------------
# Structural alerts (typed graph only, before any embedding/optimisation).
# ---------------------------------------------------------------------------


def _alert(id: str, atom_list: list[int], params: dict, meaning: str) -> dict:
    return {"id": id, "atoms": [int(i) for i in atom_list],
            "params": params, "meaning": meaning}


def _rdkit_aromatic_masks(atoms: list[int], bonds: list[list[int]]
                           ) -> tuple[set[int], set[tuple[int, int]]]:
    """Aromatic atoms/bonds off a stereo-free RDKit copy (shared perception).

    Returns (aromatic atom indices, normalized aromatic bond pairs). Empty on
    any build failure (no exclusion then). RDKit-dependent: sanitisation plus
    RDKit aromaticity perception on a stereo-free copy; stated here and in
    ``structural_alerts``. Shared by the ``enol``, ``polynitrogen_chain``
    and ``bridgehead_double_bond`` exclusions so one perception decides all
    three.
    """
    try:
        mol = to_rdkit(
            list(atoms), [list(b) for b in bonds],
            {"tetrahedral_centers": [], "double_bonds": [], "isomers": []},
            None,
        )
    except Exception:
        return set(), set()
    arom_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    arom_bonds: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            arom_bonds.add((min(a, b), max(a, b)))
    return arom_atoms, arom_bonds


def _is_saturated_carbon(i: int, adj: dict[int, set[int]],
                         orders: dict[tuple[int, int], int]) -> bool:
    """Every incident bond single (a carbonyl/imine carbon is not saturated)."""
    for j in adj[i]:
        if orders[(min(i, j), max(i, j))] != 1:
            return False
    return True


def structural_alerts(atoms: list[int], bonds: list[list[int]],
                      stereo_block: dict | None = None,
                      isomer: int | None = None) -> list[dict]:
    """Strain/reactivity warnings on the typed graph (never verdicts).

    Pure Python over atom type ids and bond orders, except RDKit-dependent
    processing stated here: ``bridgehead_double_bond`` reads SSSR ring
    membership off a stereo-free RDKit copy (its params say so), and the
    ``enol``, ``polynitrogen_chain`` and ``bridgehead_double_bond``
    exclusions use shared RDKit aromaticity perception
    (``_rdkit_aromatic_masks``) off the same stereo-free copy; all still run
    before any embedding or optimisation and are skipped when that copy
    cannot be built. ``stereo_block``/``isomer`` (candidate shape
    with ``stereoisomers`` strings, or internal shape with ``isomers``
    values) supply the requested trans assignments; without one, small
    rings with double bonds are covered by ``bridgehead_double_bond``.
    """
    n = len(atoms)
    elements = _elements(atoms)
    hyd = _hydrogens(atoms)
    orders = _bond_orders(bonds)
    adj = _adjacency(n, bonds)
    alerts: list[dict] = []
    arom_atoms, arom_bonds = _rdkit_aromatic_masks(atoms, bonds)

    double_bonds = [(a, b) for (a, b), o in orders.items() if o == 2]
    triple_bonds = [(a, b) for (a, b), o in orders.items() if o == 3]

    # Requested trans assignments (for trans_double_bond_in_small_ring),
    # translated into ring-substituent geometry: a reference that is not the
    # ring neighbour inverts the reading once per end (parity treatment,
    # as in reperceive_stereo). Only a trans ring geometry fires.
    ring_trans: set[tuple[int, int]] = set()
    if stereo_block is not None and isomer is not None:
        internal = candidate_block_to_rdkit_block(stereo_block)
        if not (0 <= isomer < len(internal["isomers"])):
            raise ValueError(f"unknown_stereoisomer_index: {isomer}")
        values = internal["isomers"][isomer]
        for entry, v in zip(internal["double_bonds"], values["double_bonds"]):
            if v == 1:
                a, b = entry["atoms"]
                ref_a, ref_b = entry["reference"]
                ring_a = {w for w in adj[a] if w != b
                          and _bfs_length(adj, w, b, skip=(a, b)) is not None}
                ring_b = {w for w in adj[b] if w != a
                          and _bfs_length(adj, w, a, skip=(a, b)) is not None}
                ref_a_is_ring = isinstance(ref_a, int) and ref_a in ring_a
                ref_b_is_ring = isinstance(ref_b, int) and ref_b in ring_b
                flip_a = not ref_a_is_ring
                flip_b = not ref_b_is_ring
                if True ^ flip_a ^ flip_b:
                    ring_trans.add((min(a, b), max(a, b)))

    for a, b in double_bonds:
        if (a, b) in ring_trans:
            size = smallest_ring_through_bond(adj, a, b)
            if size is not None and size < TRANS_DOUBLE_MAX_RING:
                alerts.append(_alert(
                    "trans_double_bond_in_small_ring", [a, b],
                    {"ring_size": size, "max_ring_atoms": TRANS_DOUBLE_MAX_RING,
                     "requested": "trans",
                     "ring_geometry": "trans"},
                    "a trans double bond in a ring of fewer than 8 atoms forces "
                    "severe twisting; the coordinate check decides whether the "
                    "relaxed geometry still shows trans"))

    for a, b in triple_bonds:
        size = smallest_ring_through_bond(adj, a, b)
        if size is not None and size < TRIPLE_MAX_RING:
            alerts.append(_alert(
                "triple_bond_in_small_ring", [a, b],
                {"ring_size": size, "max_ring_atoms": TRIPLE_MAX_RING},
                "a triple bond prefers linear geometry; a ring of fewer than "
                "8 atoms forces it to bend"))

    double_sets = {i: {j for j in adj[i] if orders[(min(i, j), max(i, j))] == 2}
                   for i in range(n)}
    for i in range(n):
        if elements[i] != "C" or len(double_sets[i]) < 2:
            continue
        incident = [smallest_ring_through_bond(adj, i, j)
                    for j in sorted(double_sets[i])]
        sizes = [s for s in incident if s is not None]
        if sizes and min(sizes) < CUMULENE_MAX_RING:
            alerts.append(_alert(
                "cumulene_in_small_ring", [i],
                {"ring_size": min(sizes),
                 "max_ring_atoms": CUMULENE_MAX_RING},
                "an atom with two double bonds prefers a linear axis; a "
                "ring of fewer than 9 atoms forces it to bend"))

    alerts.extend(_bridgehead_alerts(atoms, bonds, adj, orders,
                                     arom_atoms, arom_bonds))

    rings = _small_rings(adj, n)
    for ring in rings:
        if len(ring["atoms"]) != 3:
            continue
        a, b, c = ring["atoms"]
        if any(orders[(min(p, q), max(p, q))] > 1
               for p, q in ((a, b), (b, c), (a, c))):
            alerts.append(_alert(
                "three_membered_ring_unsaturation", [a, b, c],
                {"ring_size": 3},
                "a double or triple bond in a three-membered ring forces "
                "acute angles at unsaturated atoms"))
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i]["edges"] & rings[j]["edges"]
            if shared:
                atoms_ij = sorted({k for r in (rings[i], rings[j])
                                   for k in r["atoms"]})
                alerts.append(_alert(
                    "fused_small_rings", atoms_ij,
                    {"ring_sizes": sorted([len(rings[i]["atoms"]),
                                           len(rings[j]["atoms"])]),
                     "shared_bond": sorted(next(iter(shared)))},
                    "two 3- or 4-membered rings sharing a bond concentrate "
                    "angle strain on that bond"))
    counts = [0] * n
    for ring in rings:
        for k in ring["atoms"]:
            counts[k] += 1
    for k in range(n):
        if counts[k] >= 3:
            alerts.append(_alert(
                "cage_small_rings", [k],
                {"n_small_rings": counts[k],
                 "max_ring_atoms": SMALL_RING},
                "an atom shared by three or more 3- or 4-membered rings sits "
                "in a rigid cage of acute angles"))

    for a, b in sorted(orders):
        o = orders[(a, b)]
        ea, eb = elements[a], elements[b]
        if o == 1 and {ea, eb} == {"O"}:
            alerts.append(_alert(
                "peroxide", [a, b], {"bond_order": 1},
                "an O-O single bond is a weak linkage and a reactivity flag"))
    for i in range(n):
        if elements[i] != "O":
            continue
        o_nbrs = [j for j in adj[i]
                  if elements[j] == "O"
                  and orders[(min(i, j), max(i, j))] == 1]
        if len(o_nbrs) >= 2:
            chain = sorted({i} | set(o_nbrs))
            alerts.append(_alert(
                "polyoxide_chain", chain, {"n_oxygens": len(chain)},
                "three or more consecutive singly-bonded oxygens form a "
                "reactive oxygen chain"))
            break
    seen_triples: set[tuple[int, int, int]] = set()
    for i in range(n):
        if elements[i] != "N":
            continue
        for j in adj[i]:
            if elements[j] != "N" or orders[(min(i, j), max(i, j))] != 1:
                continue
            for k in adj[j]:
                if k != i and elements[k] == "N" \
                        and orders[(min(j, k), max(j, k))] == 1:
                    triple = tuple(sorted([i, j, k]))
                    if triple in seen_triples:
                        continue
                    if i in arom_atoms or j in arom_atoms or k in arom_atoms:
                        continue
                    pairs = [(min(i, j), max(i, j)), (min(j, k), max(j, k))]
                    if any(p in arom_bonds for p in pairs):
                        continue
                    seen_triples.add(triple)
                    alerts.append(_alert(
                        "polynitrogen_chain", list(triple),
                        {"n_nitrogens": 3},
                        "three or more consecutive singly-bonded nitrogens "
                        "form a reactive nitrogen chain"))
                    break
            else:
                continue
            break
    for a, b in sorted(orders):
        ea, eb = elements[a], elements[b]
        pair = {ea, eb}
        if len(pair) == 2 and "N" in pair and len(pair & HALOGENS) == 1:
            alerts.append(_alert(
                "n_halogen", [a, b], {},
                "a nitrogen-halogen bond is a reactivity flag"))
        if len(pair) == 2 and "O" in pair and len(pair & HALOGENS) == 1:
            alerts.append(_alert(
                "o_halogen", [a, b], {},
                "an oxygen-halogen bond is a reactivity flag"))

    for a, b in double_bonds:
        if elements[a] != "C" or elements[b] != "C":
            continue
        if a in arom_atoms or b in arom_atoms \
                or (min(a, b), max(a, b)) in arom_bonds:
            continue
        for c, other in ((a, b), (b, a)):
            for k in adj[c]:
                if k == other:
                    continue
                if elements[k] == "O" and hyd[k] >= 1 \
                        and orders[(min(c, k), max(c, k))] == 1:
                    alerts.append(_alert(
                        "enol", sorted([a, b, k]), {},
                        "a C=C-OH group is a tautomer flag: the keto form may "
                        "dominate in solution"))
    for i in range(n):
        if elements[i] != "C":
            continue
        o_nbrs = [j for j in adj[i]
                  if elements[j] == "O"
                  and orders[(min(i, j), max(i, j))] == 1]
        n_nbrs = [j for j in adj[i]
                  if elements[j] == "N"
                  and orders[(min(i, j), max(i, j))] == 1]
        oh = [j for j in o_nbrs if hyd[j] >= 1]
        if len(oh) >= 2 and _is_saturated_carbon(i, adj, orders):
            alerts.append(_alert(
                "geminal_diol", sorted([i] + oh[:2]), {},
                "a carbon with two hydroxyl groups is a hydration flag: the "
                "carbonyl form may dominate"))
        if oh and n_nbrs and _is_saturated_carbon(i, adj, orders):
            alerts.append(_alert(
                "hemiaminal", sorted([i, oh[0], n_nbrs[0]]), {},
                "a carbon with both hydroxyl and amino substitution is a "
                "tautomer flag"))
        if o_nbrs and n_nbrs and _is_saturated_carbon(i, adj, orders):
            o0 = o_nbrs[0]
            alerts.append(_alert(
                "geminal_amino_alcohol", sorted([i, o0, n_nbrs[0]]),
                {"hydroxyl": bool(hyd[o0] >= 1)},
                "a saturated carbon bonded to both oxygen and nitrogen is a "
                "hydration flag (hydroxyl true: amino alcohol; false: amino "
                "ether): the carbonyl plus free amine/ammonia forms may "
                "dominate"))

    for i, t in enumerate(atoms):
        el, _, valence = ATOM_TYPES[t]
        if el in BASE_VALENCE and valence > BASE_VALENCE[el]:
            alerts.append(_alert(
                "hypervalent_sulfur_or_phosphorus", [i],
                {"element": el, "valence": valence,
                 "routine_valence": BASE_VALENCE[el]},
                "sulphur above valence 2, or phosphorus above valence 3, "
                "carries expanded-valence bonding a force field may describe "
                "poorly"))
    return alerts


def _bridgehead_alerts(atoms: list[int], bonds: list[list[int]],
                       adj: dict[int, set[int]],
                       orders: dict[tuple[int, int], int],
                       arom_atoms: set[int] | None = None,
                       arom_bonds: set[tuple[int, int]] | None = None,
                       ) -> list[dict]:
    """Anti-Bredt flags. Ring membership comes from the SSSR bond rings
    (RDKit ring info) of a stereo-free copy of the molecule; aromatic
    double bonds are excluded via the shared aromaticity perception. A
    bridgehead endpoint must sit in at least two SSSR rings (ring
    relationship, not flow alone), and its partner must be non-adjacent
    (adjacent pairs are fused, not bridged) with three unit-capacity
    disjoint paths. Still runs before any embedding or optimisation.
    Skipped when the copy cannot be built."""
    n = len(atoms)
    arom_atoms = arom_atoms or set()
    arom_bonds = arom_bonds or set()
    double_bonds = [(a, b) for (a, b), o in orders.items() if o == 2]
    if not double_bonds:
        return []
    try:
        mol = to_rdkit(
            list(atoms), [list(b) for b in bonds],
            {"tetrahedral_centers": [], "double_bonds": [], "isomers": []},
            None,
        )
    except Exception:
        return []
    ri = mol.GetRingInfo()
    sssr_atoms = [set(r) for r in ri.AtomRings()]
    try:
        bond_rings = [set(r) for r in ri.BondRings()]
    except Exception:
        bond_rings = []
    bond_index: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_index[(min(a, b), max(a, b))] = bond.GetIdx()
    ring_count = [0] * n
    for r in sssr_atoms:
        for v in r:
            if 0 <= v < n:
                ring_count[v] += 1
    degree3 = {i for i in range(n) if len(adj[i]) >= 3}
    out = []
    for a, b in double_bonds:
        if (min(a, b), max(a, b)) in arom_bonds or a in arom_atoms \
                or b in arom_atoms:
            continue
        bi = bond_index.get((min(a, b), max(a, b)))
        if bi is None:
            continue
        holding = [len(r) for r in bond_rings if bi in r]
        if not holding:
            # Fall back to atom co-membership only when bond rings are
            # unavailable; normally every bond in a ring has a bond ring.
            holding = [len(r) for r in sssr_atoms if a in r and b in r]
        if not holding:
            continue
        largest = max(holding)
        if largest >= BRIDGEHEAD_MAX_RING:
            continue
        bridgehead = None
        for v in (a, b):
            if v not in degree3 or ring_count[v] < 2 or v in arom_atoms:
                continue
            for u in degree3:
                if u == v or u in adj[v] or ring_count[u] < 2:
                    continue
                if _max_disjoint_paths(n, adj, v, u) >= 3:
                    bridgehead = v
                    break
            if bridgehead is not None:
                break
        if bridgehead is not None:
            out.append(_alert(
                "bridgehead_double_bond", [a, b],
                {"ring_size": largest,
                 "max_ring_atoms": BRIDGEHEAD_MAX_RING,
                 "bridgehead_atom": bridgehead,
                 "ring_source": "SSSR bond rings (RDKit ring info on a stereo-free copy)"},
                "a double bond at a bridgehead of a small bridged bicyclic "
                "system cannot be planar (Bredt-type strain)"))
    return out


# ---------------------------------------------------------------------------
# Conversion, embedding, optimisation.
# ---------------------------------------------------------------------------


def _requested_assignment(stereo_block: dict | None,
                          isomer: int | None) -> tuple[dict, dict]:
    """(internal block, requested values) for one isomer (or empty)."""
    if stereo_block is None or isomer is None:
        return ({"tetrahedral_centers": [], "double_bonds": [],
                 "isomers": []},
                {"tetrahedral": [], "double_bonds": []})
    internal = candidate_block_to_rdkit_block(stereo_block)
    if not (0 <= isomer < len(internal["isomers"])):
        raise ValueError(f"unknown_stereoisomer_index: {isomer}")
    values = internal["isomers"][isomer]
    tet_names = ["ccw" if v == 0 else "cw" for v in values["tetrahedral"]]
    bond_names = ["cis" if v == 0 else "trans" for v in values["double_bonds"]]
    return internal, {"tetrahedral": tet_names, "double_bonds": bond_names}


def _convert(atoms: list[int], bonds: list[list[int]], internal: dict,
             isomer: int | None) -> tuple[Chem.Mol | None, dict, str]:
    """Returns (mol or None, assigned record, error message or '')."""
    assigned: dict = {"tetrahedral": [], "double_bonds": []}
    try:
        mol = to_rdkit(list(atoms), [list(b) for b in bonds], internal, isomer)
    except LostAssignmentError as e:
        return None, assigned, f"stereo_assignment_lost: {e}"
    except Exception as e:  # sanitisation / valence failures are unsupported
        return None, assigned, f"{type(e).__name__}: {e}"
    if isomer is not None:
        assigned = {
            "tetrahedral": [e["atom"] for e in internal["tetrahedral_centers"]],
            "double_bonds": [list(e["atoms"])
                             for e in internal["double_bonds"]],
        }
    return mol, assigned, ""


def _embed(mol: Chem.Mol, cfg: VerifyConfig) -> tuple[Chem.Mol, dict]:
    mol_h = Chem.AddHs(mol)
    rounds = []
    params = AllChem.ETKDGv3()
    params.randomSeed = cfg.seed
    params.numThreads = cfg.num_threads
    params.useRandomCoords = False
    ids = list(AllChem.EmbedMultipleConfs(
        mol_h, numConfs=cfg.conformers, params=params))
    rounds.append({"use_random_coords": False, "conformers_requested": cfg.conformers,
                   "successes": len(ids)})
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(
            mol_h, numConfs=cfg.conformers, params=params))
        rounds.append({"use_random_coords": True,
                       "conformers_requested": cfg.conformers,
                       "successes": len(ids)})
    info = {
        "conformers_requested": sum(r["conformers_requested"] for r in rounds),
        "successes": sum(r["successes"] for r in rounds),
        "rounds": rounds,
        "seed": cfg.seed,
        "conformer_ids": [int(i) for i in ids],
        "n_atoms_3d": mol_h.GetNumAtoms(),
    }
    return mol_h, info


def _optimise(mol_h: Chem.Mol, conf_ids: list[int],
              cfg: VerifyConfig) -> tuple[dict | None, str]:
    """Returns (optimisation record or None, missing-params message or '')."""
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        method = "MMFF94"
        raw = AllChem.MMFFOptimizeMoleculeConfs(
            mol_h, numThreads=cfg.num_threads, maxIters=cfg.max_iters)
    elif AllChem.UFFHasAllMoleculeParams(mol_h):
        method = "UFF"
        raw = AllChem.UFFOptimizeMoleculeConfs(
            mol_h, numThreads=cfg.num_threads, maxIters=cfg.max_iters)
    else:
        return None, "no_force_field_parameters"
    conformers = [
        {"conf_id": int(cid), "not_converged": int(nc),
         "energy_kcal_mol": float(e)}
        for cid, (nc, e) in zip(conf_ids, list(raw))
    ]
    return {
        "method": method,
        "conformers": conformers,
        "max_iterations": cfg.max_iters,
        "num_threads": cfg.num_threads,
    }, ""


# ---------------------------------------------------------------------------
# Geometry diagnostics on the kept conformer.
# ---------------------------------------------------------------------------


def _positions(mol_h: Chem.Mol, conf_id: int) -> np.ndarray:
    conf = mol_h.GetConformer(conf_id)
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z]
                     for i in range(mol_h.GetNumAtoms())])


def _bond_length_report(mol_h: Chem.Mol, pos: np.ndarray) -> dict:
    pt = Chem.GetPeriodicTable()
    outliers = []
    assessed = 0
    skipped = 0
    for bond in mol_h.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            cov = pt.GetRcovalent(mol_h.GetAtomWithIdx(a).GetSymbol()) \
                + pt.GetRcovalent(mol_h.GetAtomWithIdx(b).GetSymbol())
        except Exception:
            skipped += 1
            continue
        if cov <= 0:
            skipped += 1
            continue
        length = float(np.linalg.norm(pos[a] - pos[b]))
        ratio = length / cov
        assessed += 1
        if ratio < BOND_RATIO_LOW or ratio > BOND_RATIO_HIGH:
            outliers.append({
                "atoms": [a, b], "length_a": length,
                "covalent_sum_a": float(cov), "ratio": ratio,
                "allowed": [BOND_RATIO_LOW, BOND_RATIO_HIGH]})
    return {"assessed": assessed, "skipped_no_radii": skipped,
            "outliers": outliers,
            "threshold": [BOND_RATIO_LOW, BOND_RATIO_HIGH],
            "note": "each bond length is compared to the sum of covalent "
                    "radii (RDKit periodic table GetRcovalent)"}


def _contact_report(mol_h: Chem.Mol, pos: np.ndarray) -> dict:
    n = mol_h.GetNumAtoms()
    adj = {i: set() for i in range(n)}
    for bond in mol_h.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].add(b)
        adj[b].add(a)
    pt = Chem.GetPeriodicTable()
    contacts = []
    assessed = 0
    skipped = 0
    for i in range(n):
        dist = _bfs_lengths(adj, i)
        for j in range(i + 1, n):
            if dist.get(j, 10**9) < 4:
                continue
            try:
                vdw = pt.GetRvdw(mol_h.GetAtomWithIdx(i).GetSymbol()) \
                    + pt.GetRvdw(mol_h.GetAtomWithIdx(j).GetSymbol())
            except Exception:
                skipped += 1
                continue
            if vdw <= 0:
                skipped += 1
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            threshold = CONTACT_FACTOR * vdw
            assessed += 1
            if d < threshold:
                contacts.append({"atoms": [i, j], "distance_a": d,
                                 "threshold_a": float(threshold),
                                 "factor": CONTACT_FACTOR})
    return {"assessed_pairs": assessed, "skipped_no_radii": skipped,
            "contacts": contacts, "factor": CONTACT_FACTOR,
            "exclusion": "pairs at 1-2, 1-3 and 1-4 separation (shortest "
                         "path of 3 bonds or fewer) are excluded; only "
                         "pairs with a shortest path of 4+ bonds are assessed",
            "note": "threshold is 0.7 times the sum of van der Waals radii "
                    "(RDKit periodic table GetRvdw)"}


def _bfs_lengths(adj: dict[int, set[int]], src: int) -> dict[int, int]:
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def _angle_report(mol_h: Chem.Mol, pos: np.ndarray,
                  energy: float, n_heavy: int) -> dict:
    sp3_dev = 0.0
    sp2_dev = 0.0
    sp3_seen = False
    sp2_seen = False
    outliers = []
    nbrs = {i: [nn.GetIdx() for nn in mol_h.GetAtomWithIdx(i).GetNeighbors()]
            for i in range(mol_h.GetNumAtoms())}
    for i in range(mol_h.GetNumAtoms()):
        atom = mol_h.GetAtomWithIdx(i)
        if atom.GetSymbol() != "C":
            continue
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP3:
            ideal, is_sp3 = SP3_IDEAL_DEG, True
        elif hyb == Chem.HybridizationType.SP2:
            ideal, is_sp3 = SP2_IDEAL_DEG, False
        else:
            continue
        for x, y in [(x, y) for ii, x in enumerate(nbrs[i])
                     for y in nbrs[i][ii + 1:]]:
            v1 = pos[x] - pos[i]
            v2 = pos[y] - pos[i]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom == 0:
                continue
            cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2)) / denom))
            angle = math.degrees(math.acos(cos_a))
            dev = abs(angle - ideal)
            if is_sp3:
                sp3_seen = True
                sp3_dev = max(sp3_dev, dev)
            else:
                sp2_seen = True
                sp2_dev = max(sp2_dev, dev)
            if dev > ANGLE_OUTLIER_DEG:
                outliers.append({"atom": i, "neighbours": [x, y],
                                 "angle_deg": angle,
                                 "deviation_deg": dev, "ideal_deg": ideal})
    return {"sp3_max_deviation_deg": float(sp3_dev) if sp3_seen else None,
            "sp2_max_deviation_deg": float(sp2_dev) if sp2_seen else None,
            "outliers": outliers,
            "outlier_threshold_deg": ANGLE_OUTLIER_DEG,
            "energy_per_heavy_atom_kcal_mol": float(energy) / max(n_heavy, 1),
            "note": "largest angle deviation at sp3 carbons from 109.5 "
                    "degrees and at sp2 carbons from 120 degrees; "
                    "a deviation above 25 degrees is flagged as angle_outlier"}


def _signed_volume(p0: np.ndarray, p1: np.ndarray,
                   p2: np.ndarray, p3: np.ndarray) -> float:
    return float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray,
                  p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    denom = float(np.linalg.norm(n0) * np.linalg.norm(n1))
    if denom == 0:
        return float("nan")
    cos_d = max(-1.0, min(1.0, float(np.dot(n0, n1)) / denom))
    sign = 1.0 if float(np.dot(n0, b2)) >= 0 else -1.0
    return sign * math.degrees(math.acos(cos_d))


def reperceive_stereo(mol_h: Chem.Mol, conf_id: int, internal: dict,
                      requested: dict) -> dict:
    """Re-perceive every assigned stereo element from 3D coordinates.

    Tetrahedral: sign of the signed volume of the ligands in the
    convention's ligand order (hydrogen from the explicit atom); positive
    reads ``cw``, negative ``ccw``. Double bond: dihedral between the two
    reference ligands; |dihedral| < 90 reads cis. Near-degenerate cases
    (volume under 0.05 A^3, dihedral within 10 degrees of 90) read
    ``undetermined`` instead of guessing.
    """
    pos = _positions(mol_h, conf_id)
    heavy_nbrs = {i: {nn.GetIdx()
                      for nn in mol_h.GetAtomWithIdx(i).GetNeighbors()
                      if nn.GetAtomicNum() > 1}
                  for i in range(mol_h.GetNumAtoms())}
    tetra_out = []
    for entry, want in zip(internal["tetrahedral_centers"],
                           requested["tetrahedral"]):
        centre = entry["atom"]
        lig_pos = []
        missing = False
        for lig in entry["ligands"]:
            if isinstance(lig, int):
                lig_pos.append(pos[lig])
            else:
                hydrogens = [nn.GetIdx()
                             for nn in
                             mol_h.GetAtomWithIdx(centre).GetNeighbors()
                             if nn.GetIdx() not in heavy_nbrs[centre]]
                if len(hydrogens) != 1:
                    missing = True
                    break
                lig_pos.append(pos[hydrogens[0]])
        if missing or len(lig_pos) != 4:
            tetra_out.append({"atom": centre, "requested": want,
                              "observed": "undetermined",
                              "reason": "ligand hydrogen not resolved",
                              "match": None})
            continue
        vol = _signed_volume(lig_pos[0], lig_pos[1], lig_pos[2], lig_pos[3])
        if abs(vol) < VOLUME_TOLERANCE_A3:
            tetra_out.append({"atom": centre, "requested": want,
                              "observed": "undetermined",
                              "signed_volume_a3": vol,
                              "reason": "near-degenerate signed volume",
                              "match": None})
        else:
            observed = "cw" if vol > 0 else "ccw"
            tetra_out.append({"atom": centre, "requested": want,
                              "observed": observed,
                              "signed_volume_a3": vol,
                              "match": observed == want})
    bond_out = []
    for entry, want in zip(internal["double_bonds"],
                           requested["double_bonds"]):
        a, b = entry["atoms"]
        ref_a, ref_b = entry["reference"]
        want_trans = want == "trans"
        # Mirror of to_rdkit._resolve_end: a substituted reference (a real
        # heavy ligand standing in for H/lone_pair) inverts the reading.
        flips = []
        ends = []
        ok = True
        for end, partner, ref in ((a, b, ref_a), (b, a, ref_b)):
            nbrs = sorted(nn.GetIdx()
                          for nn in mol_h.GetAtomWithIdx(end).GetNeighbors()
                          if nn.GetIdx() != partner
                          and nn.GetAtomicNum() > 1)
            if isinstance(ref, int):
                ends.append(ref)
                flips.append(False)
            elif nbrs:
                ends.append(nbrs[0])
                flips.append(True)
            else:
                hydrogens = [nn.GetIdx()
                             for nn in
                             mol_h.GetAtomWithIdx(end).GetNeighbors()
                             if nn.GetIdx() != partner]
                if not hydrogens:
                    ok = False
                    break
                ends.append(hydrogens[0])
                flips.append(False)
        if not ok:
            bond_out.append({"atoms": [a, b], "requested": want,
                             "observed": "undetermined",
                             "reason": "reference ligand not resolved",
                             "match": None})
            continue
        expect_trans_xy = want_trans ^ (flips[0] != flips[1])
        dih = _dihedral_deg(pos[ends[0]], pos[a], pos[b], pos[ends[1]])
        if math.isnan(dih) or abs(abs(dih) - 90.0) < DIHEDRAL_MARGIN_DEG:
            bond_out.append({"atoms": [a, b], "requested": want,
                             "observed": "undetermined",
                             "dihedral_deg": dih,
                             "reason": "dihedral within 10 degrees of 90",
                             "match": None})
        else:
            observed_trans_xy = abs(dih) >= 90.0
            observed = ("trans" if observed_trans_xy != (flips[0] != flips[1])
                        else "cis")
            bond_out.append({"atoms": [a, b], "requested": want,
                             "observed": observed, "dihedral_deg": float(dih),
                             "match": observed == want})
    return {
        "tetrahedral": tetra_out,
        "double_bonds": bond_out,
        "tolerances": {
            "signed_volume_a3": VOLUME_TOLERANCE_A3,
            "dihedral_margin_deg": DIHEDRAL_MARGIN_DEG,
            "cis_threshold": "|dihedral| < 90 degrees reads cis"},
    }


# ---------------------------------------------------------------------------
# One molecule (constitution, or one stereoisomer).
# ---------------------------------------------------------------------------


def _method_record(cfg: VerifyConfig, force_field: str | None) -> dict:
    return {"force_field": force_field,
            "rdkit_version": rdkit.__version__,
            "etkdg_version": "ETKDGv3",
            "seed": cfg.seed,
            "conformers_requested": cfg.conformers,
            "max_iterations": cfg.max_iters,
            "num_threads": cfg.num_threads,
            "timeout_s": cfg.timeout_s}


def verify_candidate(atoms: list[int], bonds: list[list[int]],
                     stereo_block: dict | None = None,
                     isomer: int | None = None,
                     config: VerifyConfig | None = None) -> dict:
    """Verify one constitution (isomer=None) or one stereoisomer.

    Returns a per-molecule result dict with protocol
    ``physical-verification-v1``, exactly one status of
    ``force_field_optimization_converged`` / ``calculation_failed`` /
    ``unsupported`` / ``error``, the alert list (present with every
    status), and the three verbatim non-evaluation statements.
    """
    cfg = config or VerifyConfig()
    t0 = time.time()
    label = "constitution" if isomer is None else f"stereoisomer_{isomer}"
    try:
        internal, requested = _requested_assignment(stereo_block, isomer)
    except ValueError as e:
        return _failure(
            atoms, bonds, cfg, label, isomer, None, [], ERROR, str(e),
            {"tetrahedral": [], "double_bonds": []}, t0,
            requested={"tetrahedral": [], "double_bonds": []})
    try:
        alerts = structural_alerts(list(atoms), [list(b) for b in bonds],
                                   stereo_block, isomer)
    except ValueError as e:  # bad isomer index surfaced by the alerts
        return _failure(
            atoms, bonds, cfg, label, isomer, None, [], ERROR, str(e),
            {"tetrahedral": [], "double_bonds": []}, t0, requested=requested)
    except Exception as e:
        return _failure(
            atoms, bonds, cfg, label, isomer, None, [], ERROR,
            f"alerts_failed: {type(e).__name__}: {e}",
            {"tetrahedral": [], "double_bonds": []}, t0, requested=requested)

    mol, assigned, conv_err = _convert(atoms, bonds, internal, isomer)
    conversion = {"ok": mol is not None,
                  "assigned": assigned,
                  "detail": conv_err or "conversion and sanitisation succeeded"}
    if mol is None:
        status = ERROR if conv_err.startswith("stereo_assignment_lost") \
            else UNSUPPORTED
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, status,
            conv_err, assigned, t0, requested=requested)

    try:
        mol_h, embed_info = _embed(mol, cfg)
    except Exception as e:
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, FAILED,
            f"embedding_exception: {type(e).__name__}: {e}", assigned, t0,
            requested=requested,
            embed_info={"conformers_requested": 0, "successes": 0, "rounds": []})
    if not embed_info["conformer_ids"]:
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, FAILED,
            "embedding_failed: this bounded procedure found no 3D geometry "
            f"in {embed_info['conformers_requested']} requested conformers "
            "(this says the embedding procedure failed, never that the "
            "molecule cannot exist)",
            assigned, t0, requested=requested, embed_info=embed_info)

    try:
        opt_info, missing = _optimise(mol_h, embed_info["conformer_ids"], cfg)
    except Exception as e:
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, FAILED,
            f"optimisation_exception: {type(e).__name__}: {e}", assigned, t0,
            requested=requested, embed_info=embed_info)
    if opt_info is None:
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, UNSUPPORTED,
            "no_force_field_parameters: neither MMFF94 nor UFF has "
            "parameters for the whole molecule", assigned, t0,
            requested=requested, embed_info=embed_info)
    converged = [c for c in opt_info["conformers"] if c["not_converged"] == 0]
    if not converged:
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, FAILED,
            "optimisation_not_converged: no conformer converged within "
            f"{cfg.max_iters} iterations", assigned, t0,
            requested=requested, embed_info=embed_info,
            optim_info=opt_info, force_field=opt_info["method"])
    kept = min(converged, key=lambda c: c["energy_kcal_mol"])

    pos = _positions(mol_h, kept["conf_id"])
    finite = bool(np.all(np.isfinite(pos)))
    bond_rep = _bond_length_report(mol_h, pos)
    contact_rep = _contact_report(mol_h, pos)
    n_heavy = mol_h.GetNumAtoms() - sum(
        1 for i in range(mol_h.GetNumAtoms())
        if mol_h.GetAtomWithIdx(i).GetAtomicNum() == 1)
    angle_rep = _angle_report(mol_h, pos, kept["energy_kcal_mol"], n_heavy)
    stereo_rep = None
    if isomer is not None and (
            requested["tetrahedral"] or requested["double_bonds"]):
        stereo_rep = reperceive_stereo(mol_h, kept["conf_id"], internal,
                                       requested)
    failed: list[str] = []
    if not finite:
        failed.append("coordinates_not_finite")
    stereo_mismatch = False
    if stereo_rep is not None:
        for entry in stereo_rep["tetrahedral"] + stereo_rep["double_bonds"]:
            if entry["match"] is False:
                stereo_mismatch = True
        if stereo_mismatch:
            failed.append("stereo_not_preserved")
    diagnostics = {
        "coordinates_finite": finite,
        "bond_lengths": bond_rep,
        "close_contacts": contact_rep,
        "angles": angle_rep,
        "stereo": stereo_rep,
        "failed": failed,
        "warnings": sorted(
            {*(["bond_length_outlier"] if bond_rep["outliers"] else []),
              *(["close_contact"] if contact_rep["contacts"] else []),
              *(["angle_outlier"] if angle_rep["outliers"] else []),
              *(["stereo_undetermined"] if stereo_rep is not None and any(
                  e["match"] is None
                  for e in stereo_rep["tetrahedral"] + stereo_rep["double_bonds"])
                 else [])}),
        "note": "bond_length_outlier, close_contact, angle_outlier and "
                "stereo_undetermined are recorded warnings; "
                "coordinates_not_finite and stereo_not_preserved fail the "
                "molecule",
    }
    if failed:
        reason = ("stereo_not_preserved: the relaxed coordinates do not show "
                  "the requested stereo assignment"
                  if stereo_mismatch else
                  "coordinates_not_finite: the kept conformer has "
                  "non-finite coordinates")
        return _failure(
            atoms, bonds, cfg, label, isomer, conversion, alerts, FAILED,
            reason, assigned, t0, requested=requested,
            embed_info=embed_info, optim_info=opt_info,
            force_field=opt_info["method"], diagnostics=diagnostics,
            kept=kept)
    return {
        "protocol": PROTOCOL,
        "label": label,
        "subject": {"isomer": isomer, "requested": requested},
        "status": CONVERGED,
        "reason": ("embedding and optimisation converged; geometry "
                   "diagnostics recorded (see diagnostics and warnings)"),
        "alerts": alerts,
        "conversion": conversion,
        "embedding": embed_info,
        "optimisation": {**opt_info, "kept_conf_id": kept["conf_id"],
                         "kept_energy_kcal_mol": kept["energy_kcal_mol"]},
        "diagnostics": diagnostics,
        "stability": dict(STABILITY),
        "electronic_structure": dict(ELECTRONIC_STRUCTURE),
        "energy_comparability": ENERGY_COMPARABILITY,
        "method": _method_record(cfg, opt_info["method"]),
        "elapsed_s": time.time() - t0,
    }


def _failure(atoms: list[int], bonds: list[list[int]], cfg: VerifyConfig,
             label: str, isomer: int | None, conversion: dict | None,
             alerts: list[dict], status: str, reason: str, assigned: dict,
             t0: float, requested: dict | None = None,
             embed_info: dict | None = None,
             optim_info: dict | None = None,
             force_field: str | None = None,
             diagnostics: dict | None = None, kept: dict | None = None) -> dict:
    if optim_info is not None and kept is not None:
        optim_info = {**optim_info, "kept_conf_id": kept["conf_id"],
                      "kept_energy_kcal_mol": kept["energy_kcal_mol"]}
    return {
        "protocol": PROTOCOL,
        "label": label,
        "subject": {"isomer": isomer,
                    "requested": requested or {"tetrahedral": [],
                                               "double_bonds": []}},
        "status": status,
        "reason": reason,
        "alerts": alerts,
        "conversion": conversion or {"ok": False, "assigned": assigned,
                                     "detail": reason},
        "embedding": embed_info,
        "optimisation": optim_info,
        "diagnostics": diagnostics,
        "stability": dict(STABILITY),
        "electronic_structure": dict(ELECTRONIC_STRUCTURE),
        "energy_comparability": ENERGY_COMPARABILITY,
        "method": _method_record(cfg, force_field),
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Response-level driver (CLI).
# ---------------------------------------------------------------------------


def _verify_unit_worker(payload: dict) -> dict:
    cfg = VerifyConfig.from_dict(payload["config"])
    return verify_candidate(payload["atoms"], payload["bonds"],
                            payload.get("stereo"), payload.get("isomer"),
                            cfg)


def _worker_entry(payload: dict, queue) -> None:
    try:
        queue.put(("ok", _verify_unit_worker(payload)))
    except Exception as e:  # pragma: no cover - crash path
        try:
            queue.put(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass


def _preliminary_alerts(atoms: list[int], bonds: list[list[int]],
                        stereo: dict | None, isomer: int | None) -> list[dict]:
    """Best-effort alerts computed in the parent for timeout/crash results."""
    try:
        return structural_alerts(list(atoms), [list(b) for b in bonds],
                                 stereo, isomer)
    except Exception:
        return []


def run_unit_in_worker(atoms: list[int], bonds: list[list[int]],
                       stereo: dict | None, isomer: int | None,
                       cfg: VerifyConfig, _context=None) -> dict:
    """Evaluate one molecule in a worker process with a wall-clock limit.

    The worker is always ended and joined: on timeout it is terminated then
    joined; on crash the (already exited) process is joined and no inline
    fallback runs without a limit. Timeout/crash results keep the
    parent-computed preliminary alerts instead of ``[]``.
    """
    label = "constitution" if isomer is None else f"stereoisomer_{isomer}"
    prelim = _preliminary_alerts(atoms, bonds, stereo, isomer)
    payload = {"atoms": list(atoms),
               "bonds": [list(b) for b in bonds],
               "stereo": stereo, "isomer": isomer,
               "config": cfg.to_dict()}
    ctx = _context or mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(payload, queue))
    proc.start()
    proc.join(timeout=cfg.timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return _failure(
            atoms, bonds, cfg, label, isomer,
            {"ok": False,
             "assigned": {"tetrahedral": [], "double_bonds": []},
             "detail": "timeout"}, prelim, FAILED,
            f"timeout: the per-molecule wall-clock limit of "
            f"{cfg.timeout_s}s was exceeded; the worker was terminated "
            f"and joined",
            {"tetrahedral": [], "double_bonds": []}, time.time())
    proc.join()
    try:
        kind, value = queue.get_nowait()
    except Exception:
        kind, value = "error", f"worker exitcode {proc.exitcode}"
    try:
        queue.close()
    except Exception:
        pass
    if kind == "ok":
        return value
    return _failure(
        atoms, bonds, cfg, label, isomer,
        {"ok": False,
         "assigned": {"tetrahedral": [], "double_bonds": []},
         "detail": "worker_failed"}, prelim, ERROR,
        f"worker_failed: {value}",
        {"tetrahedral": [], "double_bonds": []}, time.time())


def _candidate_status(units: list[dict]) -> str:
    order = {ERROR: 0, FAILED: 1, UNSUPPORTED: 2, CONVERGED: 3}
    worst = min(units, key=lambda u: order[u["status"]])
    return worst["status"]


def verify_response(response: dict, cfg: VerifyConfig, mode: str,
                    max_candidates: int | None,
                    run_unit) -> dict:
    """Copy of `response` with `physical_verification` replaced.

    `run_unit(atoms, bonds, stereo, isomer)` evaluates one molecule; the
    caller supplies the timeout wrapper. Candidate order and count are
    untouched; candidates are never re-ranked.
    """
    out = dict(response)
    candidates = list(response.get("candidates", []))
    new_candidates = []
    status_counts = {CONVERGED: 0, FAILED: 0, UNSUPPORTED: 0, ERROR: 0}
    isomer_counts = {CONVERGED: 0, FAILED: 0, UNSUPPORTED: 0, ERROR: 0}
    alert_counts: dict[str, int] = {}
    n_isomer_units = 0
    skipped = 0
    for rank, cand in enumerate(candidates):
        if max_candidates is not None and rank >= max_candidates:
            new_candidates.append(cand)
            skipped += 1
            continue
        atoms = cand.get("atoms", [])
        bonds = cand.get("bonds", [])
        stereo = cand.get("stereo")
        if not isinstance(atoms, list) or not isinstance(bonds, list):
            unit = _failure(
                [], [], cfg, "constitution", None,
                {"ok": False, "assigned": {"tetrahedral": [],
                                           "double_bonds": []},
                 "detail": "missing_graph"}, [], ERROR,
                "missing_graph: candidate has no atoms/bonds lists",
                {"tetrahedral": [], "double_bonds": []}, time.time())
            units = [unit]
            isomer_units: list[dict] = []
        else:
            units = [run_unit(atoms, bonds, stereo, None)]
            isomer_units = []
            isomers = stereo.get("stereoisomers", []) \
                if isinstance(stereo, dict) else []
            if isomers and mode != "none":
                idxs = range(len(isomers)) if mode == "all" else [0]
                for i in idxs:
                    isomer_units.append(run_unit(atoms, bonds, stereo, i))
        for u in units + isomer_units:
            for a in u["alerts"]:
                alert_counts[a["id"]] = alert_counts.get(a["id"], 0) + 1
        for u in isomer_units:
            isomer_counts[u["status"]] += 1
        n_isomer_units += len(isomer_units)
        conv = sum(1 for u in units + isomer_units
                   if u["status"] == CONVERGED)
        converged_clean = sum(
            1 for u in units + isomer_units
            if u["status"] == CONVERGED and not u["alerts"])
        failed = sum(1 for u in units + isomer_units
                     if u["status"] in (FAILED, ERROR))
        unsupp = sum(1 for u in units + isomer_units
                     if u["status"] == UNSUPPORTED)
        union = sorted({a["id"] for u in units + isomer_units
                        for a in u["alerts"]})
        warnings_union = sorted({
            w for u in units + isomer_units
            for w in (((u.get("diagnostics") or {}).get("warnings")) or [])
        })
        cand_status = _candidate_status(units + isomer_units)
        status_counts[cand_status] += 1
        entry = dict(cand)
        entry["physical_verification"] = {
            "protocol": PROTOCOL,
            "constitution": units[0],
            "stereoisomers": isomer_units,
            "summary": {
                "evaluated": len(units) + len(isomer_units),
                "converged_without_structural_alerts": converged_clean,
                "converged": conv,
                "failed": failed,
                "unsupported": unsupp,
                "candidate_status": cand_status,
                "alerts": union,
                "diagnostic_warnings": warnings_union,            },
            "stability": dict(STABILITY),
            "electronic_structure": dict(ELECTRONIC_STRUCTURE),
            "energy_comparability": ENERGY_COMPARABILITY,
        }
        new_candidates.append(entry)
    out["candidates"] = new_candidates
    return out, {"status_counts_candidates": status_counts,
                 "status_counts_stereoisomers": isomer_counts,
                 "n_isomer_units": n_isomer_units,
                 "alert_counts": alert_counts,
                 "candidates_skipped_by_max_candidates": skipped}


def load_responses(path: Path) -> tuple[list[dict], str]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        return docs, "jsonl"
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        return docs, "jsonl"
    if isinstance(doc, dict):
        return [doc], "single"
    if isinstance(doc, list):
        return doc, "array"
    raise ValueError("input must be a response object, an array, or JSONL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optional geometry-and-strain screen "
                    "(physical-verification-v1) for "
                    "molecular-completion-generate-v1 responses.")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--stereo", choices=["all", "first", "none"],
                        default="all")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    cfg = VerifyConfig(conformers=args.conformers, max_iters=args.max_iters,
                       seed=args.seed, timeout_s=args.timeout_s,
                       num_threads=1)
    responses, kind = load_responses(Path(args.inp))
    t0 = time.time()

    def run_unit(atoms: list[int], bonds: list[list[int]],
                 stereo: dict | None, isomer: int | None) -> dict:
        return run_unit_in_worker(atoms, bonds, stereo, isomer, cfg)

    totals = {CONVERGED: 0, FAILED: 0, UNSUPPORTED: 0, ERROR: 0}
    n_iso_total = 0
    done = []
    per_response = []
    for response in responses:
        entry, counts = verify_response(response, cfg, args.stereo,
                                        args.max_candidates, run_unit)
        done.append(entry)
        per_response.append(counts)
        for k in totals:
            totals[k] += counts["status_counts_candidates"][k]
        n_iso_total += counts["n_isomer_units"]
    wall = time.time() - t0
    method = _method_record(cfg, None)
    method["timeout_mechanism"] = (
        "each molecule (constitution or stereoisomer) runs in a worker "
        "process terminated and joined after timeout_s; a crashed worker "
        "is joined and reported without an inline fallback; the optimiser "
        f"additionally caps each conformer at {cfg.max_iters} iterations")
    method["stereo_mode"] = args.stereo
    for entry, counts in zip(done, per_response):
        summary = {
            "protocol": PROTOCOL,
            "status_counts_candidates": counts["status_counts_candidates"],
            "status_counts_stereoisomers":
                counts["status_counts_stereoisomers"],
            "stereoisomer_units_evaluated": counts["n_isomer_units"],
            "alert_counts": counts["alert_counts"],
            "candidates_skipped_by_max_candidates":
                counts["candidates_skipped_by_max_candidates"],
            "wall_time_s": wall,
            "stability": dict(STABILITY),
            "electronic_structure": dict(ELECTRONIC_STRUCTURE),
            "energy_comparability": ENERGY_COMPARABILITY,
            "method": method,
            "note": ("candidates are never reordered, filtered or re-ranked "
                     "by this tool; each candidate keeps its rank and gains "
                     "a physical_verification block"),
        }
        entry["physical_verification"] = summary
    out_path = Path(args.out)
    if kind == "single":
        out_path.write_text(json.dumps(done[0], indent=2) + "\n")
    elif kind == "array":
        out_path.write_text(json.dumps(done, indent=2) + "\n")
    else:
        with out_path.open("w") as fh:
            for entry in done:
                fh.write(json.dumps(entry) + "\n")
    print(f"responses={len(done)} candidates=" +
          str(sum(len(e.get("candidates", [])) for e in done)) +
          f" converged={totals[CONVERGED]} failed={totals[FAILED]} "
          f"unsupported={totals[UNSUPPORTED]} error={totals[ERROR]} "
          f"isomer_units={n_iso_total} wall_s={wall:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
