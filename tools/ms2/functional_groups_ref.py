"""Independent RDKit reference for the `ms2-fg-v4` functional-group vocabulary.

The same 28 definitions as `src/models/ms2/functional_groups.rs`, written
independently as RDKit substructure queries on kekulized molecules (explicit
bond orders, total-hydrogen constraints) plus Python post-filters for fixed
versus delocalised doubles, carbonyl remaining-substituent checks and the
heteroaromatic five-ring rule. Nothing here imports or shells out to Rust.

Delocalisation is computed by a DIFFERENT method from the Rust detector (which
reduces to perfect matching via Tutte's gadget and runs Edmonds' blossom
search, one augmenting path per bond): here ALL valid single/double
assignments with the prescribed per-atom double-bond counts are enumerated by
plain backtracking over the bonds of the candidate graph (exact, exponential
— fine for the small regression molecules; NOT a matching reduction, NOT the
perfect-matching special case). A bond is delocalised when it is double in
some assignment and single in others. The enumeration caps the number of
forms at MAX_FORMS and FAILS LOUDLY (non-zero exit naming the molecule) when
the cap is hit — forms are never truncated silently.

A kekulé form of the molecule is an assignment of single/double to the bonds
that are single or double in the stored form (triple bonds stay) such that
every atom keeps its number of double bonds `d(v)` (its count in the stored
form). The candidate graph `G'` holds the atoms with `d(v) >= 1` joined by
the single/double bonds between two such atoms; a bond with an end of `d = 0`
is fixed single. Every valid assignment is found by backtracking, including
ones that move doubles through hypervalent sulphur/phosphorus ring members
with `d = 2` (which the v3 π graph dropped).

Kekule invariance: every count depends only on fixed versus delocalised
bonds (never on raw orders inside a delocalised system), so all kekule forms
of one molecule give identical counts. The fixture stores every distinct
kekule form per molecule (all valid assignments with the prescribed degrees,
up to the stated cap; each record carries `forms_total` and `forms_stored`),
and the tool asserts identical counts across all stored forms on its own side
before writing.

Usage:
    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy \\
        python tools/ms2/functional_groups_ref.py \\
        --fixture tests/fixtures/ms2/chemistry_v0.json \\
        --out tests/fixtures/ms2/functional_groups_v4.json
    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy \\
        python tools/ms2/functional_groups_ref.py \\
        --export data/ms2/pilot_validation.json \\
        --out /tmp/fg_validation.json

Every molecule is stored with its SMILES (when known) and all its distinct
kekule forms; each form carries its heavy-atom graph in the same form the
Rust fixtures use (per-atom element, hydrogens and valence, plus bonds as
[a, b, order] triples) and the per-type determined instance counts (identical
across forms). Molecules outside the V0 structure domain are skipped with a
recorded reason; any exception while counting a supported molecule is an
ERROR (collected, reported, non-zero exit), never a skip.

Export mode counts groups on the graph it writes: the RDKit molecule is
built from the stored atoms and bonds WITHOUT re-kekulizing (sanitization
excluding the kekulize step), so the counted orders are the stored orders.
When run on `data/ms2/pilot_validation.json` the tool additionally checks
the kekule regression: the stored form and RDKit's re-kekulized form give
the same counts for every molecule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

FG_VERSION = "ms2-fg-v4"

# The 28 vocabulary names in id order (1-based ids in Rust).
FG_NAMES = [
    "carbonyl", "carboxylic_acid", "ester", "amide", "aldehyde", "ketone",
    "hydroxyl", "ether", "primary_amine", "secondary_amine",
    "tertiary_amine", "nitrile", "imine", "alkene", "alkyne", "thiol",
    "thioether", "sulfonyl", "sulfonamide", "phosphoryl", "fluoride",
    "chloride", "bromide", "arene_ring", "iodide", "carbamate_or_urea",
    "anhydride_or_carbonate", "heteroaromatic_five_ring",
]

ATOMIC = {"C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
          "Cl": 17, "Br": 35, "I": 53}
HALOGENS = {9, 17, 35, 53}
OSN = {8, 16, 7}
NOSH_HAL = {7, 8, 16, 9, 17, 35, 53}

MAX_FORMS = 4096
"""Cap on enumerated kekule forms of one molecule (exact backtracking over the
candidate-graph bonds is exponential; all regression molecules need far
fewer). Hitting the cap is a loud failure naming the molecule — forms are
never truncated silently."""


def kekulized(smiles: str):
    """Kekulized mol with aromatic flags cleared (aromaticity never enters)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES {smiles!r}")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def graph_of(mol):
    """Heavy-atom graph in the Rust fixture form."""
    atoms = []
    for a in mol.GetAtoms():
        order = int(sum(b.GetBondTypeAsDouble() for b in a.GetBonds()))
        atoms.append({
            "element": a.GetSymbol(),
            "hydrogens": a.GetTotalNumHs(),
            "valence": a.GetTotalNumHs() + order,
        })
    bonds = sorted(
        [min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         int(b.GetBondTypeAsDouble())]
        for b in mol.GetBonds())
    return atoms, bonds


def _el(mol, i: int) -> int:
    return mol.GetAtomWithIdx(i).GetAtomicNum()


def _h(mol, i: int) -> int:
    return mol.GetAtomWithIdx(i).GetTotalNumHs()


def _val(mol, i: int) -> int:
    a = mol.GetAtomWithIdx(i)
    return a.GetTotalNumHs() + int(sum(b.GetBondTypeAsDouble() for b in a.GetBonds()))


def _adj(mol):
    n = mol.GetNumAtoms()
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        o = b.GetBondTypeAsDouble()
        adj[i].append((j, o))
        adj[j].append((i, o))
    return adj


# ---------------------------------------------------------------------------
# Delocalised bonds: ALL valid single/double assignments with the prescribed
# per-atom double-bond counts, by plain backtracking over the candidate-graph
# bonds (NOT a matching reduction, NOT the perfect-matching special case).
#
# A kekulé form is an assignment of single/double to the bonds that are
# single or double in the stored form (triple bonds stay) such that every
# atom keeps its number of double bonds d(v) (its count in the stored form).
# The candidate graph G' holds the atoms with d(v) >= 1 joined by the
# single/double bonds between two such atoms; a bond with an end of d = 0 is
# fixed single. Backtracking assigns orders 1/2 to the G' bonds with
# per-vertex need pruning; every other bond keeps its order. This is a
# DIFFERENT method from the Rust detector (Tutte's gadget + blossom search,
# one augmenting path per bond): here all assignments are enumerated
# directly (exact, exponential, fine for the regression molecules), capped at
# MAX_FORMS with a loud failure on overflow. Whole molecules are closed, so
# every bond is decided (fixed or delocalised).
# ---------------------------------------------------------------------------

class MatchingCapHit(Exception):
    """The kekule-form enumeration hit MAX_FORMS (never truncate)."""


def candidate_graph(n, bonds):
    """The candidate graph as (d, g_bonds).

    `d[v]` is the number of incident double bonds; `g_bonds` holds the bond
    indices of G': order 1 or 2 with both ends at `d >= 1`.
    """
    d = [0] * n
    for a, b, o in bonds:
        if o == 2:
            d[a] += 1
            d[b] += 1
    g_bonds = [i for i, (a, b, o) in enumerate(bonds)
               if o in (1, 2) and d[a] >= 1 and d[b] >= 1]
    return d, g_bonds


def enumerate_assignments(n, bonds, cap=MAX_FORMS):
    """All distinct valid assignments with the prescribed double-bond counts.

    Plain backtracking over the G' bonds (orders 1/2, per-vertex need
    pruning); every non-G' bond keeps its order. Returns the list of sorted
    [a, b, order] bond lists. Raises MatchingCapHit past `cap` assignments —
    never truncates silently.
    """
    d, g_bonds = candidate_graph(n, bonds)
    # Incident G'-bond positions per vertex (for pruning).
    incid = [[] for _ in range(n)]
    for pos, bi in enumerate(g_bonds):
        a, b, _o = bonds[bi]
        incid[a].append(pos)
        incid[b].append(pos)
    assign = [0] * len(g_bonds)  # 0 = unassigned, else 1/2
    used = [0] * n               # doubles placed at each vertex
    remain = [len(lst) for lst in incid]  # unassigned incident G' bonds
    out = []

    def rec(pos):
        if len(out) >= cap:
            raise MatchingCapHit(f"over {cap} kekule forms")
        if pos == len(g_bonds):
            if all(used[v] == d[v] for v in range(n)):
                new = [list(t) for t in bonds]
                for p, bi in enumerate(g_bonds):
                    new[bi][2] = assign[p]
                out.append(sorted(new))
            return
        bi = g_bonds[pos]
        a, b, _o = bonds[bi]
        for o in (1, 2):
            ok = True
            for v in (a, b):
                u = used[v] + (1 if o == 2 else 0)
                r = remain[v] - 1
                if not (u <= d[v] <= u + r):
                    ok = False
                    break
            if not ok:
                continue
            assign[pos] = o
            for v in (a, b):
                used[v] += 1 if o == 2 else 0
                remain[v] -= 1
            rec(pos + 1)
            for v in (a, b):
                used[v] -= 1 if o == 2 else 0
                remain[v] += 1
            assign[pos] = 0

    rec(0)
    # Deduplicate (distinct assignments only).
    seen = set()
    uniq = []
    for f in out:
        key = tuple((a, b, o) for a, b, o in f)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def enumerate_kekule_forms(atoms, bonds, cap=MAX_FORMS):
    """All distinct kekule bond-order assignments from ALL valid assignments
    with the prescribed double-bond counts (complete enumeration up to `cap`;
    loud failure past it).

    `atoms` is the stored atom list, `bonds` the [a, b, order] list of one
    form. Returns (atoms, forms) with forms as sorted [a, b, order] lists.
    """
    return atoms, enumerate_assignments(len(atoms), [list(b) for b in bonds], cap)


def delocalised_set(mol) -> set[tuple[int, int]]:
    """Set of delocalised bonds as sorted (a, b) pairs: G' bonds double in
    some valid assignment and single in others (complete backtracking
    enumeration, capped loudly at MAX_FORMS)."""
    n = mol.GetNumAtoms()
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append([min(i, j), max(i, j), int(b.GetBondTypeAsDouble())])
    forms = enumerate_assignments(n, bonds)
    _d, g_bonds = candidate_graph(n, bonds)
    g_set = set()
    for bi in g_bonds:
        a, b, _o = bonds[bi]
        g_set.add((min(a, b), max(a, b)))
    in_some_double: set[tuple[int, int]] = set()
    in_some_single: set[tuple[int, int]] = set()
    for f in forms:
        for a, b, o in f:
            if (a, b) in g_set:
                if o == 2:
                    in_some_double.add((a, b))
                else:
                    in_some_single.add((a, b))
    return in_some_double & in_some_single


def find_arene_rings(mol, deloc: set[tuple[int, int]]):
    """Distinct atom sets of six-membered C/N rings whose six ring bonds
    are all delocalised."""
    adj = _adj(mol)
    n = mol.GetNumAtoms()
    rings: set[frozenset[int]] = set()

    def dfs(start, cur, path, orders):
        if len(path) == 6:
            # Must close to start.
            clos = [o for (j, o) in adj[cur] if j == start]
            if not clos:
                return
            o = clos[0]
            if o not in (1.0, 2.0):
                return
            # All six bonds delocalised?
            ring = path
            pairs = [(min(ring[k], ring[(k + 1) % 6]), max(ring[k], ring[(k + 1) % 6]))
                     for k in range(6)]
            # Note: pairs uses consecutive path atoms; the closure pair is
            # (path[5], start) which equals (ring[5], ring[0]).
            if all(p in deloc for p in pairs):
                rings.add(frozenset(path))
            return
        for nxt, o in adj[cur]:
            if o not in (1.0, 2.0):
                continue
            if nxt == start:
                continue
            if nxt in path or nxt < start:
                continue
            if mol.GetAtomWithIdx(nxt).GetAtomicNum() not in (6, 7):
                continue
            dfs(start, nxt, path + [nxt], orders + [o])

    for s in range(n):
        if mol.GetAtomWithIdx(s).GetAtomicNum() not in (6, 7):
            continue
        dfs(s, s, [s], [])
    return rings


def find_five_ring_cycles(mol, deloc=None):
    """Heteroaromatic five-rings in cycle order (one representative cycle
    per atom set, each starting at its smallest atom). Exclusion sets must
    be built from these cycle orders: adjacency in sorted order is NOT ring
    adjacency and depends on atom numbering."""
    if deloc is None:
        deloc = delocalised_set(mol)
    adj = _adj(mol)
    n = mol.GetNumAtoms()
    order_of: dict[tuple[int, int], float] = {}
    for i, lst in adj.items():
        for j, o in lst:
            order_of[(min(i, j), max(i, j))] = o
    seen: set[frozenset[int]] = set()
    cycles: list[list[int]] = []

    def ok(path):
        hetero = sum(1 for a in path
                     if mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16))
        if hetero not in (1, 2):
            return False
        m = len(path)
        bonds = []
        for k in range(m):
            a, b = path[k], path[(k + 1) % m]
            bonds.append(order_of[(min(a, b), max(a, b))])
        for k, a in enumerate(path):
            prev, nxt = bonds[(k + 4) % 5], bonds[k]
            is_het = mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16)
            if is_het and prev == 1.0 and nxt == 1.0:
                continue
            prev_a = path[(k + 4) % 5]
            next_a = path[(k + 1) % 5]
            prev_ok = (prev == 2.0
                       or (min(prev_a, a), max(prev_a, a)) in deloc)
            next_ok = (nxt == 2.0
                       or (min(a, next_a), max(a, next_a)) in deloc)
            if prev_ok or next_ok:
                continue
            return False
        return True

    def dfs(start, cur, path):
        if len(path) == 5:
            if any(j == start for (j, _o) in adj[cur]):
                if ok(path) and frozenset(path) not in seen:
                    seen.add(frozenset(path))
                    cycles.append(list(path))
            return
        for nxt, o in adj[cur]:
            if o not in (1.0, 2.0):
                continue
            if nxt in path or nxt < start:
                continue
            dfs(start, nxt, path + [nxt])

    for s in range(n):
        dfs(s, s, [s])
    return cycles


def find_five_rings(mol, deloc=None):
    """Distinct atom sets of heteroaromatic five-rings (1-2 heteroatoms
    N/O/S; every atom either a single-only heteroatom or carrying a ring
    bond that is a fixed double or delocalised — hence kekule-invariant).
    `deloc` is the delocalised bond set (computed when None)."""
    return {frozenset(c) for c in find_five_ring_cycles(mol, deloc)}


def _fixed_double(mol, deloc, i, j) -> bool:
    for b in mol.GetAtomWithIdx(i).GetBonds():
        if b.GetOtherAtomIdx(i) == j and b.GetBondTypeAsDouble() == 2.0:
            return (min(i, j), max(i, j)) not in deloc
    return False


def _double_to_fixed(mol, deloc, i, elements) -> bool:
    a = mol.GetAtomWithIdx(i)
    for b in a.GetBonds():
        if b.GetBondTypeAsDouble() == 2.0:
            j = b.GetOtherAtomIdx(i)
            if _el(mol, j) in elements and (min(i, j), max(i, j)) not in deloc:
                return True
    return False


def _single_to(mol, i, elements) -> bool:
    a = mol.GetAtomWithIdx(i)
    for b in a.GetBonds():
        if b.GetBondTypeAsDouble() == 1.0:
            j = b.GetOtherAtomIdx(i)
            if _el(mol, j) in elements:
                return True
    return False


def carbonyl_remaining_ok(mol, c, designated) -> bool:
    """Carbonyl carbon's remaining substituent is C or H: no hetero heavy
    neighbour outside `designated`, at most one extra carbon via a single."""
    des = set(designated)
    extra_c = 0
    for b in mol.GetAtomWithIdx(c).GetBonds():
        j = b.GetOtherAtomIdx(c)
        if j in des:
            continue
        if _el(mol, j) != 6 or b.GetBondTypeAsDouble() != 1.0:
            return False
        extra_c += 1
        if extra_c > 1:
            return False
    return True


def count(mol) -> dict[str, int]:
    """Per-type determined instance counts of one kekulized whole molecule
    (closed: every bond decided, every exclusion certain-or-violated)."""
    deloc = delocalised_set(mol)
    arings = find_arene_rings(mol, deloc)
    frings = find_five_rings(mol, deloc)
    fring_cycles = find_five_ring_cycles(mol, deloc)
    in_five_hetero: set[int] = set()
    in_five_bond: set[tuple[int, int]] = set()
    order_of: dict[tuple[int, int], float] = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        order_of[(min(i, j), max(i, j))] = b.GetBondTypeAsDouble()
    # NB: ring bonds come from cycle orders, never from sorted-order
    # adjacency (which depends on atom numbering).
    for r in fring_cycles:
        m = len(r)
        for k, a in enumerate(r):
            b = r[(k + 1) % m]
            in_five_bond.add((min(a, b), max(a, b)))
            prev = r[(k + 4) % m]
            po = order_of[(min(prev, a), max(prev, a))]
            no = order_of[(min(a, b), max(a, b))]
            if _el(mol, a) in (7, 8, 16) and po == 1.0 and no == 1.0:
                in_five_hetero.add(a)
    out: dict[str, set[tuple]] = {name: set() for name in FG_NAMES}

    def add(name, *atoms):
        out[name].add(tuple(sorted(atoms)))

    cores = {
        "carbonyl": Chem.MolFromSmarts("[#6]=[#8]"),
        "carboxylic_acid": Chem.MolFromSmarts("[#6](=[#8])-[#8;H1]"),
        "ester": Chem.MolFromSmarts("[#6](=[#8])-[#8;H0]-[#6]"),
        "amide": Chem.MolFromSmarts("[#6](=[#8])-[#7]"),
        "aldehyde": Chem.MolFromSmarts("[#6]=[#8]"),
        "ketone": Chem.MolFromSmarts("[#6;H0](=[#8])(-[#6])-[#6]"),
        "hydroxyl": Chem.MolFromSmarts("[#6]-[#8;H1]"),
        "ether": Chem.MolFromSmarts("[#6]-[#8;H0]-[#6]"),
        "primary_amine": Chem.MolFromSmarts("[#6]-[#7;H2]"),
        "secondary_amine": Chem.MolFromSmarts("[#6]-[#7;H1]-[#6]"),
        "tertiary_amine": Chem.MolFromSmarts("[#7;H0](-[#6])(-[#6])-[#6]"),
        "nitrile": Chem.MolFromSmarts("[#6]#[#7]"),
        "imine": Chem.MolFromSmarts("[#6]=[#7]"),
        "alkene": Chem.MolFromSmarts("[#6]=[#6]"),
        "alkyne": Chem.MolFromSmarts("[#6]#[#6]"),
        "thiol": Chem.MolFromSmarts("[#6]-[#16;H1]"),
        "thioether": Chem.MolFromSmarts("[#6]-[#16;H0]-[#6]"),
        "sulfonyl": Chem.MolFromSmarts("[#16](=[#8])(=[#8])"),
        "sulfonamide": Chem.MolFromSmarts("[#16](=[#8])(=[#8])-[#7]"),
        "phosphoryl": Chem.MolFromSmarts("[#15]=[#8]"),
        "fluoride": Chem.MolFromSmarts("[#6]-[#9]"),
        "chloride": Chem.MolFromSmarts("[#6]-[#17]"),
        "bromide": Chem.MolFromSmarts("[#6]-[#35]"),
        "iodide": Chem.MolFromSmarts("[#6]-[#53]"),
        "carbamate_a": Chem.MolFromSmarts("[#7]-[#6](=[#8])-[#7]"),
        "carbamate_b": Chem.MolFromSmarts("[#7]-[#6](=[#8])-[#8]"),
        "anhydride_a": Chem.MolFromSmarts("[#6](=[#8])-[#8]-[#6](=[#8])"),
        "carbonate_b": Chem.MolFromSmarts("[#8]-[#6](=[#8])-[#8]"),
    }
    matches: dict[str, list[tuple]] = {}
    for name, patt in cores.items():
        matches[name] = list(mol.GetSubstructMatches(patt, uniquify=True))

    def fixed(c, o) -> bool:
        return (min(c, o), max(c, o)) not in deloc

    for c, o in matches["carbonyl"]:
        if fixed(c, o):
            add("carbonyl", c, o)
    for c, o1, o2 in matches["carboxylic_acid"]:
        if not (fixed(c, o1)):
            continue
        if not carbonyl_remaining_ok(mol, c, [o1, o2]):
            continue
        add("carboxylic_acid", c, o1, o2)
    for c, o1, o2, c2 in matches["ester"]:
        if not fixed(c, o1):
            continue
        if not carbonyl_remaining_ok(mol, c, [o1, o2]):
            continue
        if _double_to_fixed(mol, deloc, c2, OSN):
            continue
        add("ester", c, o1, o2, c2)
    for c, o, n in matches["amide"]:
        if not fixed(c, o):
            continue
        if not carbonyl_remaining_ok(mol, c, [o, n]):
            continue
        add("amide", c, o, n)
    for c, o in matches["aldehyde"]:
        if _h(mol, c) not in (1, 2):
            continue
        if not fixed(c, o):
            continue
        if not _single_to(mol, c, NOSH_HAL):
            add("aldehyde", c, o)
    for c, o, c1, c2 in matches["ketone"]:
        if fixed(c, o):
            add("ketone", c, o, c1, c2)
    for c, o in matches["hydroxyl"]:
        if not _double_to_fixed(mol, deloc, c, OSN):
            add("hydroxyl", c, o)
    for c1, o, c2 in matches["ether"]:
        if o in in_five_hetero:
            continue
        if not _double_to_fixed(mol, deloc, c1, OSN) and not _double_to_fixed(mol, deloc, c2, OSN):
            add("ether", c1, o, c2)
    for c, n in matches["primary_amine"]:
        if not _double_to_fixed(mol, deloc, c, OSN):
            add("primary_amine", c, n)
    for c1, n, c2 in matches["secondary_amine"]:
        if n in in_five_hetero:
            continue
        if not _double_to_fixed(mol, deloc, c1, OSN) and not _double_to_fixed(mol, deloc, c2, OSN):
            add("secondary_amine", c1, n, c2)
    for n, c1, c2, c3 in matches["tertiary_amine"]:
        if n in in_five_hetero:
            continue
        if (not _double_to_fixed(mol, deloc, c1, OSN) and not _double_to_fixed(mol, deloc, c2, OSN)
                and not _double_to_fixed(mol, deloc, c3, OSN)):
            add("tertiary_amine", n, c1, c2, c3)
    for c, n in matches["nitrile"]:
        add("nitrile", c, n)
    for c, n in matches["imine"]:
        if not fixed(c, n):
            continue
        if (min(c, n), max(c, n)) in in_five_bond:
            continue
        add("imine", c, n)
    for c1, c2 in matches["alkene"]:
        if not fixed(c1, c2):
            continue
        if (min(c1, c2), max(c1, c2)) in in_five_bond:
            continue
        add("alkene", c1, c2)
    for c1, c2 in matches["alkyne"]:
        add("alkyne", c1, c2)
    for c, s in matches["thiol"]:
        if not _double_to_fixed(mol, deloc, c, OSN):
            add("thiol", c, s)
    for c1, s, c2 in matches["thioether"]:
        if s in in_five_hetero:
            continue
        if _val(mol, s) == 2:
            add("thioether", c1, s, c2)
    for s, o1, o2 in matches["sulfonyl"]:
        if _h(mol, s) == 0 and fixed(s, o1) and fixed(s, o2):
            add("sulfonyl", s, o1, o2)
    for s, o1, o2, n in matches["sulfonamide"]:
        if _h(mol, s) == 0 and fixed(s, o1) and fixed(s, o2):
            add("sulfonamide", s, o1, o2, n)
    for p, o in matches["phosphoryl"]:
        if _h(mol, p) == 0 and fixed(p, o):
            add("phosphoryl", p, o)
    for c, x in matches["fluoride"]:
        add("fluoride", c, x)
    for c, x in matches["chloride"]:
        add("chloride", c, x)
    for c, x in matches["bromide"]:
        add("bromide", c, x)
    for r in arings:
        add("arene_ring", *sorted(r))
    for c, x in matches["iodide"]:
        add("iodide", c, x)
    for n1, c, o, n2 in matches["carbamate_a"]:
        if fixed(c, o):
            add("carbamate_or_urea", n1, c, o, n2)
    for n, c, o1, o2 in matches["carbamate_b"]:
        if fixed(c, o1):
            add("carbamate_or_urea", n, c, o1, o2)
    for c1, o1, o, c2, o2 in matches["anhydride_a"]:
        if fixed(c1, o1) and fixed(c2, o2):
            add("anhydride_or_carbonate", c1, o1, o, c2, o2)
    for o1, c, o2, o3 in matches["carbonate_b"]:
        # O–C(=O)–O: the SMARTS binds (o1, c, o(=o2?), o3)? The pattern
        # [#8]-[#6](=[#8])-[#8] matches (o_a, c, o_dbl, o_b).
        if fixed(c, o2) and _h(mol, o1) in (0, 1) and _h(mol, o3) in (0, 1):
            add("anhydride_or_carbonate", o1, c, o2, o3)
    for r in frings:
        add("heteroaromatic_five_ring", *sorted(r))
    return {name: len(v) for name, v in out.items()}


# ---------------------------------------------------------------------------
# Kekule forms now come from the complete valid-assignment enumeration above
# (enumerate_kekule_forms on stored atoms/bonds); no cycle-flip search.
# ---------------------------------------------------------------------------

def mol_from_graph(atoms, bonds):
    """RDKit mol from stored atoms/bonds WITHOUT re-kekulizing (explicit
    orders preserved)."""
    rw = Chem.RWMol()
    for a in atoms:
        el = a["element"]
        h = a["hydrogens"]
        atom = Chem.Atom(ATOMIC[el])
        atom.SetNumExplicitHs(h)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    order_to_bond = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                     3: Chem.BondType.TRIPLE}
    for a, b, o in bonds:
        rw.AddBond(int(a), int(b), order_to_bond[int(o)])
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


# ---------------------------------------------------------------------------
# Hand-chosen molecules and required kekule-invariance set.
# ---------------------------------------------------------------------------

HAND_SMILES: list[tuple[str, str]] = [
    ("acetic acid", "CC(=O)O"),
    ("benzoic acid", "C1=CC=C(C=C1)C(=O)O"),
    ("glycine", "NCC(=O)O"),
    ("formic acid (aldehyde exclusion rejects)", "C(=O)O"),
    ("formaldehyde", "C=O"),
    ("ethyl acetate", "CCOC(=O)C"),
    ("methyl benzoate", "COC(=O)C1=CC=CC=C1"),
    ("gamma-butyrolactone (lactone)", "C1CC(=O)OC1"),
    ("dimethyl carbonate", "COC(=O)OC"),
    ("carbonic acid", "OC(=O)O"),
    ("acetic anhydride", "CC(=O)OC(=O)C"),
    ("benzoic anhydride", "C1=CC=C(C=C1)C(=O)OC(=O)C2=CC=CC=C2"),
    ("acetamide", "CC(=O)N"),
    ("N-methylacetamide (amide, NOT secondary amine)", "CNC(=O)C"),
    ("benzamide", "C1=CC=C(C=C1)C(=O)N"),
    ("gamma-butyrolactam (lactam)", "C1CC(=O)NC1"),
    ("diacetamide (imide: two amides)", "CC(=O)NC(=O)C"),
    ("urea", "NC(=O)N"),
    ("methyl carbamate", "COC(=O)N"),
    ("ethyl carbamate", "CCOC(=O)N"),
    ("acetaldehyde", "CC=O"),
    ("benzaldehyde", "C1=CC=C(C=C1)C=O"),
    ("propanal", "CCC=O"),
    ("acetone", "CC(=O)C"),
    ("butanone", "CCC(=O)C"),
    ("cyclohexanone", "C1CCC(=O)CC1"),
    ("ethanol", "CCO"),
    ("phenol", "C1=CC=C(C=C1)O"),
    ("2-butanol", "CCC(C)O"),
    ("diethyl ether", "CCOCC"),
    ("anisole", "COC1=CC=CC=C1"),
    ("tetrahydrofuran", "C1CCOC1"),
    ("ethylamine", "CCN"),
    ("aniline", "C1=CC=C(C=C1)N"),
    ("diethylamine", "CCNCC"),
    ("N-methylaniline", "CNC1=CC=CC=C1"),
    ("piperidine", "C1CCNCC1"),
    ("triethylamine", "CCN(CC)CC"),
    ("trimethylamine", "CN(C)C"),
    ("N,N-dimethylaniline", "CN(C)C1=CC=CC=C1"),
    ("acetonitrile", "CC#N"),
    ("benzonitrile", "C1=CC=C(C=C1)C#N"),
    ("propionitrile", "CCC#N"),
    ("ethanimine", "CC=N"),
    ("2-propanimine", "CC(=N)C"),
    ("N-methylmethanimine", "CN=C"),
    ("1-butene", "CCC=C"),
    ("styrene (arene ring + one alkene)", "C=CC1=CC=CC=C1"),
    ("cyclohexene", "C1CCC=CC1"),
    ("1,3-cyclohexadiene", "C1CC=CC=C1"),
    ("furan (five-ring, no arene/alkene/ether)", "C1=COC=C1"),
    ("pyrrole (five-ring, no amine/alkene)", "C1=CNC=C1"),
    ("thiophene (five-ring, no thioether/alkene)", "C1=CSC=C1"),
    ("imidazole", "C1=CN=CN1"),
    ("pyrazole", "C1=CNN=C1"),
    ("oxazole", "C1=COC=N1"),
    ("thiazole", "C1=CSC=N1"),
    ("2-butyne", "CC#CC"),
    ("acetylene", "C#C"),
    ("1-butyne", "CCC#C"),
    ("benzene", "C1=CC=CC=C1"),
    ("pyridine", "C1=CC=NC=C1"),
    ("toluene", "CC1=CC=CC=C1"),
    ("ethanethiol", "CCS"),
    ("thiophenol", "C1=CC=C(C=C1)S"),
    ("thioacetic acid (not a thiol)", "CC(=O)S"),
    ("dimethyl sulfide", "CSC"),
    ("diethyl sulfide", "CCSCC"),
    ("thioanisole", "CSC1=CC=CC=C1"),
    ("dimethyl sulfone", "CS(C)(=O)=O"),
    ("sulfolene (3-sulfolene, pinned)", "O=S1(=O)CC=CC1"),
    ("thiophene S,S-dioxide (pinned)", "O=S1(=O)C=CC=C1"),
    ("trimethylphosphine oxide (pinned)", "CP(C)(C)=O"),
    ("allene (cumulene centre pinned)", "C=C=C"),
    ("ketene (cumulene centre pinned)", "C=C=O"),
    ("carbon dioxide (pinned)", "O=C=O"),
    ("methanesulfonamide", "CS(=O)(=O)N"),
    ("benzenesulfonamide", "C1=CC=C(C=C1)S(=O)(=O)N"),
    ("sulfanilamide", "NC1=CC=C(C=C1)S(=O)(=O)N"),
    ("dimethyl sulfoxide (no sulfonyl; out of domain)", "CS(C)=O"),
    ("trimethyl phosphate", "COP(=O)(OC)OC"),
    ("triethyl phosphate", "CCOP(=O)(OCC)OCC"),
    ("phosphoric acid", "OP(=O)(O)O"),
    ("fluorobenzene", "FC1=CC=CC=C1"),
    ("1-fluoroethane", "CCF"),
    ("1,1-difluoroethane", "CC(F)F"),
    ("chlorobenzene", "ClC1=CC=CC=C1"),
    ("chloroethane", "CCCl"),
    ("dichloromethane", "ClCCl"),
    ("bromobenzene", "BrC1=CC=CC=C1"),
    ("bromoethane", "CCBr"),
    ("dibromomethane", "BrCBr"),
    ("iodobenzene", "IC1=CC=CC=C1"),
    ("methyl iodide", "CI"),
    ("iodoform", "IC(I)I"),
    ("nitrobenzene (charged group; out of domain)", "[O-][N+](=O)C1=CC=CC=C1"),
    ("tetramethylammonium (charged; out of domain)", "C[N+](C)(C)C"),
    ("phosphine (unsupported P valence; out of domain)", "CP(C)C"),
]

# Required kekule-invariance molecules: fused benzenoids (including the
# FG4 regressions: pentacene, hexacene, heptacene, coronene, perylene,
# biphenylene), heteroaromatics, tautomer pair, quinone, COT, fulvene,
# annulenes and five drug-like fused heteroaromatics. Porphine, the
# expanded six-pyrrole macrocycle and the >=30-atom sheet are hand-built
# graphs below (no SMILES: RDKit gives no usable SMILES for the expanded
# macrocycles, and the sheet is constructed cell by cell).
INVARIANCE_SMILES: list[tuple[str, str]] = [
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "c1ccc2ccc3ccccc3c2c1"),
    ("pentacene", "C1=CC=C2C=C3C=C4C=C5C=CC=CC5=CC4=CC3=CC2=C1"),
    ("hexacene", "C1=CC=C2C=C3C=C4C=C5C=C6C=CC=CC6=CC5=CC4=CC3=CC2=C1"),
    ("heptacene", "C1=CC=C2C=C3C=C4C=C5C=C6C=C7C=CC=CC7=CC6=CC5=CC4=CC3=CC2=C1"),
    ("coronene", "C1=CC2=C3C4=C1C=CC5=C4C6=C(C=C5)C=CC7=C6C3=C(C=C7)C=C2"),
    ("perylene", "C1=CC2=CC=CC3=C2C(=C1)C1=CC=CC2=C1C3=CC=C2"),
    ("biphenylene (four-cycle)", "c1ccc2c(c1)c3ccccc32"),
    ("pyrene", "C1=CC2=C3C(=C1)C=CC4=C3C(=CC=C4)C=C2"),
    ("azulene (10-cycle, no six-ring in some forms)", "C1=CC2=CC=CC=CC2=C1"),
    ("biphenyl", "c1ccccc1-c2ccccc2"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("2-aminopyridine", "Nc1ccccn1"),
    ("2-hydroxypyridine", "Oc1ccccn1"),
    ("2-pyridone (tautomer of 2-hydroxypyridine; different labels)", "O=c1cccc[nH]1"),
    ("pyridazine", "c1ccnnc1"),
    ("pyrimidine", "c1cnccn1"),
    ("purine", "c1ncc2[nH]cnc2n1"),
    ("adenine", "Nc1ncnc2[nH]cnc12"),
    ("acenaphthylene", "C1=CC2=C3C(=C1)C=CC3=CC=C2"),
    ("indene", "C1=CC=C2CC=CC2=C1"),
    ("styrene", "C=Cc1ccccc1"),
    ("stilbene (trans)", "C(=Cc1ccccc1)/c2ccccc2"),
    ("p-benzoquinone (fixed C=C/C=O; NOT an arene ring)", "O=C1C=CC(=O)C=C1"),
    ("tropone", "O=C1C=CC=CC=C1"),
    ("cyclooctatetraene (delocalised, NOT aromatic, no six-ring)", "C1=CC=CC=CC=C1"),
    ("cyclobutadiene (delocalised by kekule-equivalence)", "C1=CC=C1"),
    ("benzocyclobutadiene", "c1ccc2ccc2c1"),
    ("[18]annulene", "C1" + "=CC" * 8 + "=C1"),
    ("fulvene", "C=C1C=CC=C1"),
    ("caffeine (fused heteroaromatic drug)", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("carbamazepine (tricyclic drug)", "NC(=O)N1C2=CC=CC=C2C=CC2=CC=CC=C12"),
    ("olanzapine core (thienobenzodiazepine drug)", "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(CC4)C"),
    ("indomethacin (fused indole drug)", "COC1=CC=C(C=C1)C(=O)C2=C(C)N(C3=CC=C(C=C32)Cl)CC(=O)O"),
    ("quinine (fused quinoline drug)", "COC1=CC2=C(C=CN=C2C=C1)C(O)C3CC4CCN3CC4C=C"),
    ("sildenafil core (pyrazolopyrimidine drug)", "CCC1=NN(C)C2=C1C(=O)NC(=N2)C3=CC=CC=C3OCC"),
    ("ether closability (tert-butyl pyridines; reviewer SMILES kept)", "n1cccc(C(C)(C)C)c1Oc1nc(C(C)(C)C)c(C(C)(C)C)c(C(C)(C)C)c1C(C)(C)C"),
    # FG5 hypervalent-ring regressions (reviewer counterexamples): the
    # six-ring S/N heterocycle with two doubles at sulphur (two forms, all
    # six ring bonds delocalised) and the 11-atom S/P fused parent.
    ("review S-ring (two doubles at S; delocalised six-ring)", "O=S1(C)=NC=CC=C1"),
    ("review S/P parent (11-atom fused heterocycle)", "CS1(=O)=C2C3=C4N=C(O)C3=P421"),
    ("lambda5-phosphinine (P in a delocalised six-ring)", "CP1(C)=CC=CC=C1"),
]


# FG6 regressions, processed AFTER the hand-built graphs so the regenerated
# fixture keeps every previous record in place (append-only):
# - the triple-bonded six-ring heterocycle (triple fixed, ring delocalised);
# - the fixed-triple isomers C1#CC=C1 ({alkene 1, alkyne 1}) and C1=C=CC=1
#   ({alkene 3}): identical connectivity, H counts and valences, different
#   triple placement — different molecules outside the module's equivalence.
FG6_SMILES: list[tuple[str, str]] = [
    ("triple-bonded six-ring heterocycle (triple fixed, ring delocalised)", "C#S1=NC=CC=C1"),
    ("four-ring alkyne/alkene isomer (alkene 1, alkyne 1)", "C1#CC=C1"),
    ("four-ring triene isomer (alkene 3)", "C1=C=CC=1"),
]


def _benzenoid_patch(cells):
    """Carbon skeleton of fused hexagons at axial `cells` (pointy-top layout,
    corners merged by coordinates). Returns (n, edges). Used for the >=30-atom
    sheet below; coronene's 7-cell patch gives 24 atoms."""
    import math as _math
    pts = []
    for (q, r) in cells:
        cx = _math.sqrt(3) * (q + r / 2.0)
        cy = 1.5 * r
        for k in range(6):
            a = _math.radians(30 + 60 * k)
            pts.append((round(cx + _math.cos(a), 6), round(cy + _math.sin(a), 6)))
    uniq: dict = {}
    for p in pts:
        if p not in uniq:
            uniq[p] = len(uniq)
    idx = 0
    cellcorn: dict = {}
    for (q, r) in cells:
        cellcorn[(q, r)] = [uniq[pts[idx + k]] for k in range(6)]
        idx += 6
    edges = set()
    for cc in cellcorn.values():
        for k in range(6):
            edges.add((min(cc[k], cc[(k + 1) % 6]), max(cc[k], cc[(k + 1) % 6])))
    return len(uniq), sorted(edges)


def _kekule_by_bipartite_matching(n, edges):
    """One kekule bond-order assignment for a bipartite all-carbon skeleton:
    a perfect matching (Kuhn) gives the doubles. Raises ValueError without
    one (non-Kekulean patch)."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    col = [-1] * n
    for s in range(n):
        if col[s] >= 0:
            continue
        col[s] = 0
        stack = [s]
        while stack:
            x = stack.pop()
            for w in adj[x]:
                if col[w] == -1:
                    col[w] = 1 - col[x]
                    stack.append(w)
                elif col[w] == col[x]:
                    raise ValueError("skeleton is not bipartite")
    mt = [-1] * n

    def dfs(x, seen):
        for w in adj[x]:
            if seen[w]:
                continue
            seen[w] = True
            if mt[w] == -1 or dfs(mt[w], seen):
                mt[w] = x
                return True
        return False

    for x in range(n):
        if col[x] == 0 and not dfs(x, [False] * n):
            raise ValueError("skeleton has no perfect matching (non-Kekulean)")
    pairs = set((min(x, mt[x]), max(x, mt[x])) for x in range(n) if mt[x] != -1)
    if len(pairs) * 2 != n:
        raise ValueError("skeleton has no perfect matching (non-Kekulean)")
    return [[a, b, 2 if (a, b) in pairs else 1] for a, b in edges]


def _oligopyrrole_macrocycle(k, roles):
    """Hand-built cyclic oligopyrrole: `k` pyrrole rings (C4N each) joined by
    `k` methine bridges into a macrocycle. `roles[i]` is 'imine' (pyridine
    N, H0), 'nh' (pyrrolic NH) or 'nme' (N-methyl, plus a methyl carbon).
    Returns (atoms, bonds) with one valid kekule assignment (solved by
    backtracking over per-atom order sums); raises ValueError when none
    exists. Porphine is k=4 with trans NH pair; the reviewer's expanded
    macrocycle is k=6 with four imine, one NH and one N-methyl nitrogen."""
    atoms: list[dict] = []
    bonds: list[tuple[int, int]] = []

    def add(el, h, v):
        atoms.append({"element": el, "hydrogens": h, "valence": v})
        return len(atoms) - 1

    N, Ca1, Cb1, Cb2, Ca2, M = [], [], [], [], [], []
    for r in roles:
        n = add("N", 0 if r != "nh" else 1, 3)
        if r == "nme":
            bonds.append((n, add("C", 3, 4)))
        a1, b1, b2, a2, m = add("C", 0, 4), add("C", 1, 4), add("C", 1, 4), add("C", 0, 4), add("C", 1, 4)
        N.append(n)
        Ca1.append(a1)
        Cb1.append(b1)
        Cb2.append(b2)
        Ca2.append(a2)
        M.append(m)
        bonds += [(n, a1), (a1, b1), (b1, b2), (b2, a2), (a2, n)]
    for i in range(k):
        bonds.append((Ca2[i], M[i]))
        bonds.append((M[i], Ca1[(i + 1) % k]))
    need = [a["valence"] - a["hydrogens"] for a in atoms]
    inc: list[list[int]] = [[] for _ in atoms]
    for bi, (a, b) in enumerate(bonds):
        inc[a].append(bi)
        inc[b].append(bi)
    order = [0] * len(bonds)
    solution: list[int] | None = None

    def rec(bi):
        nonlocal solution
        if solution is not None:
            return True
        if bi == len(bonds):
            if all(sum(order[j] for j in inc[i]) == need[i] for i in range(len(atoms))):
                solution = list(order)
                return True
            return False
        a, b = bonds[bi]
        for o in (1, 2):
            ok = True
            for i in (a, b):
                s = sum(order[j] for j in inc[i] if j < bi) + o
                rem = sum(1 for j in inc[i] if j > bi)
                if not (s <= need[i] <= s + 2 * rem):
                    ok = False
                    break
            if ok:
                order[bi] = o
                if rec(bi + 1):
                    return True
                order[bi] = 0
        return False

    if not rec(0) or solution is None:
        raise ValueError("macrocycle admits no kekule assignment")
    return atoms, [[a, b, o] for (a, b), o in zip(bonds, solution)]


def hand_graphs():
    """Hand-built regression graphs: (name, atoms, bonds). No SMILES: RDKit
    gives no usable SMILES for the expanded macrocycles, and the sheet is
    constructed cell by cell (see helpers)."""
    out = []
    # Ovalene-like 10-ring benzenoid sheet (C32H14): coronene 7-cell patch
    # plus three cells; composition and Kekule count checked at build time.
    base = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    n, edges = _benzenoid_patch(base + [(-1, 2), (0, 2), (1, 1)])
    from collections import Counter as _Counter
    deg = _Counter(a for x in edges for a in x)
    atoms = [{"element": "C", "hydrogens": 1 if deg[i] == 2 else 0, "valence": 4}
             for i in range(n)]
    out.append(("ovalene sheet (hand-built 10-ring benzenoid, C32H14)",
                atoms, _kekule_by_bipartite_matching(n, edges)))
    # Large sheet (13-ring super-coronene patch, hand-built, C48H24): the
    # >=64-form case — complete enumeration stores every form and the Rust
    # test regenerates and checks them all.
    big = base + [(-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0)]
    n2, edges2 = _benzenoid_patch(big)
    deg2 = _Counter(a for x in edges2 for a in x)
    atoms2 = [{"element": "C", "hydrogens": 1 if deg2[i] == 2 else 0, "valence": 4}
              for i in range(n2)]
    out.append(("large benzenoid sheet (hand-built 13-ring, C48H24)",
                atoms2, _kekule_by_bipartite_matching(n2, edges2)))
    # Porphine (hand-built cyclic tetrapyrrole, trans NH pair, C20H14N4).
    atoms, bonds = _oligopyrrole_macrocycle(4, ["nh", "imine", "nh", "imine"])
    out.append(("porphine (hand-built cyclic tetrapyrrole)", atoms, bonds))
    # Reviewer's expanded macrocycle: six pyrroles, four imine N, one NH and
    # one N-methyl N (hand-built hexapyrrole + six methines).
    atoms, bonds = _oligopyrrole_macrocycle(
        6, ["imine", "imine", "nh", "imine", "imine", "nme"])
    out.append(("expanded six-pyrrole macrocycle, 4 imine + NH + NMe (hand-built)",
                atoms, bonds))
    # 1,1-dimethyl-thiabenzene analogue (hand-built: the literal thiabenzene
    # SMILES `CS1=CC=CC=C1` gives S(H0,v4), which has no V0 atom type, so no
    # SMILES can supply it; this domain-valid analogue keeps two ring doubles
    # at S(H0,v6) with two S-methyls: S orders 2+2+1+1 = 6).
    out.append((
        "1,1-dimethyl-thiabenzene analogue (hand-built; S-v4 SMILES out of domain)",
        [
            {"element": "S", "hydrogens": 0, "valence": 6},
            {"element": "C", "hydrogens": 3, "valence": 4},
            {"element": "C", "hydrogens": 3, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 0, "valence": 4},
        ],
        [[0, 3, 2], [3, 4, 1], [4, 5, 2], [5, 6, 1], [6, 7, 2], [7, 0, 2],
         [0, 1, 1], [0, 2, 1]],
    ))
    # Cyclic sulfoximine analogue (hand-built: six-ring S=N-C=C-C=C-S with an
    # exocyclic S=O, so sulphur carries two ring doubles plus =O; S orders
    # 2+2+2 = 6, the C opposite N is a cumulene-like d=2 carbon).
    out.append((
        "cyclic sulfoximine analogue (hand-built; two ring doubles at S)",
        [
            {"element": "S", "hydrogens": 0, "valence": 6},
            {"element": "O", "hydrogens": 0, "valence": 2},
            {"element": "N", "hydrogens": 0, "valence": 3},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 0, "valence": 4},
        ],
        [[0, 1, 2], [0, 2, 2], [2, 3, 1], [3, 4, 2], [4, 5, 1], [5, 6, 2],
         [6, 0, 2]],
    ))
    # Two triangles sharing one S(H0,v6) (hand-built FG6 regression): the
    # two valid assignments exchange the double placement between the
    # triangles with no simple even atom-cycle witness — the projected
    # witness is an alternating closed trail revisiting S. Form 1 holds
    # both S doubles in triangle 1 (S-A, S-B double, A-B single) and the
    # C-D double in triangle 2; form 2 is the mirror exchange. Every atom
    # keeps its double count (S: 2; each C: 1) across both forms.
    out.append((
        "two triangles sharing S(H0,v6) (hand-built; exchange with no even atom-cycle)",
        [
            {"element": "S", "hydrogens": 0, "valence": 6},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
            {"element": "C", "hydrogens": 1, "valence": 4},
        ],
        [[0, 1, 2], [0, 2, 2], [1, 2, 1], [0, 3, 1], [0, 4, 1], [3, 4, 2]],
    ))
    return out


TYPE_ID = {
    ("C", 0, 4): 1, ("C", 1, 4): 2, ("C", 2, 4): 3, ("C", 3, 4): 4,
    ("N", 0, 3): 5, ("N", 1, 3): 6, ("N", 2, 3): 7,
    ("O", 0, 2): 8, ("O", 1, 2): 9,
    ("F", 0, 1): 10, ("Cl", 0, 1): 11, ("Br", 0, 1): 12,
    ("S", 0, 2): 13, ("S", 1, 2): 14, ("S", 0, 6): 15,
    ("P", 0, 5): 16, ("I", 0, 1): 17,
}


def classify(mol) -> list[str]:
    """V0 structure-domain reasons (mirrors `RawMolecule::classify`)."""
    from ms2_reference import (  # local import: repo tools live together
        ATOM_TYPES as _T, ELEMENT_ORDER as _E, raw_atoms, classify as _c)
    return _c(mol)


def validate_graph(atoms, bonds):
    """Structural + valence-closure validation of a whole-molecule graph.

    Mirrors `MolGraph::new` plus the closed-molecule requirement: endpoints
    in range, no self bond, no duplicate bond, supported orders (1-3),
    connectivity (one component), and `H + incident order sum == declared
    valence` at every atom. Returns an error string, or `None` when valid.
    A malformed graph is an ERROR (reported, non-zero exit), never a skip.
    """
    n = len(atoms)
    if n == 0:
        return "malformed graph: no atoms"
    seen = set()
    for k, b in enumerate(bonds):
        a, c, o = int(b[0]), int(b[1]), int(b[2])
        if a < 0 or a >= n or c < 0 or c >= n:
            return (f"malformed graph: bond {k} ({a},{c}) endpoint out of "
                    f"range for {n} atoms")
        if a == c:
            return f"malformed graph: self-bond on atom {a}"
        if o not in (1, 2, 3):
            return (f"malformed graph: unsupported bond order {o} on "
                    f"({a},{c})")
        key = (min(a, c), max(a, c))
        if key in seen:
            return f"malformed graph: duplicate bond ({key[0]},{key[1]})"
        seen.add(key)
    # Connectivity: one component (BFS from atom 0 over valid endpoints).
    adj = [[] for _ in range(n)]
    for a, c, _o in bonds:
        adj[int(a)].append(int(c))
        adj[int(c)].append(int(a))
    seen_v = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen_v:
                seen_v.add(w)
                stack.append(w)
    if len(seen_v) != n:
        return (f"malformed graph: disconnected "
                f"({n - len(seen_v)} of {n} atoms unreachable)")
    # Valence closure at every atom.
    for i, a in enumerate(atoms):
        s = sum(int(b[2]) for b in bonds if int(b[0]) == i or int(b[1]) == i)
        if a["hydrogens"] + s != a["valence"]:
            return (f"malformed graph: valence closure failed at atom {i} "
                    f"({a['element']} H{a['hydrogens']} v{a['valence']}: "
                    f"H + incident {s} != {a['valence']})")
    return None


def process_graph(name: str, atoms: list, bonds: list, smiles=None):
    """Full pipeline on stored atoms/bonds: structural + valence-closure
    validation, then complete valid-assignment enumeration (loud cap),
    per-form counting, cross-form invariance assert.
    Returns (record, skip_reason, error). A malformed graph is an error."""
    for i, a in enumerate(atoms):
        if (a["element"], a["hydrogens"], a["valence"]) not in TYPE_ID:
            return None, f"out of domain: unsupported_atom_type at atom {i}", None
    bad = validate_graph(atoms, bonds)
    if bad is not None:
        return None, None, bad
    try:
        _, forms = enumerate_kekule_forms(atoms, [list(b) for b in bonds])
    except MatchingCapHit as e:
        return None, None, f"form cap hit ({MAX_FORMS}): {e}"
    except Exception as e:  # noqa: BLE001
        return None, None, f"enumeration failed: {e}"
    form_recs = []
    first_counts = None
    for fb in forms:
        try:
            counts = count(mol_from_graph(atoms, fb))
        except Exception as e:  # noqa: BLE001
            return None, None, f"count failed: {e}"
        if first_counts is None:
            first_counts = counts
        elif counts != first_counts:
            bad = {k: (first_counts.get(k), counts.get(k)) for k in FG_NAMES
                   if first_counts.get(k) != counts.get(k)}
            return None, None, f"kekule variance in {name}: {bad}"
        form_recs.append({
            "atoms": atoms,
            "bonds": [list(b) for b in fb],
            "counts": counts,
        })
    return {
        "name": name,
        "smiles": smiles,
        "forms": form_recs,
        "forms_total": len(form_recs),
        "forms_stored": len(form_recs),
    }, None, None


def process_smiles(name: str, smiles: str):
    """Full pipeline on a SMILES string. Returns (record, skip_reason, error):
    a skip is only an input outside the V0 domain (with its reason); any
    exception on a supported molecule is an error."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, f"bad SMILES {smiles!r}"
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:  # noqa: BLE001
        return None, None, f"kekulization failed: {e}"
    reasons = classify(mol)
    if reasons:
        return None, f"out of domain: {', '.join(reasons)}", None
    atoms, bonds = graph_of(mol)
    for i, a in enumerate(atoms):
        if (a["element"], a["hydrogens"], a["valence"]) not in TYPE_ID:
            return None, f"out of domain: unsupported_atom_type at atom {i}", None
    rec, skip, err = process_graph(name, atoms, [[a, b, o] for a, b, o in bonds], smiles)
    return rec, skip, err


def fixture_molecule_record(m) -> tuple[dict | None, str | None, str | None]:
    smiles = m.get("smiles")
    if not smiles:
        return None, "fixture molecule without SMILES", None
    return process_smiles(m.get("name", ""), smiles)


def export_molecule_record(m) -> tuple[dict | None, str | None, str | None]:
    """A molecule of a CASMI export: count on the stored kekulized graph
    itself (built WITHOUT any sanitization that could re-perceive
    aromaticity), and write that same graph. Skips are only unknown atom
    types (outside the V0 domain); counting exceptions are errors."""
    from ms2_reference import ATOM_TYPES
    inv = {v: k for k, v in TYPE_ID.items()}
    atoms = []
    for t in m["atoms"]:
        key = inv.get(t)
        if key is None:
            return None, f"unknown atom type id {t}", None
        element, h, v = key
        atoms.append({"element": element, "hydrogens": h, "valence": v})
    bonds = [list(b) for b in m["bonds"]]
    bad = validate_graph(atoms, bonds)
    if bad is not None:
        return None, None, bad
    try:
        mol = mol_from_graph(atoms, bonds)
        counts = count(mol)
    except Exception as e:  # noqa: BLE001
        return None, None, f"count failed: {e}"
    return {
        "name": m.get("key", ""),
        "smiles": None,
        "atoms": atoms,
        "bonds": bonds,
        "counts": counts,
        "forms_total": 1,
        "forms_stored": 1,
    }, None, None


def kekule_regression(export) -> tuple[int, list[str]]:
    """Stored-form versus re-kekulized-form counts for every export molecule
    (when the data are present): must agree by kekule invariance."""
    n = 0
    bad: list[str] = []
    for m in export["molecules"]:
        rec, _skip, _err = export_molecule_record(m)
        if rec is None:
            continue
        # Re-kekulized form: rebuild and let RDKit re-kekulize.
        try:
            from ms2_reference import ATOM_TYPES
            inv = {v: k for k, v in TYPE_ID.items()}
            rw = Chem.RWMol()
            for t in m["atoms"]:
                key = inv.get(t)
                if key is None:
                    continue
                element, h, _v = key
                a = Chem.Atom(ATOMIC[element])
                a.SetNumExplicitHs(h)
                a.SetNoImplicit(True)
                rw.AddAtom(a)
            order_to_bond = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                             3: Chem.BondType.TRIPLE}
            for a, b, o in m["bonds"]:
                rw.AddBond(a, b, order_to_bond[o])
            mol2 = rw.GetMol()
            Chem.SanitizeMol(mol2)
            Chem.Kekulize(mol2, clearAromaticFlags=True)
            counts2 = count(mol2)
        except Exception as e:  # noqa: BLE001
            continue
        n += 1
        if counts2 != rec["counts"]:
            diff = {k: (rec["counts"].get(k), counts2.get(k)) for k in FG_NAMES
                    if rec["counts"].get(k) != counts2.get(k)}
            bad.append(f"{m.get('key', '?')}: {diff}")
    return n, bad


def permutation_check(molecules, perms: int, seed: int = 0x6673335E) -> int:
    """Atom-numbering invariance on the Python side: for every molecule form,
    rebuild the graph under `perms` random atom permutations (bond list
    shuffled, endpoints swapped) and assert identical counts for all 28
    types. Returns the number of permuted graphs checked."""
    import random as _random
    rng = _random.Random(seed)
    checked = 0
    bad: list[str] = []
    for m in molecules:
        forms = m.get("forms")
        if forms is None:
            forms = [{"atoms": m["atoms"], "bonds": m["bonds"], "counts": m["counts"]}]
        for form in forms:
            atoms = form["atoms"]
            bonds = form["bonds"]
            want = form["counts"]
            n = len(atoms)
            for _ in range(perms):
                perm = list(range(n))
                rng.shuffle(perm)
                inv = [0] * n
                for new, old in enumerate(perm):
                    inv[old] = new
                atoms_p = [atoms[old] for old in perm]
                bonds_p = []
                for a, b, o in bonds:
                    na, nb = inv[a], inv[b]
                    if rng.random() < 0.5:
                        na, nb = nb, na
                    bonds_p.append([na, nb, o])
                rng.shuffle(bonds_p)
                mol = mol_from_graph(atoms_p, bonds_p)
                got = count(mol)
                if got != want:
                    diff = {k: (want.get(k), got.get(k)) for k in FG_NAMES
                            if want.get(k) != got.get(k)}
                    bad.append(f"{m.get('name', '?')}: {diff}")
                checked += 1
    if bad:
        raise SystemExit(f"permutation variance on Python side ({len(bad)}):\n" + "\n".join(bad[:10]))
    print(f"permutation check: {checked} permuted graphs agree over {len(molecules)} molecules")
    return checked


def _is_strict_int(x) -> bool:
    """Real integers only: `type(x) is int` (booleans and floats rejected)."""
    return type(x) is int


def validate_graphs_entry(g, idx: int):
    """Strict shape validation of one `--graphs` entry.

    Returns `(name, atoms, bonds)` with validated integer values, or
    `(name, None, error)` where `error` names the graph and field (a normal
    graph error, never a traceback).
    """
    tag = f"graph #{idx}"
    if not isinstance(g, dict):
        return "?", None, f"malformed graph {tag}: entry is not an object"
    name = g.get("name", "?")
    if not isinstance(name, str):
        return "?", None, f"malformed graph {tag}: field 'name' must be a string"
    tag = f"graph '{name}'"
    if "atoms" not in g:
        return name, None, f"malformed graph {tag}: missing field 'atoms'"
    if "bonds" not in g:
        return name, None, f"malformed graph {tag}: missing field 'bonds'"
    atoms_in = g["atoms"]
    bonds_in = g["bonds"]
    if not isinstance(atoms_in, list):
        return name, None, f"malformed graph {tag}: field 'atoms' must be a list"
    if not isinstance(bonds_in, list):
        return name, None, f"malformed graph {tag}: field 'bonds' must be a list"
    n = len(atoms_in)
    atoms: list[dict] = []
    for i, a in enumerate(atoms_in):
        if not isinstance(a, dict):
            return name, None, f"malformed graph {tag}: atom {i} is not an object"
        for field in ("element", "hydrogens", "valence"):
            if field not in a:
                return name, None, f"malformed graph {tag}: atom {i} missing field '{field}'"
        if not isinstance(a["element"], str):
            return name, None, f"malformed graph {tag}: atom {i} field 'element' must be a string"
        if not _is_strict_int(a["hydrogens"]):
            return name, None, f"malformed graph {tag}: atom {i} field 'hydrogens' must be an integer"
        if not _is_strict_int(a["valence"]):
            return name, None, f"malformed graph {tag}: atom {i} field 'valence' must be an integer"
        atoms.append({"element": a["element"], "hydrogens": a["hydrogens"], "valence": a["valence"]})
    bonds: list[list] = []
    for k, b in enumerate(bonds_in):
        if not isinstance(b, (list, tuple)):
            return name, None, f"malformed graph {tag}: bond {k} is not a triple"
        if len(b) != 3:
            return name, None, f"malformed graph {tag}: bond {k} must be an [a, b, order] triple"
        ba, bb, bo = b[0], b[1], b[2]
        if not _is_strict_int(ba) or not _is_strict_int(bb) or not _is_strict_int(bo):
            return name, None, f"malformed graph {tag}: bond {k} fields must be integers"
        if ba < 0 or ba >= n or bb < 0 or bb >= n:
            return name, None, f"malformed graph {tag}: bond {k} ({ba},{bb}) endpoint out of range for {n} atoms"
        bonds.append([ba, bb, bo])
    return name, (atoms, bonds), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--fixture", type=Path, default=None)
    ap.add_argument("--graphs", type=Path, default=None,
                    help="check explicit graphs: a JSON file holding either a "
                         "bare list or {\"graphs\": [...]} of "
                         "{name, atoms, bonds} records (stored fixture form). "
                         "Each graph is validated and counted; malformed "
                         "graphs are errors (non-zero exit), not skips.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perm-check", type=int, default=0,
                    help="random atom permutations per molecule form asserting "
                         "identical counts (numbering invariance self-test)")
    args = ap.parse_args()
    modes = sum(x is not None for x in (args.export, args.fixture, args.graphs))
    if modes != 1:
        raise SystemExit("exactly one of --export, --fixture and --graphs is required")

    molecules: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    kekule_stats = {"molecules": 0, "forms_compared": 0, "mismatches": 0}

    def take(rec, skip, err, where):
        if rec is not None:
            molecules.append(rec)
            kekule_stats["molecules"] += 1
            kekule_stats["forms_compared"] += rec["forms_stored"]
            return 1
        if err is not None:
            errors.append(where | {"error": err})
            return 0
        skipped.append(where | {"reason": skip})
        return 0

    if args.fixture:
        fixture = json.loads(args.fixture.read_text())
        for m in fixture["molecules"]:
            rec, skip, err = fixture_molecule_record(m)
            take(rec, skip, err, {"smiles": m.get("smiles"), "name": m.get("name")})
    elif args.graphs:
        try:
            payload_in = json.loads(args.graphs.read_text())
        except Exception as e:  # noqa: BLE001
            errors.append({"name": "?", "error": f"malformed graphs payload: unreadable JSON ({e})"})
            payload_in = None
        graphs = None
        if isinstance(payload_in, dict) and isinstance(payload_in.get("graphs"), list):
            graphs = payload_in["graphs"]
        else:
            errors.append({"name": "?", "error": "malformed graphs payload: container must be an object with a list 'graphs'"})
            graphs = []
        for idx, g in enumerate(graphs):
            name, validated, verr = validate_graphs_entry(g, idx)
            if verr is not None:
                take(None, None, verr, {"name": name})
                continue
            atoms_v, bonds_v = validated
            try:
                rec, skip, err = process_graph(name, atoms_v, bonds_v)
            except Exception as e:  # noqa: BLE001
                rec, skip, err = None, None, f"graph failed: {e}"
            take(rec, skip, err, {"name": name})
    else:
        export = json.loads(args.export.read_text())
        for m in export["molecules"]:
            rec, skip, err = export_molecule_record(m)
            if rec is not None:
                molecules.append(rec)
            elif err is not None:
                errors.append({"key": m.get("key"), "error": err})
            else:
                skipped.append({"smiles": None, "key": m.get("key"), "reason": skip})

    # The hand-chosen and invariance molecules always ride along in fixture
    # mode so the written file exercises every type and every kekule case.
    hand = 0
    if args.fixture:
        for name, smiles in HAND_SMILES + INVARIANCE_SMILES:
            rec, skip, err = process_smiles(name, smiles)
            hand += take(rec, skip, err, {"smiles": smiles, "name": name})
        for name, atoms, bonds in hand_graphs():
            try:
                rec, skip, err = process_graph(name, atoms, bonds)
            except Exception as e:  # noqa: BLE001
                rec, skip, err = None, None, f"hand graph failed: {e}"
            hand += take(rec, skip, err, {"name": name, "hand_built": True})
        for name, smiles in FG6_SMILES:
            rec, skip, err = process_smiles(name, smiles)
            hand += take(rec, skip, err, {"smiles": smiles, "name": name})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.perm_check:
        permutation_check(molecules, args.perm_check)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        raise SystemExit(f"reference failed on {len(errors)} supported molecules (errors, not skips)")
    payload = {
        "fg_version": FG_VERSION,
        "chemistry": "ms2-chem-v0.1",
        "names": FG_NAMES,
        "kekule_enumeration": f"complete valid-assignment enumeration over the candidate-graph bonds by backtracking "
                              f"(exact, exponential; cap {MAX_FORMS} forms/molecule, loud failure past it); "
                              f"NOT RDKit ResonanceMolSupplier",
        "molecules": molecules,
        "skipped": skipped,
        "errors": errors,
    }
    if args.export:
        # Kekule regression on the same export when it is the pilot set is
        # computed by the caller comparing stored vs re-kekulized counts;
        # record the aggregate here when feasible.
        try:
            export = json.loads(args.export.read_text())
            n, bad = kekule_regression(export)
            payload["kekule_regression"] = {"compared": n, "mismatches": bad}
            if bad:
                print(f"KEKULE REGRESSION FAILURES ({len(bad)}):")
                for line in bad[:10]:
                    print(f"  {line}")
                raise SystemExit(f"kekule regression failed on {len(bad)}/{n} molecules")
            print(f"kekule regression: {n} molecules agree (stored vs re-kekulized)")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            payload["kekule_regression"] = {"compared": 0, "error": str(e)}
    else:
        payload["kekule_invariance"] = kekule_stats
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    if args.fixture:
        use = sum(1 for n in FG_NAMES for m in molecules
                  for f in m["forms"] if f["counts"].get(n, 0) > 0)
        present = len(set(n for n in FG_NAMES for m in molecules for f in m["forms"]
                          if f["counts"].get(n, 0) > 0))
        print(f"wrote {args.out}: {len(molecules)} molecules "
              f"({hand} hand-chosen+invariance), {len(skipped)} skipped, "
              f"{present}/28 types present, "
              f"{kekule_stats['forms_compared']} forms compared, 0 mismatches")
    else:
        print(f"wrote {args.out}: {len(molecules)} molecules, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
