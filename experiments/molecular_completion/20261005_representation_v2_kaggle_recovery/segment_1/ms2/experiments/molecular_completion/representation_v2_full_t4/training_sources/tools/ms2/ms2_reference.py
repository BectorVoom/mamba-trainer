"""Python reference of docs/MS2_CONTRACTS.md, shared by the MS2 tools.

The chemistry domain (section 4), the integer mass arithmetic (section 5), the
graph-action grammar and canonical traversal (sections 4.4 and 7.4) and the
pseudo-label recipe `q-cut-v1` (section 7.2), written from the contract with
RDKit and Python integers. `make_fixtures.py` turns it into the fixture the Rust
reference is tested against; `audit_casmi.py` and `pilot_targets.py` apply it to
the CASMI data. Nothing here is derived from Rust output.
"""
from __future__ import annotations

import itertools
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

SCALE = 1_000_000
# docs/MS2_CONTRACTS.md section 4.1 (NIST / AME2016).
EXACT = {
    "H": "1.00782503223", "C": "12", "N": "14.00307400443", "O": "15.99491461957",
    "F": "18.99840316273", "P": "30.97376199842", "S": "31.9720711744",
    "Cl": "34.968852682", "Br": "78.9183376", "I": "126.9044719",
}
ELECTRON_EXACT = "0.000548579909065"
ELEMENT_ORDER = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def to_int(decimal_text: str) -> int:
    return int((Decimal(decimal_text) * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))


def residual_nda(decimal_text: str) -> int:
    """Absolute rounding residual in nano-dalton, rounded up."""
    exact = Decimal(decimal_text) * SCALE
    return int((abs(exact - to_int(decimal_text)) * 1000).to_integral_value(rounding="ROUND_CEILING"))


MASS = {e: to_int(v) for e, v in EXACT.items()}
RESIDUAL = {e: residual_nda(v) for e, v in EXACT.items()}
ELECTRON = to_int(ELECTRON_EXACT)
ELECTRON_RESIDUAL = residual_nda(ELECTRON_EXACT)

# docs/MS2_CONTRACTS.md section 4.2: id -> (element, parent hydrogens, valence).
ATOM_TYPES = {
    1: ("C", 0, 4), 2: ("C", 1, 4), 3: ("C", 2, 4), 4: ("C", 3, 4),
    5: ("N", 0, 3), 6: ("N", 1, 3), 7: ("N", 2, 3),
    8: ("O", 0, 2), 9: ("O", 1, 2),
    10: ("F", 0, 1), 11: ("Cl", 0, 1), 12: ("Br", 0, 1),
    13: ("S", 0, 2), 14: ("S", 1, 2), 15: ("S", 0, 6),
    16: ("P", 0, 5), 17: ("I", 0, 1),
}
TYPE_ID = {v: k for k, v in ATOM_TYPES.items()}
ADDUCTS = {1: ("[M+H]+", +1, +1), 2: ("[M-H]-", -1, -1)}  # id -> (name, hydrogens, charge)

U32_MAX = 2**32 - 1
UNKNOWN_UNCERTAINTY = U32_MAX
# Integer weights (contract section 7.2 step 5): intensities in units of 2^-20, and
# a split factor divisible by every count up to 16 (4096 * lcm(1..16)).
INTENSITY_UNITS = 1 << 20
SPLIT_UNITS = 4096 * 720720
MAX_ATOMS, MIN_ATOMS, MAX_CLOSURES, MAX_CUTS, MAX_SHIFT, MAX_TARGETS = 16, 3, 4, 2, 2, 16
PAD, START, ADD, CLOSE, STOP = 0, 1, 2, 3, 4

def raw_atoms(mol):
    out = []
    for a in mol.GetAtoms():
        order = int(sum(b.GetBondTypeAsDouble() for b in a.GetBonds()))
        out.append({
            "element": a.GetSymbol(), "charge": a.GetFormalCharge(), "hydrogens": a.GetTotalNumHs(),
            "isotope": a.GetIsotope(), "radical_electrons": a.GetNumRadicalElectrons(),
            "valence": a.GetTotalNumHs() + order,
        })
    return out


def classify(mol):
    """Every reason the molecule is outside the V0 structure domain (sorted).

    `unsupported_atom_type` is reported for an atom that has none of the other
    per-atom defects and still is not in the vocabulary.
    """
    reasons = set()
    if len(Chem.GetMolFrags(mol)) != 1:
        reasons.add("disconnected")
    for a in raw_atoms(mol):
        own = set()
        if a["element"] not in MASS or a["element"] == "H":
            own.add("element_outside_domain")
        if a["charge"] != 0:
            own.add("formal_charge")
        if a["isotope"] != 0:
            own.add("isotope_label")
        if a["radical_electrons"] != 0:
            own.add("radical")
        if not own and (a["element"], a["hydrogens"], a["valence"]) not in TYPE_ID:
            own.add("unsupported_atom_type")
        reasons |= own
    return sorted(reasons)


def kekulized(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def graph_of(mol):
    atoms = [TYPE_ID[(a["element"], a["hydrogens"], a["valence"])] for a in raw_atoms(mol)]
    bonds = sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         int(b.GetBondTypeAsDouble()))
        for b in mol.GetBonds())
    return atoms, [list(b) for b in bonds]


def composition(atoms):
    c = Counter()
    for t in atoms:
        element, h, _ = ATOM_TYPES[t]
        c[element] += 1
        c["H"] += h
    return c


def mass_of(counts) -> int:
    return sum(MASS[e] * n for e, n in counts.items())


def error_nda(counts) -> int:
    """Representation error bound of a neutral composition, nano-dalton."""
    return sum(RESIDUAL[e] * n for e, n in counts.items())


def enumerate_subgraphs(atoms, bonds):
    """{sorted atom tuple: smallest boundary count} for the contract's bond-cut recipe."""
    n = len(atoms)
    found = {}
    for k in range(0, MAX_CUTS + 1):
        for cut in itertools.combinations(range(len(bonds)), k):
            comp = list(range(n))

            def find(x):
                while comp[x] != x:
                    x = comp[x]
                return x

            for i, (a, b, _) in enumerate(bonds):
                if i not in cut:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        comp[ra] = rb
            roots = [find(i) for i in range(n)]
            if any(roots[bonds[i][0]] == roots[bonds[i][1]] for i in cut):
                continue
            groups = {}
            for i, r in enumerate(roots):
                groups.setdefault(r, []).append(i)
            for r, members in groups.items():
                boundary = sum((roots[bonds[i][0]] == r) + (roots[bonds[i][1]] == r) for i in cut)
                inside = set(members)
                closures = sum(1 for a, b, _ in bonds if a in inside and b in inside) - (len(members) - 1)
                if MIN_ATOMS <= len(members) <= MAX_ATOMS and closures <= MAX_CLOSURES:
                    key = tuple(members)
                    if key not in found or boundary < found[key][0]:
                        found[key] = (boundary, closures)
    return found


def fragment_smiles(mol, atoms_types, members):
    """RDKit canonical SMILES of the fragment with each atom tagged by its type id."""
    tagged = Chem.RWMol(mol)
    # Identity is atom type and bond order only (contract section 7.4): drop the
    # parent's stereo marks, which an isomeric SMILES would otherwise keep.
    Chem.RemoveStereochemistry(tagged)
    for i in members:
        tagged.GetAtomWithIdx(i).SetIsotope(atoms_types[i])
    inside = set(members)
    bond_ids = [b.GetIdx() for b in tagged.GetBonds()
                if b.GetBeginAtomIdx() in inside and b.GetEndAtomIdx() in inside]
    return Chem.MolFragmentToSmiles(tagged, atomsToUse=list(members), bondsToUse=bond_ids, canonical=True,
                                    kekuleSmiles=True, allHsExplicit=True, isomericSmiles=True)


# ---- grammar: traversal, canonical trace and legality, written from the contract ----

def sub_adjacency(members, bonds):
    index = {a: i for i, a in enumerate(members)}
    adj = {i: {} for i in range(len(members))}
    for a, b, order in bonds:
        if a in index and b in index:
            adj[index[a]][index[b]] = order
            adj[index[b]][index[a]] = order
    return adj


def all_bfs_traces(types, adj):
    """Every breadth-first trace of a connected labeled graph (exhaustive)."""
    n = len(types)
    out = []

    def extend(order, head, tokens):
        if len(order) == n:
            out.append(tokens + [(STOP, 0, 0, 0)])
            return
        placed = {a: i for i, a in enumerate(order)}
        while head < len(order):
            u = order[head]
            fresh = [v for v in adj[u] if v not in placed]
            if fresh:
                break
            head += 1
        else:
            return  # disconnected: not reachable for the fixtures
        for v in fresh:
            block = [(ADD, types[v], adj[u][v], head)]
            closing = sorted(placed[w] for w in adj[v] if w in placed and w != u)
            block += [(CLOSE, 0, adj[v][order[p]], p) for p in closing]
            extend(order + [v], head, tokens + block)

    for root in range(n):
        extend([root], 0, [(START, 0, 0, 0), (ADD, types[root], 0, 0)])
    return out


def canonical_trace(types, adj):
    """Exhaustive minimum over every breadth-first traversal: for small graphs only.

    This is the unpruned check of the contract's definition; it has no work limit,
    so the fixture uses it up to 9 atoms. The bounded search is the Rust reference.
    """
    closures = sum(len(n) for n in adj.values()) // 2 - (len(types) - 1)
    if not 1 <= len(types) <= MAX_ATOMS or closures > MAX_CLOSURES:
        raise ValueError("graph outside the grammar limits")
    return min(all_bfs_traces(types, adj))


def first_bfs_trace(types, adj, root=0):
    """One breadth-first trace: neighbours discovered in increasing atom index.

    Not canonical. It gives large graphs a legal trace whose legality masks can be
    written down without the exhaustive search.
    """
    order, placed = [root], {root: 0}
    tokens = [(START, 0, 0, 0), (ADD, types[root], 0, 0)]
    head = 0
    while head < len(order):
        u = order[head]
        for v in sorted(adj[u]):
            if v in placed:
                continue
            tokens.append((ADD, types[v], adj[u][v], head))
            closing = sorted(placed[w] for w in adj[v] if w in placed and w != u)
            tokens += [(CLOSE, 0, adj[v][order[p]], p) for p in closing]
            placed[v] = len(order)
            order.append(v)
        head += 1
    return tokens + [(STOP, 0, 0, 0)]


class Replay:
    """Grammar state of a trace prefix: the legality rules of contract section 4.4."""

    def __init__(self, budget):
        self.budget = budget  # parent composition (element -> count, H included), or None
        self.types, self.residual, self.parent_of, self.bonded = [], [], [], set()
        self.last_parent, self.last_close, self.closures = 0, -1, 0
        self.used = Counter()
        self.step = 0
        self.stopped = False

    def type_fits(self, t):
        if self.budget is None:
            return True
        element, h, _ = ATOM_TYPES[t]
        return (self.used[element] + 1 <= self.budget.get(element, 0)
                and self.used["H"] + h <= self.budget.get("H", 0))

    def add_options(self):
        """{type: {bond: [pointers]}} for a non-root ADD_ATOM."""
        opts = {}
        if not self.types or len(self.types) >= MAX_ATOMS:
            return opts
        for t, (_, h, valence) in ATOM_TYPES.items():
            if not self.type_fits(t):
                continue
            for b in (1, 2, 3):
                if b > valence - h:
                    continue
                ptrs = [p for p in range(self.last_parent, len(self.types)) if self.residual[p] >= b]
                if ptrs:
                    opts.setdefault(t, {})[b] = ptrs
        return opts

    def close_options(self):
        """{bond: [pointers]} for a CLOSE_RING on the newest atom."""
        opts = {}
        if len(self.types) < 2 or self.closures >= MAX_CLOSURES:
            return opts
        newest = len(self.types) - 1
        lo = max(self.parent_of[newest], self.last_close) + 1
        for b in (1, 2, 3):
            if self.residual[newest] < b:
                continue
            ptrs = [p for p in range(lo, newest) if self.residual[p] >= b and (p, newest) not in self.bonded]
            if ptrs:
                opts[b] = ptrs
        return opts

    def masks(self, token):
        """(kind, type, bond, pointer) masks at this step, conditioned on `token`'s
        earlier fields, and whether `token` itself is legal."""
        kind, t, b, p = token
        if self.stopped:
            return [0, 0, 0, 0], False
        if self.step == 0:
            return [1 << START, 0, 0, 0], token == (START, 0, 0, 0)
        if self.step == 1:
            # The root ADD_ATOM is legal only if some atom type fits the budget.
            type_mask = sum(1 << x for x in ATOM_TYPES if self.type_fits(x))
            ok = kind == ADD and t in ATOM_TYPES and bool(type_mask >> t & 1) and b == 0 and p == 0
            return [(1 << ADD) if type_mask else 0, type_mask, 0, 0], ok
        adds, closes = self.add_options(), self.close_options()
        kind_mask = (1 << STOP) | ((1 << ADD) if adds else 0) | ((1 << CLOSE) if closes else 0)
        if kind == ADD and adds:
            type_mask = sum(1 << x for x in adds)
            bond_mask = sum(1 << x for x in adds.get(t, {}))
            ptr_mask = sum(1 << x for x in adds.get(t, {}).get(b, []))
            return [kind_mask, type_mask, bond_mask, ptr_mask], p in adds.get(t, {}).get(b, [])
        if kind == CLOSE and closes:
            bond_mask = sum(1 << x for x in closes)
            ptr_mask = sum(1 << x for x in closes.get(b, []))
            return [kind_mask, 0, bond_mask, ptr_mask], t == 0 and p in closes.get(b, [])
        if kind == STOP:
            return [kind_mask, 0, 0, 0], (t, b, p) == (0, 0, 0)
        return [kind_mask, 0, 0, 0], False

    def apply(self, token):
        kind, t, b, p = token
        if kind == ADD:
            element, h, valence = ATOM_TYPES[t]
            self.used[element] += 1
            self.used["H"] += h
            self.types.append(t)
            self.residual.append(valence - h)
            if self.step == 1:
                self.parent_of.append(-1)
            else:
                self.residual[p] -= b
                self.residual[-1] -= b
                self.parent_of.append(p)
                self.bonded.add((p, len(self.types) - 1))
                self.last_parent = p
            self.last_close = -1
        elif kind == CLOSE:
            newest = len(self.types) - 1
            self.residual[p] -= b
            self.residual[newest] -= b
            self.bonded.add((p, newest))
            self.last_close = p
            self.closures += 1
        elif kind == STOP:
            self.stopped = True
        self.step += 1


def legality_masks(trace, budget):
    """Per-step masks of a legal trace, and the residual valences at its end."""
    state = Replay(budget)
    out = []
    for token in trace:
        masks, ok = state.masks(tuple(token))
        assert ok, (state.step, token)
        out.append(masks)
        state.apply(tuple(token))
    return out, state.residual


def first_illegal_step(trace, budget=None):
    """Index of the first token the grammar forbids, or None for a legal trace."""
    state = Replay(budget)
    for i, token in enumerate(trace):
        _, ok = state.masks(tuple(token))
        if not ok:
            return i
        state.apply(tuple(token))
    return None


S0, STOP_T = (START, 0, 0, 0), (STOP, 0, 0, 0)
INVALID_TRACES = [
    ("missing START", [(ADD, 1, 0, 0), STOP_T]),
    ("root carries a bond", [S0, (ADD, 1, 1, 0), STOP_T]),
    ("STOP before any atom", [S0, STOP_T]),
    ("bond above the new atom's capacity", [S0, (ADD, 1, 0, 0), (ADD, 4, 2, 0), STOP_T]),
    ("bond above the parent's residual valence", [S0, (ADD, 9, 0, 0), (ADD, 1, 2, 0), STOP_T]),
    ("pointer to an atom that does not exist", [S0, (ADD, 1, 0, 0), (ADD, 1, 1, 1), STOP_T]),
    ("parent pointer decreases", [S0, (ADD, 1, 0, 0), (ADD, 1, 1, 0), (ADD, 1, 1, 1), (ADD, 1, 1, 0), STOP_T]),
    ("ring closure right after the root", [S0, (ADD, 1, 0, 0), (CLOSE, 0, 1, 0), STOP_T]),
    ("ring closure onto the parent", [S0, (ADD, 1, 0, 0), (ADD, 1, 1, 0), (CLOSE, 0, 1, 0), STOP_T]),
    ("ring closure pointers not increasing",
     [S0, (ADD, 1, 0, 0), (ADD, 1, 1, 0), (ADD, 1, 1, 0), (ADD, 1, 1, 1), (CLOSE, 0, 1, 2), (CLOSE, 0, 1, 2), STOP_T]),
    ("ring closure onto itself", [S0, (ADD, 1, 0, 0), (ADD, 1, 1, 0), (ADD, 1, 1, 0), (CLOSE, 0, 1, 2), STOP_T]),
    ("unknown atom type", [S0, (ADD, 18, 0, 0), STOP_T]),
    ("bond order zero", [S0, (ADD, 1, 0, 0), (ADD, 1, 0, 0), STOP_T]),
    ("token after STOP", [S0, (ADD, 1, 0, 0), STOP_T, (ADD, 1, 1, 0)]),
    ("PAD inside the trace", [S0, (ADD, 1, 0, 0), (PAD, 0, 0, 0), STOP_T]),
    ("halogen cannot take a second bond", [S0, (ADD, 10, 0, 0), (ADD, 1, 1, 0), (ADD, 1, 1, 0), STOP_T]),
    ("seventeenth atom", [S0, (ADD, 1, 0, 0)] + [(ADD, 3, 1, i) for i in range(16)] + [STOP_T]),
    ("STOP with fields set", [S0, (ADD, 1, 0, 0), (STOP, 0, 0, 1)]),
]


# ---- ions and pseudo-labels ----

def tolerance(mz: int, ppm_tenths: int) -> int:
    return mz * ppm_tenths // 10_000_000


def ion(counts, adduct_id, shift):
    """(m/z, error bound in micro-dalton rounded up) of ion hypothesis (g, s), or None."""
    if adduct_id not in ADDUCTS:
        raise ValueError(f"unsupported_adduct: {adduct_id}")
    _, h_a, z = ADDUCTS[adduct_id]
    extra_h = h_a + shift
    if counts["H"] + extra_h < 0:
        return None
    mz = mass_of(counts) + extra_h * MASS["H"] - z * ELECTRON
    if not 0 <= mz <= U32_MAX:
        raise ValueError(f"mass_overflow: ion m/z {mz}")
    # The ion's own composition: the subgraph's atoms with the net hydrogen count.
    nda = error_nda(counts) + extra_h * RESIDUAL["H"] + ELECTRON_RESIDUAL
    return mz, -(-nda // 1000)


def decide(observed: int, computed: int, error: int, tol: int) -> str:
    r = abs(observed - computed)
    if r + error <= tol:
        return "accept"
    if r > tol + error:
        return "reject"
    return "ambiguous"


def hypotheses(g, adduct_id):
    """Every supported ion of subgraph `g`: [(shift, (m/z, arithmetic error bound))], cached on `g`."""
    key = ("ions", adduct_id)
    if key not in g:
        limit = min(g["boundary"], MAX_SHIFT)
        found = [(s, ion(g["counts"], adduct_id, s)) for s in range(-limit, limit + 1)]
        g[key] = [(s, hyp) for s, hyp in found if hyp is not None]
    return g[key]


def intensity_units(relative: float) -> int:
    """A relative intensity in integer units of 2^-20, rounded half up."""
    return int(relative * INTENSITY_UNITS + 0.5)


def targets(peaks, subgraphs, adduct_id, ppm_tenths, uncertainty=0, ids=None):
    """Recipe q-cut-v1 before the top-16 cut, in integer weights.

    `peaks`: (m/z, linear relative intensity); `ids`: their `peak_id`s (positions
    when omitted). `subgraphs`: dicts with `counts`, `boundary` and `class` (graph
    identity); several embeddings may share a class, and a peak is explained by a
    class when any embedding accepts it at any allowed shift. `uncertainty` is the
    observation half-width added to every ion's arithmetic bound; the unknown
    sentinel disables exact-mass decisions, so nothing is matched or counted.
    A peak with integer intensity I explained by n classes gives each
    `I * SPLIT_UNITS // n`. Returns (weight per class, anchors per class as
    [peak id, shift], ambiguous (peak, embedding, shift) count, explained peak ids).
    """
    weight = Counter()
    anchors = {}
    ambiguous = 0
    explained = set()
    if uncertainty == UNKNOWN_UNCERTAINTY:
        return weight, anchors, ambiguous, explained
    for index, (mz, intensity) in enumerate(peaks):
        peak_id = index if ids is None else ids[index]
        tol = tolerance(mz, ppm_tenths)
        hit = {}
        for g in subgraphs:
            for s, hyp in hypotheses(g, adduct_id):
                verdict = decide(mz, hyp[0], hyp[1] + uncertainty, tol)
                if verdict == "accept":
                    hit.setdefault(g["class"], set()).add(s)
                elif verdict == "ambiguous":
                    ambiguous += 1
        if hit:
            explained.add(peak_id)
        share = intensity_units(intensity) * SPLIT_UNITS // len(hit) if hit else 0
        for cls, shifts in hit.items():
            weight[cls] += share
            anchors.setdefault(cls, []).extend([peak_id, s] for s in sorted(shifts))
    return weight, anchors, ambiguous, explained


def retain(weight, order_key=None, max_targets=None):
    """Top `max_targets` classes by integer weight (ties by `order_key(class)`).

    Returns ({class: q}, dropped weight fraction), `q` normalised over the kept.
    """
    total = sum(weight.values())
    if not total:
        return {}, 0.0
    limit = MAX_TARGETS if max_targets is None else max_targets
    key = order_key or (lambda cls: cls)
    ranked = sorted(weight, key=lambda cls: (-weight[cls], key(cls)))
    kept = ranked[:limit]
    kept_total = sum(weight[c] for c in kept)
    return {c: weight[c] / kept_total for c in kept}, 1.0 - kept_total / total


def filter_peaks(mz, intensity, precursor_mz):
    """Contract section 2 without the cap: (indices kept, linear relative intensity).

    Intensities must already be finite and non-negative (request validation,
    contract section 3.1, rejects anything else before this point).
    """
    for value in intensity:
        if not (value == value and 0 <= value < float("inf")):
            raise ValueError(f"invalid intensity {value!r}: validate the request first")
    keep = [i for i, m in enumerate(mz) if 0 < m <= precursor_mz + 2 * SCALE]
    if not keep:
        return [], []
    top = max(intensity[i] for i in keep)
    if top <= 0:
        return [], []
    keep = [i for i in keep if intensity[i] / top >= 1e-3]
    return keep, [intensity[i] / top for i in keep]


def mz_uncertainty(decimals: int) -> int:
    """Half the last stored digit, in integer units, rounded up (at least 1)."""
    if decimals >= 6:
        return 1
    return -(-(10 ** (6 - decimals)) // 2)


def stored_decimals(values) -> int:
    """Decimals needed to write every value exactly, capped at 6."""
    need = 0
    for v in values:
        for d in range(need, 7):
            if abs(round(v, d) - v) < 1e-9:
                need = d
                break
        else:
            return 6
    return need
