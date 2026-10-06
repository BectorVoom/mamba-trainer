#!/usr/bin/env python3
"""Database-ambiguity ladder for molecular completion (MC19).

Model-free analysis: for each kind of evidence, how many known molecules
remain consistent with it. If more than 25 remain for a query, no
ranking-free method can guarantee a top-25 hit for it; if at most 25
remain, a lookup already succeeds.

Usage::

    uv run --project /Users/ods/Documents/Enveda_CASMI python \\
        tools/ms2/completion_evidence_ambiguity.py \\
        --structures <structures.parquet> --out <evidence_ambiguity.json>

Universe: structures with ``fold_identity`` in {1, 2, 3, 4} (fold 0 is a
sealed test fold: its rows are never read beyond the fold column, via a
Parquet pushdown filter), parseable by RDKit, deduplicated by
``scorer_key`` (first occurrence wins). Queries: universe members with
``fold_identity == 1`` and ``identity_group % 3 == 0``. Everything is
reported twice: for all queries, and for queries with at most
``--max-heavy`` heavy atoms.

Evidence keys (one canonical hashable value per universe molecule):

1. ``mass``: neutral monoisotopic mass (``Descriptors.ExactMolWt``); used
   with a +/- ``--ppm`` window, never as an exact key.
2. ``formula``: molecular formula (``rdMolDescriptors.CalcMolFormula``).
3. ``fg``: sorted multiset of functional groups by Ertl's algorithm
   (``rdkit.Contrib.IFG.ifg.identify_functional_groups``). Each group is
   the canonical fragment SMILES of its atom ids,
   ``Chem.MolFragmentToSmiles(mol, atomIds, allHsExplicit=True,
   canonical=True)``, so hydroxyl (``[OH]``) and ether (``[O]``) oxygen
   differ. Stored as the ``\\x1f``-join of the sorted fragments
   (``''`` when there is no group).
4. ``fg_env``: the same multiset, each entry paired with the group's
   environment (the ``type`` string ifg.py returns, e.g. what kind of
   carbon the group sits on). Stored as the ``\\x1f``-join of the sorted
   ``fragment``/``type`` segments.
5. ``ring_counts``: ``(n_rings, tuple(sorted ring sizes),
   n_fully_aromatic_rings)`` from ``mol.GetRingInfo()``.
6. ``aromatic_systems``: sorted multiset of canonical SMILES, one per
   aromatic ring system. Rings in which every atom is aromatic are merged
   when they share at least one atom; each system is serialised as the
   subgraph of its ring atoms joined by ring bonds only (bond orders and
   heteroatoms kept, formal charges kept). Stored as the ``\\x1f``-join
   of the sorted SMILES (``''`` when acyclic or non-aromatic).
7. ``ring_systems``: same construction over *all* rings (fused or spiro
   rings sharing an atom form one system; rings joined by a linker bond
   stay separate systems).
8. ``carbon_types``: counts of carbon atoms by (hybridisation, aromatic,
   total hydrogen count), i.e. roughly what a 13C/DEPT NMR spectrum gives
   in counts. Hybridisation is the RDKit enum value mapped to
   ``{S, SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED, OTHER}``; hydrogens are
   ``atom.GetTotalNumHs()``. Stored as a sorted tuple of
   ``((hyb, aromatic, h), count)`` items.
9. ``atom_types``: same counts over all heavy atoms by (atomic number,
   hydrogen count, aromatic, hybridisation).
10. ``degree_sequence``: sorted tuple of heavy-atom neighbour counts
    (``atom.GetDegree()`` on the molecule without explicit hydrogens).
11. ``scaffold``: Bemis-Murcko scaffold SMILES
    (``MurckoScaffold.GetScaffoldForMol`` + ``Chem.MolToSmiles``);
    ``''`` for acyclic molecules. Recomputed, not read from the column;
    agreement with the column is asserted on a sample and reported.
12. ``generic_scaffold``: the scaffold with every atom carbon and every
    bond single (``MurckoScaffold.MakeScaffoldGeneric``); ``''`` when
    there is no scaffold.
13. ``fg_distances``: sorted multiset of ``(sig_a, sig_b, d)`` over all
    unordered pairs of functional groups, where ``sig`` is the group's
    fragment SMILES as in ``fg`` (``sig_a <= sig_b``) and ``d`` is the
    shortest bond-path length between the two groups (minimum over atom
    pairs of ``Chem.GetDistanceMatrix``). Empty when fewer than two
    groups; overlapping groups give distance 0.
14. ``ecfp2_counts`` / ``ecfp4_counts`` / ``ecfp6_counts``: unhashed
    Morgan count fingerprints of radius 1 / 2 / 3
    (``rdFingerprintGenerator.GetMorganGenerator(radius=r)
    .GetSparseCountFingerprint(mol)``: the full sorted list of
    (environment identifier, count) pairs). They stand for "all local
    atom environments up to that radius are known", i.e. what a perfect
    fingerprint predictor would supply. Stored as the hex SHA-256 of the
    canonical pair list (equality-preserving digest; raw pair lists are
    too large to keep for 200k molecules).
15. ``skeleton``: canonical SMILES of the molecular graph with atom
    elements kept and every bond order set to single (aromaticity flags
    cleared, re-sanitised): connectivity without bond orders.

Pool size of a query under an evidence set E = number of universe
molecules (the query included) agreeing with the query on every key in E;
for ``mass``, agreement is ``|dm| <= ppm * m``. Exact keys are counted by
grouping on the tuple of canonical values; sets containing ``mass``
combine a sorted-mass window scan (bisect) with grouping on the other
keys.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import math
import multiprocessing
import os
import statistics
import sys
import time
from collections import Counter

import pyarrow.parquet as pq

SCOPE_TEXT = (
    "Pool sizes count known molecules in the competition's training structures "
    "(folds 1-4) that share the stated evidence with the query. They are lower "
    "bounds on the number of molecules consistent with that evidence: chemical "
    "space outside this table is not counted. A pool of at most 25 means a lookup "
    "in this table would place the query in a top-25 list regardless of ranking; "
    "it says nothing about a model that must generate the molecule without the table."
)

BASE_KEYS = ["formula", "fg"]
# Keys 4-15 of the spec: every evidence key except mass/formula/fg.
CANDIDATE_KEYS = [
    "fg_env",
    "ring_counts",
    "aromatic_systems",
    "ring_systems",
    "carbon_types",
    "atom_types",
    "degree_sequence",
    "scaffold",
    "generic_scaffold",
    "fg_distances",
    "ecfp2_counts",
    "ecfp4_counts",
    "ecfp6_counts",
    "skeleton",
]
LADDER_ADDITIONS = [
    "ring_counts",
    "carbon_types",
    "aromatic_systems",
    "ring_systems",
    "fg_env",
    "fg_distances",
    "scaffold",
    "ecfp4_counts",
]
REF_SETS = {
    "formula + ecfp4_counts": ["formula", "ecfp4_counts"],
    "formula + scaffold": ["formula", "scaffold"],
    "formula + ring_systems + carbon_types": ["formula", "ring_systems", "carbon_types"],
}

_HYB_NAMES = {
    0: "UNSPECIFIED",
    1: "S",
    2: "SP",
    3: "SP2",
    4: "SP3",
    5: "SP3D",
    6: "SP3D2",
    7: "OTHER",
}

_GENERATORS: dict[int, object] = {}


def _hyb_name(atom) -> str:
    return _HYB_NAMES.get(int(atom.GetHybridization()), f"HYB{int(atom.GetHybridization())}")


def _ring_system_smiles(mol, systems: list[frozenset], ring_bond_idxs: set[int]) -> tuple:
    """Canonical SMILES per ring system (ring atoms + ring bonds only)."""
    from rdkit import Chem

    out = []
    for atoms in systems:
        rw = Chem.RWMol()
        index = {}
        for a in sorted(atoms):
            atom = mol.GetAtomWithIdx(a)
            new = Chem.Atom(atom.GetAtomicNum())
            new.SetIsAromatic(atom.GetIsAromatic())
            new.SetFormalCharge(atom.GetFormalCharge())
            index[a] = rw.AddAtom(new)
        for bond in mol.GetBonds():
            if bond.GetIdx() not in ring_bond_idxs:
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in index and b in index:
                rw.AddBond(index[a], index[b], bond.GetBondType())
        try:
            Chem.SanitizeMol(rw)
            out.append(Chem.MolToSmiles(rw))
        except Exception:
            out.append("")
    return tuple(sorted(out))


def _merge_rings(rings: list) -> list[frozenset]:
    """Union-find over rings; rings sharing >= 1 atom form one system."""
    parent = list(range(len(rings)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    atom_sets = [set(r) for r in rings]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if atom_sets[i] & atom_sets[j]:
                parent[find(i)] = find(j)
    groups: dict[int, set[int]] = {}
    for i, atoms in enumerate(atom_sets):
        groups.setdefault(find(i), set()).update(atoms)
    return [frozenset(g) for g in groups.values()]


def _generic_smiles(scaffold_mol) -> str:
    """Scaffold SMILES with every atom carbon and every bond single."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold_mol))
    except Exception:
        pass
    try:
        rw = Chem.RWMol(scaffold_mol)
        for atom in rw.GetAtoms():
            atom.SetAtomicNum(6)
            atom.SetIsAromatic(False)
        for bond in rw.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return ""


def _skeleton_smiles(mol) -> str:
    """Whole-molecule connectivity SMILES (elements kept, bonds single)."""
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        # Hypervalent corner cases (e.g. a carbon with four ring bonds):
        # fall back to a non-sanitised canonical SMILES of the same graph.
        try:
            return Chem.MolToSmiles(rw, sanitize=False)
        except Exception:
            return ""


def describe_smiles(smiles: str) -> dict | None:
    """Canonical evidence keys for one SMILES; None when unparseable."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Contrib.IFG import ifg as rdkit_ifg

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    global _GENERATORS
    if not _GENERATORS:
        for radius in (1, 2, 3):
            _GENERATORS[radius] = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    frags: list[str] = []
    frag_env: list[str] = []
    group_atom_ids: list[tuple] = []
    for group in rdkit_ifg.identify_functional_groups(mol):
        ids = tuple(sorted(group.atomIds))
        frag = Chem.MolFragmentToSmiles(
            mol, ids, allHsExplicit=True, canonical=True
        )
        frags.append(frag)
        frag_env.append(frag + "\x1f" + str(group.type))
        group_atom_ids.append(ids)
    if len(group_atom_ids) >= 2:
        dist = Chem.GetDistanceMatrix(mol)
        pairs = []
        for i in range(len(group_atom_ids)):
            for j in range(i + 1, len(group_atom_ids)):
                a, b = frags[i], frags[j]
                if a > b:
                    a, b = b, a
                d = min(dist[x][y] for x in group_atom_ids[i] for y in group_atom_ids[j])
                pairs.append((a, b, int(round(d))))
        fg_distances: tuple = tuple(sorted(pairs))
    else:
        fg_distances = ()

    ring_info = mol.GetRingInfo()
    atom_rings = [tuple(r) for r in ring_info.AtomRings()]
    bond_rings = [tuple(r) for r in ring_info.BondRings()]
    ring_bond_idxs = {i for r in bond_rings for i in r}
    n_aromatic = sum(
        1 for r in atom_rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    )
    ring_counts = (ring_info.NumRings(), tuple(sorted(len(r) for r in atom_rings)), n_aromatic)
    aromatic_rings = [
        r for r in atom_rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    aromatic_bonds = set()
    for r in bond_rings:
        for i in r:
            bond = mol.GetBondWithIdx(i)
            ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ends):
                aromatic_bonds.add(i)
    aromatic_systems = _ring_system_smiles(mol, _merge_rings(aromatic_rings), aromatic_bonds)
    all_systems = _ring_system_smiles(mol, _merge_rings(atom_rings), ring_bond_idxs)

    carbon_counter: Counter = Counter()
    atom_counter: Counter = Counter()
    for atom in mol.GetAtoms():
        hyb = _hyb_name(atom)
        arom = bool(atom.GetIsAromatic())
        nh = atom.GetTotalNumHs()
        if atom.GetAtomicNum() == 6:
            carbon_counter[(hyb, arom, nh)] += 1
        atom_counter[(atom.GetAtomicNum(), nh, arom, hyb)] += 1

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol)
    generic = _generic_smiles(scaffold_mol) if scaffold else ""

    ecfp = {}
    for radius, name in ((1, "ecfp2_counts"), (2, "ecfp4_counts"), (3, "ecfp6_counts")):
        items = tuple(
            sorted(_GENERATORS[radius].GetSparseCountFingerprint(mol).GetNonzeroElements().items())
        )
        ecfp[name] = hashlib.sha256(repr(items).encode()).hexdigest()

    return {
        "mass": float(Descriptors.ExactMolWt(mol)),
        "formula": str(rdMolDescriptors.CalcMolFormula(mol)),
        "n_heavy": int(mol.GetNumHeavyAtoms()),
        "fg": "\x1f".join(sorted(frags)),
        "fg_env": "\x1f".join(sorted(frag_env)),
        "ring_counts": ring_counts,
        "aromatic_systems": "\x1f".join(aromatic_systems),
        "ring_systems": "\x1f".join(all_systems),
        "carbon_types": tuple(sorted(carbon_counter.items())),
        "atom_types": tuple(sorted(atom_counter.items())),
        "degree_sequence": tuple(sorted(a.GetDegree() for a in mol.GetAtoms())),
        "scaffold": scaffold,
        "generic_scaffold": generic,
        "fg_distances": fg_distances,
        "skeleton": _skeleton_smiles(mol),
        **ecfp,
    }


def load_universe(path: str, limit_universe: int = 0) -> dict:
    """Read folds 1-4 only (Parquet pushdown filter), dedup by scorer_key.

    Fold-0 rows are never materialised beyond the fold column used by the
    filter itself.
    """
    table = pq.read_table(
        path,
        columns=["scorer_key", "smiles", "fold_identity", "identity_group", "scaffold"],
        filters=[("fold_identity", "in", [1, 2, 3, 4])],
    )
    data = table.to_pylist()
    universe: list[dict] = []
    seen: set[str] = set()
    n_duplicates = 0
    for row in data:
        key = row["scorer_key"]
        if key in seen:
            n_duplicates += 1
            continue
        seen.add(key)
        universe.append(row)
        if limit_universe and len(universe) >= limit_universe:
            break
    return {"rows": universe, "n_duplicates": n_duplicates}


def compute_descriptors(smiles_list: list[str], workers: int) -> list:
    """Per-molecule evidence keys, in input order (deterministic)."""
    if workers <= 1:
        return [describe_smiles(s) for s in smiles_list]
    with multiprocessing.Pool(workers) as pool:
        return pool.map(describe_smiles, smiles_list, chunksize=64)


def combined_key(desc: dict, keys: list[str]) -> tuple:
    """Hashable grouping key for an evidence set (mass excluded)."""
    return tuple(desc[k] for k in keys if k != "mass")


def exact_pools(key_tuples: list[tuple], query_idx: list[int]) -> list[int]:
    """Pool sizes by grouping on exact key tuples."""
    counts = Counter(key_tuples)
    return [counts[key_tuples[q]] for q in query_idx]


def count_within(sorted_masses: list[float], center: float, ppm: float) -> int:
    """Molecules with |m - center| <= ppm * center (inclusive boundary)."""
    tol = ppm * 1e-6 * center
    lo = bisect.bisect_left(sorted_masses, center - tol)
    hi = bisect.bisect_right(sorted_masses, center + tol)
    return hi - lo


def mass_pools(
    masses: list[float],
    group_keys: list | None,
    query_idx: list[int],
    ppm: float,
) -> list[int]:
    """Pool sizes for evidence sets containing the mass window.

    ``group_keys`` holds the combined exact key per molecule (None for a
    mass-only set): within each exact-key group the sorted masses are
    window-scanned with bisect.
    """
    if group_keys is None:
        ordered = sorted(masses)
        return [count_within(ordered, masses[q], ppm) for q in query_idx]
    groups: dict = {}
    for i, (key, mass) in enumerate(zip(group_keys, masses)):
        groups.setdefault(key, []).append(mass)
    for key in groups:
        groups[key].sort()
    return [count_within(groups[group_keys[q]], masses[q], ppm) for q in query_idx]


def evidence_pool_sizes(
    descs: list[dict], query_idx: list[int], keys: list[str], ppm: float
) -> list[int]:
    """Pool sizes for one evidence set over the query indices."""
    if "mass" in keys:
        others = [k for k in keys if k != "mass"]
        groups = None if not others else [combined_key(d, others) for d in descs]
        return mass_pools([d["mass"] for d in descs], groups, query_idx, ppm)
    tuples = [combined_key(d, keys) for d in descs]
    return exact_pools(tuples, query_idx)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Bootstrap-free 95% Wilson score interval for a binomial fraction."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(pools: list[int]) -> dict:
    """Counts, threshold fractions (+Wilson CIs), median/p90/p99/max."""
    n = len(pools)
    ordered = sorted(pools)
    out: dict = {"n": n}
    for name, thr in (("eq1", 1), ("le5", 5), ("le25", 25), ("le100", 100)):
        if thr == 1:
            k = sum(1 for p in pools if p == 1)
        else:
            k = sum(1 for p in pools if p <= thr)
        out[f"frac_{name}"] = k / n if n else 0.0
        if thr != 1:
            lo, hi = wilson_interval(k, n)
            out[f"ci_{name}"] = [lo, hi]
    if n:
        out["median"] = float(statistics.median(ordered))
        out["p90"] = ordered[min(n - 1, math.ceil(0.90 * n) - 1)]
        out["p99"] = ordered[min(n - 1, math.ceil(0.99 * n) - 1)]
        out["max"] = ordered[-1]
    else:
        out.update({"median": 0.0, "p90": 0, "p99": 0, "max": 0})
    return out


def greedy_selection(
    descs: list[dict],
    query_all: list[int],
    query_le: list[int],
    base: list[str],
    threshold: int,
    ppm: float,
    target: float = 0.99,
) -> dict:
    """Forward selection from ``base`` maximising the pool<=thr fraction.

    Selection uses the all-queries fraction; each step also records the
    heavy-restricted stats. Ties break by CANDIDATE_KEYS order
    (deterministic). Stops at ``target`` or when no key strictly helps.
    """
    base_pools = evidence_pool_sizes(descs, query_all, base, ppm)
    base_frac = sum(1 for p in base_pools if p <= threshold) / max(len(base_pools), 1)
    chosen: list[str] = []
    remaining = list(CANDIDATE_KEYS)
    steps: list[dict] = []
    frac = base_frac
    while remaining and frac < target:
        best_key, best_frac, best_pools = None, frac, None
        for key in remaining:
            pools = evidence_pool_sizes(descs, query_all, base + chosen + [key], ppm)
            f = sum(1 for p in pools if p <= threshold) / max(len(pools), 1)
            if f > best_frac:
                best_key, best_frac, best_pools = key, f, pools
        if best_key is None:
            break
        chosen.append(best_key)
        remaining.remove(best_key)
        frac = best_frac
        steps.append(
            {
                "added": best_key,
                "keys": base + list(chosen),
                "all": summarize(best_pools),
                "le": summarize(evidence_pool_sizes(descs, query_le, base + chosen, ppm)),
            }
        )
    return {"base_keys": base, "base_frac": base_frac, "threshold": threshold, "steps": steps}


def set_name(keys: list[str]) -> str:
    return " + ".join(keys)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-heavy", type=int, default=32)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-universe", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    loaded = load_universe(args.structures, args.limit_universe)
    rows = loaded["rows"]
    t_load = time.perf_counter()

    descs_raw = compute_descriptors([r["smiles"] for r in rows], workers)
    kept_rows, descs = [], []
    n_parse_failures = 0
    for row, desc in zip(rows, descs_raw):
        if desc is None:
            n_parse_failures += 1
            continue
        kept_rows.append(row)
        descs.append(desc)
    t_desc = time.perf_counter()

    query_all = [
        i for i, r in enumerate(kept_rows) if r["fold_identity"] == 1 and r["identity_group"] % 3 == 0
    ]
    query_le = [i for i in query_all if descs[i]["n_heavy"] <= args.max_heavy]
    if args.limit_queries:
        query_all = query_all[: args.limit_queries]
        allowed = set(query_all)
        query_le = [i for i in query_le if i in allowed]

    # Scaffold-column agreement on a sample (recomputed, not trusted blindly).
    # The column was built from the canonical tautomer (see the folds
    # README), so report both the exact-string rate and the rate after
    # tautomer canonicalisation.
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    sample_n = min(500, len(descs))
    taut_enumerator = rdMolStandardize.TautomerEnumerator()
    agree_exact = agree_taut = 0
    for row, desc in zip(kept_rows[:sample_n], descs[:sample_n]):
        col = row["scaffold"] or ""
        agree_exact += desc["scaffold"] == col
        mol = Chem.MolFromSmiles(row["smiles"])
        try:
            tm = taut_enumerator.Canonicalize(mol)
            agree_taut += (
                Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(tm)) == col
            )
        except Exception:
            agree_taut += desc["scaffold"] == col
    scaffold_agreement = {
        "sample": sample_n,
        "agree_exact": agree_exact,
        "rate_exact": agree_exact / max(sample_n, 1),
        "agree_tautomer_canonicalised": agree_taut,
        "rate_tautomer_canonicalised": agree_taut / max(sample_n, 1),
        "note": "column scaffold is computed on the canonical tautomer; "
        "exact-string mismatches are tautomeric variants",
    }

    evidence: list[dict] = []
    seen_sets: dict[tuple, dict] = {}

    def add_entry(keys: list[str]) -> dict:
        name = set_name(keys)
        if name in seen_sets:
            return seen_sets[name]
        pools_all = evidence_pool_sizes(descs, query_all, keys, args.ppm)
        pools_le = evidence_pool_sizes(descs, query_le, keys, args.ppm)
        entry = {
            "name": name,
            "keys": keys,
            "all": summarize(pools_all),
            "le": summarize(pools_le),
        }
        evidence.append(entry)
        seen_sets[name] = entry
        return entry

    # (a) current-model evidence and its parts.
    plan_a = [["mass"], ["formula"], ["mass", "fg"], ["formula", "fg"]]
    # (b) formula + fg plus exactly one more key (4-15).
    plan_b = [["formula", "fg", k] for k in CANDIDATE_KEYS]
    # (c) cumulative ladder.
    ladder_keys: list[list[str]] = [["formula", "fg"]]
    for k in LADDER_ADDITIONS:
        ladder_keys.append(ladder_keys[-1] + [k])
    # (e) reference sets without functional groups.
    plan_e = [REF_SETS[k] for k in REF_SETS]
    for keys in plan_a + plan_b + ladder_keys + plan_e:
        add_entry(keys)
    # (d) greedy forward selection for pool <= 25 and pool <= 1.
    greedy_le25 = greedy_selection(descs, query_all, query_le, BASE_KEYS, 25, args.ppm)
    greedy_le1 = greedy_selection(descs, query_all, query_le, BASE_KEYS, 1, args.ppm)
    for run in (greedy_le25, greedy_le1):
        for step in run["steps"]:
            entry = {
                "name": f"greedy_le{run['threshold']}:" + set_name(step["keys"]),
                "keys": step["keys"],
                "all": step["all"],
                "le": step["le"],
            }
            evidence.append(entry)
    t_pool = time.perf_counter()

    import json

    from rdkit import rdBase

    document = {
        "scope": SCOPE_TEXT,
        "params": {
            "structures": args.structures,
            "max_heavy": args.max_heavy,
            "ppm": args.ppm,
            "workers": workers,
            "limit_universe": args.limit_universe,
            "limit_queries": args.limit_queries,
        },
        "rdkit_version": rdBase.rdkitVersion,
        "universe_size": len(descs),
        "n_parse_failures": n_parse_failures,
        "n_duplicate_scorer_keys": loaded["n_duplicates"],
        "scaffold_agreement": scaffold_agreement,
        "n_queries_all": len(query_all),
        "n_queries_le": len(query_le),
        "evidence": evidence,
        "ladder": [set_name(k) for k in ladder_keys],
        "greedy_le25": {
            "order": [s["added"] for s in greedy_le25["steps"]],
            "base_frac_le25": greedy_le25["base_frac"],
            "fractions_le25_all": [s["all"]["frac_le25"] for s in greedy_le25["steps"]],
        },
        "greedy_le1": {
            "order": [s["added"] for s in greedy_le1["steps"]],
            "base_frac_le1": greedy_le1["base_frac"],
            "fractions_le1_all": [s["all"]["frac_eq1"] for s in greedy_le1["steps"]],
        },
        "timing_s": {
            "load": t_load - t0,
            "descriptors": t_desc - t_load,
            "pools": t_pool - t_desc,
            "total": t_pool - t0,
        },
    }
    with open(args.out, "w") as fh:
        json.dump(document, fh, indent=2, sort_keys=False)
        fh.write("\n")

    def block(title: str, field: str, nq: int):
        print(f"--- {title} (n={nq}) ---")
        print(
            f"{'evidence':52s} {'=1':>7s} {'<=5':>7s} {'<=25':>20s} "
            f"{'<=100':>7s} {'med':>9s} {'p90':>9s} {'p99':>9s} {'max':>9s}"
        )
        for entry in evidence:
            s = entry[field]
            lo, hi = s["ci_le25"]
            print(
                f"{entry['name']:52s} {s['frac_eq1']:7.4f} {s['frac_le5']:7.4f} "
                f"{s['frac_le25']:7.4f} [{lo:.4f},{hi:.4f}] "
                f"{s['frac_le100']:7.4f} {s['median']:9.1f} {s['p90']:9d} "
                f"{s['p99']:9d} {s['max']:9d}"
            )

    block("ALL QUERIES", "all", len(query_all))
    block(f"QUERIES WITH <= {args.max_heavy} HEAVY ATOMS", "le", len(query_le))
    print(f"greedy_le25 order: {' -> '.join(s['added'] for s in greedy_le25['steps'])}")
    print(f"greedy_le1 order: {' -> '.join(s['added'] for s in greedy_le1['steps'])}")
    print(
        f"universe={len(descs)} queries_all={len(query_all)} "
        f"queries_le={len(query_le)} parse_failures={n_parse_failures} "
        f"scaffold_agree_exact={agree_exact}/{sample_n} "
        f"scaffold_agree_taut={agree_taut}/{sample_n} "
        f"rdkit={rdBase.rdkitVersion}"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
