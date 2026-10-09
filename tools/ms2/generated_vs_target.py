"""What would have had to be in the query for the model to generate the target?

Reads a `--dump-candidates` file of `examples/ms2_spectral_completion.rs` (the
target graph plus every accepted candidate of each query) and asks two
questions per query:

1. **How close does the model get?** Morgan Tanimoto of the best candidate to
   the target, and whether the target is in the pool at all.
2. **Which property is the model getting wrong?** For each structural
   descriptor `D` that a spectrum model could plausibly predict, the share of
   the generated pool that already matches the target's value of `D`. A small
   share means the budget is spent on molecules that `D` would have excluded:
   conditioning on `D` would multiply the effective budget per matching
   molecule by about `1 / share`, which is the quantity to compare against
   the measured coverage.

Descriptors, all computed from the candidate graph itself (never from the
answer): ring count, aromatic ring count, the multiset of ring sizes, the
largest ring-system size, the Murcko scaffold, the multiset of Ertl
functional groups, carbon-type counts (hybridisation plus hydrogens), the
degree sequence, and the set of radius-1 and radius-2 atom environments
(Morgan bits at that radius). The last two are what a perfect fingerprint
predictor would supply.

The shares are measured on the pool the model actually produced, so they say
what this model wastes its samples on — not what a database would contain.

    PYTHONPATH=tools/ms2 python tools/ms2/generated_vs_target.py \
        --dump data/ms2/specgen/runD/val_k1024.dump.jsonl \
        --out data/ms2/specgen/runD/val_k1024.diagnosis.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")


def mol_of(atoms, bonds):
    """RDKit molecule of a typed graph (atom type ids, `(a, b, order)`)."""
    rw = Chem.RWMol()
    for type_id in atoms:
        element, hydrogens, _ = ref.ATOM_TYPES[type_id]
        atom = Chem.Atom(element)
        atom.SetNumExplicitHs(int(hydrogens))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for a, b, o in bonds:
        rw.AddBond(int(a), int(b), order[int(o)])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def environments(mol, radius: int) -> frozenset:
    """Morgan bit identifiers of exactly this radius (unhashed)."""
    info: dict = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    return frozenset(k for k, places in info.items() if any(r == radius for _, r in places))


def descriptors(mol) -> dict:
    """The structural descriptors compared per query."""
    rings = mol.GetRingInfo().AtomRings()
    ring_sizes = tuple(sorted(len(r) for r in rings))
    aromatic = sum(1 for r in rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    # Ring systems: rings sharing an atom belong to one system.
    systems: list[set] = []
    for ring in rings:
        joined = set(ring)
        rest = []
        for system in systems:
            if system & joined:
                joined |= system
            else:
                rest.append(system)
        rest.append(joined)
        systems = rest
    carbon_types = Counter()
    degrees = []
    for atom in mol.GetAtoms():
        degrees.append(atom.GetDegree())
        if atom.GetSymbol() == "C":
            carbon_types[(str(atom.GetHybridization()), atom.GetTotalNumHs())] += 1
    try:
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        scaffold = ""
    return {
        "ring_count": len(rings),
        "aromatic_ring_count": aromatic,
        "ring_sizes": ring_sizes,
        "largest_ring_system": max((len(s) for s in systems), default=0),
        "murcko_scaffold": scaffold,
        "carbon_types": tuple(sorted(carbon_types.items())),
        "degree_sequence": tuple(sorted(degrees)),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "env_radius1": environments(mol, 1),
        "env_radius2": environments(mol, 2),
    }


DESCRIPTORS = ["ring_count", "aromatic_ring_count", "ring_sizes", "largest_ring_system",
               "murcko_scaffold", "carbon_types", "degree_sequence", "rotatable_bonds",
               "env_radius1", "env_radius2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N queries (0: all)")
    args = ap.parse_args()

    matches = {d: [] for d in DESCRIPTORS}
    best_tanimoto, best_tanimoto_missed = [], []
    in_pool, pool_sizes, formula_ok = [], [], []
    for number, line in enumerate(open(args.dump)):
        if args.limit and number >= args.limit:
            break
        row = json.loads(line)
        try:
            target = mol_of(row["target"]["atoms"], row["target"]["bonds"])
        except Exception:
            continue
        want = descriptors(target)
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target, 2, nBits=4096)
        target_smiles = Chem.MolToSmiles(target)
        hit = False
        best = 0.0
        per_query = {d: 0 for d in DESCRIPTORS}
        kept = 0
        for candidate in row["candidates"]:
            try:
                mol = mol_of(candidate["atoms"], candidate["bonds"])
            except Exception:
                continue
            kept += 1
            if Chem.MolToSmiles(mol) == target_smiles:
                hit = True
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
            best = max(best, DataStructs.TanimotoSimilarity(target_fp, fp))
            have = descriptors(mol)
            for d in DESCRIPTORS:
                if have[d] == want[d]:
                    per_query[d] += 1
        if kept == 0:
            continue
        pool_sizes.append(kept)
        in_pool.append(hit)
        best_tanimoto.append(best)
        if not hit:
            best_tanimoto_missed.append(best)
        for d in DESCRIPTORS:
            matches[d].append(per_query[d] / kept)
        formula_ok.append(row.get("formula_search", {}).get("true_formula_sampled", None))

    n = len(pool_sizes)
    report = {
        "dump": args.dump.name,
        "queries": n,
        "target_in_pool": int(sum(in_pool)),
        "mean_pool_size": round(float(np.mean(pool_sizes)), 1),
        "best_tanimoto_to_target": {
            "all_mean": round(float(np.mean(best_tanimoto)), 3),
            "missed_mean": round(float(np.mean(best_tanimoto_missed)), 3) if best_tanimoto_missed else None,
            "missed_median": round(float(np.median(best_tanimoto_missed)), 3) if best_tanimoto_missed else None,
            "missed_p90": round(float(np.percentile(best_tanimoto_missed, 90)), 3) if best_tanimoto_missed else None,
        },
        "true_formula_sampled": int(sum(1 for f in formula_ok if f)),
        "descriptor_share_of_pool_matching_the_target": {
            d: {
                "mean_share": round(float(np.mean(matches[d])), 4),
                "median_share": round(float(np.median(matches[d])), 4),
                "queries_with_no_match": int(sum(1 for x in matches[d] if x == 0.0)),
                "effective_budget_multiplier_if_conditioned": (
                    round(1.0 / float(np.mean(matches[d])), 1) if np.mean(matches[d]) > 0 else None
                ),
            }
            for d in DESCRIPTORS
        },
        "note": "shares are over the pool this model generated, so they measure what its samples are spent on; a descriptor the model already matches carries no extra information for it",
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
