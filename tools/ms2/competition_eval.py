"""Score a generation run the way the competition scores it.

The driver's own report counts a hit by the typed-graph isomorphism of
`same_identity`, and `completion_fp_rerank.py` counts one by stereo-free
canonical SMILES. The competition does neither: it canonicalises tautomers
and compares the **first 14 characters of the InChIKey**, which is the
identity this tool uses:

    key(smiles) = MolToInchiKey(TautomerEnumerator().Canonicalize(mol))[:14]

(`rdkit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator`; the same
expression the reference CASMI pipeline uses for its scorer key.) Two
candidates that collapse to one key are therefore **one** entry of the
submitted list, which the driver's own counts do not model: deduplicating
under this key is part of scoring, since a repeated entry wastes a rank
without adding coverage.

Reported, for each run:

- `pool_recall`: the fraction of queries whose answer appears anywhere among
  the returned candidates. This is the generator's reach and is not a
  competition score.
- `mrr25` and `top1/top10/top25`: over the first 25 **distinct** keys in the
  order the run returned them. This is the competition metric, with the
  run's own ranking.
- the same after re-ranking each query's candidates by the agreement of
  their own `morgan4096` fingerprint with the query's input fingerprint,
  which is only honest when that fingerprint is one a deployment would have
  (say so when it is not).
- `empty`: queries that returned nothing, counted in every denominator.

Every rate is over **all** queries of the file: a query with no candidate,
an unparseable candidate or an unresolvable target is a miss, never a
removal. A 95% interval comes from 2,000 bootstrap resamples of queries.

    PYTHONPATH=tools/ms2 python tools/ms2/competition_eval.py \
        --predictions data/ms2/specgen/progress/final_w256_predictions.jsonl \
        --out data/ms2/specgen/progress/final_w256_competition.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")

_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


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


def scorer_key(mol) -> str | None:
    """The competition's identity: tautomer-canonical InChIKey first block."""
    try:
        return Chem.MolToInchiKey(_ENUMERATOR.Canonicalize(mol))[:14]
    except Exception:
        return None


def fingerprint(mol):
    """`morgan4096` of the molecule, through SMILES as the definition does."""
    through = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if through is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(through, 2, nBits=4096)


def bernoulli_score(bits, dense_logit: np.ndarray) -> float:
    """Log-likelihood of a candidate's bits under the query's probabilities."""
    on = np.zeros(4096, dtype=bool)
    on[list(bits.GetOnBits())] = True
    return float(dense_logit[on].sum())


def rank_of(keys: list[str], answer: str) -> int | None:
    """1-based rank of `answer` among the first 25 distinct keys."""
    seen: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.append(key)
        if key == answer:
            return len(seen)
        if len(seen) >= 25:
            break
    return None


def bootstrap(values: np.ndarray, draws: int = 2000, seed: int = 20261007):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [0.0, 0.0]
    index = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[index].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fingerprint-rerank", action="store_true", default=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.predictions) if line.strip()]
    n = len(rows)
    reciprocal_model = np.zeros(n)
    reciprocal_rerank = np.zeros(n)
    in_pool = np.zeros(n)
    hits_model = {1: 0, 10: 0, 25: 0}
    hits_rerank = {1: 0, 10: 0, 25: 0}
    empty = unresolved_target = 0
    distinct_counts, collapse = [], []
    for q, row in enumerate(rows):
        try:
            target = mol_of(row["target"]["atoms"], row["target"]["bonds"])
            answer = scorer_key(target)
        except Exception:
            answer = None
        if answer is None:
            unresolved_target += 1
            continue
        candidates = row.get("candidates", [])
        if not candidates:
            empty += 1
            continue
        keys, fps = [], []
        for candidate in candidates:
            try:
                mol = mol_of(candidate["atoms"], candidate["bonds"])
                key = scorer_key(mol)
                fp = fingerprint(mol)
            except Exception:
                key, fp = None, None
            keys.append(key)
            fps.append(fp)
        kept = [k for k in keys if k]
        distinct_counts.append(len(set(kept)))
        collapse.append(len(kept) - len(set(kept)))
        if answer in set(kept):
            in_pool[q] = 1.0
        rank = rank_of([k for k in keys if k], answer)
        if rank:
            reciprocal_model[q] = 1.0 / rank
            for limit in hits_model:
                if rank <= limit:
                    hits_model[limit] += 1
        # Re-rank by the query's own fingerprint input.
        supplied = row.get("inputs", {}).get("fingerprint", {})
        pairs = supplied.get("bits") or []
        if args.fingerprint_rerank and pairs:
            logit = np.full(4096, np.log(0.01 / 0.99))
            for bit, probability in pairs:
                probability = min(max(float(probability), 1e-4), 1 - 1e-4)
                logit[int(bit)] = np.log(probability / (1 - probability))
            order = sorted(
                range(len(keys)),
                key=lambda i: -(bernoulli_score(fps[i], logit) if fps[i] is not None else -1e30),
            )
            rank = rank_of([keys[i] for i in order if keys[i]], answer)
            if rank:
                reciprocal_rerank[q] = 1.0 / rank
                for limit in hits_rerank:
                    if rank <= limit:
                        hits_rerank[limit] += 1
        else:
            reciprocal_rerank[q] = reciprocal_model[q]
            for limit in hits_rerank:
                if reciprocal_model[q] >= 1.0 / limit:
                    hits_rerank[limit] += 1

    report = {
        "predictions": args.predictions.name,
        "identity": "tautomer-canonical InChIKey first 14 characters (MolToInchiKey(TautomerEnumerator().Canonicalize(mol))[:14])",
        "queries": n,
        "denominator": "every query of the file; an empty candidate list, an unparseable candidate and an unresolvable target are misses, never removals",
        "empty_candidate_lists": empty,
        "unresolvable_targets": unresolved_target,
        "mean_distinct_keys_returned": round(float(np.mean(distinct_counts)), 1) if distinct_counts else 0.0,
        "mean_candidates_collapsing_to_an_earlier_key": round(float(np.mean(collapse)), 1) if collapse else 0.0,
        "pool_recall": round(float(in_pool.mean()), 4),
        "pool_recall_count": int(in_pool.sum()),
        "model_order": {
            "mrr25": round(float(reciprocal_model.mean()), 4),
            "mrr25_ci95": bootstrap(reciprocal_model),
            **{f"top{k}": v for k, v in hits_model.items()},
        },
        "fingerprint_reranked": {
            "mrr25": round(float(reciprocal_rerank.mean()), 4),
            "mrr25_ci95": bootstrap(reciprocal_rerank),
            **{f"top{k}": v for k, v in hits_rerank.items()},
            "note": "honest only when the query's fingerprint is one a deployment would have; with a true fingerprint this is an oracle ranking",
        },
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
