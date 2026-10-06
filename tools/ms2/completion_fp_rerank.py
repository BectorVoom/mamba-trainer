"""Re-rank completion candidates by fingerprint agreement (task MC20c).

The Rust completion experiment (`--dump-candidates`) writes one JSON line per
evaluation query with the target typed graph, the query composition and every
accepted candidate in model rank order, but it cannot score candidates against
the fingerprint evidence (it never computes candidate fingerprints). This tool
does that: it rebuilds every candidate with
`completion_stereo_check.to_rdkit` (no stereo assignment), computes its
`morgan4096` fingerprint with `export_fingerprints_mist.fingerprint_bits`, and
re-ranks the candidates by agreement with the query fingerprint.

Dump line format (see `completion_experiment.rs`, `--dump-candidates` writer;
`Composition` is `[u16; 10]` in C,H,N,O,F,P,S,Cl,Br,I order, bonds are
`[a, b, order]` triples, candidates are in model rank order)::

    {"target": {"atoms": [...], "bonds": [[a,b,o], ...]},
     "composition": [c0, ..., c9],
     "candidates": [{"atoms": [...], "bonds": [...],
                     "samples": n, "best_log_prob": x}, ...]}

A dump line carries no molecule key, key hash or source index: the only query
identity in it is the target typed graph (plus the composition). Panel
molecules are therefore matched to dump queries by the stereo-free canonical
SMILES of the `to_rdkit` molecule; any dump target with zero matches, or with
several matches whose fingerprints disagree, is a loud error. A bits file
(`export_fingerprints_mist bits` output) holds `bits_by_molecule` aligned with
the validation export order, so it must have exactly one entry per dump line
(count check) and every entry must equal the fingerprint computed from the
dump target itself (identity check); both mismatches are loud errors.

Query fingerprint (`--fp-eval-mode` in the Rust code):

- `exact`: every true bit has probability 1, every other bit 0
  (clipped to `[epsilon, 1-epsilon]`); supplied with `--bits`.
- `predicted` / `mist_like`: the panel's `fp_pred_mean` sparse
  `[[bit, prob], ...]` list (only `p >= 0.01` entries are stored); bits absent
  from the list have probability `epsilon`. Supplied with `--panel`. A
  `mist_like` run's sampled noise cannot be reproduced here, so panel runs are
  always scored against the stored mean prediction.

Scores (higher is better; ties keep the model's rank order):

1. `log_likelihood`: sum over all 4096 bits of
   `b*log(p) + (1-b)*log(1-p)` with `p` clipped to `[epsilon, 1-epsilon]`.
2. `cosine`: cosine similarity between the candidate's 0/1 bit vector and the
   query probability vector.
3. `tanimoto`: Tanimoto between the candidate's on-bit set and the query bits
   thresholded at 0.5 (1.0 when both sets are empty).

Target identity among the candidates: the dump records no target flag, so the
target is found by stereo-free canonical SMILES comparison of the `to_rdkit`
molecules (reported as such in the output).

Output holds aggregates only (no SMILES, no molecule keys: the data is
CC BY-NC). `--out`, when given, must point outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from completion_stereo_check import to_rdkit
from export_fingerprints_mist import N_BITS, fingerprint_bits

REPO = Path(__file__).resolve().parents[2]

SCORES = ("log_likelihood", "cosine", "tanimoto")
MODEL_ORDER = "model"
ORDERINGS = (MODEL_ORDER,) + SCORES
TOP_KS = (1, 10, 25)
WILSON_Z = 1.96

TARGET_IDENTITY_METHOD = (
    "canonical SMILES (stereo-free) of to_rdkit molecules; "
    "the dump records no per-candidate target flag"
)


def mol_of(atoms, bonds) -> Chem.Mol:
    """`to_rdkit` molecule for a typed graph, without any stereo assignment."""
    return to_rdkit(list(atoms), [list(b) for b in bonds], {}, None)


def canonical_smiles(mol: Chem.Mol) -> str:
    """Stereo-free canonical SMILES (identity key, never written to output)."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def densify_query_fp(query_fp, *, n_bits: int = N_BITS, epsilon: float) -> np.ndarray:
    """Full-length probability vector for a query fingerprint.

    Accepts a sparse `[[bit, prob], ...]` list (panel `fp_pred_mean`), a
    `{bit: prob}` dict, a full-length dense sequence, or an exact on-bit
    collection (set/frozenset, or a list of ints that is not pairs and whose
    length differs from `n_bits`). Absent sparse bits get `epsilon`; exact
    on-bits get `1 - epsilon`, off-bits `epsilon`. Probabilities are clipped
    to `[epsilon, 1 - epsilon]`.
    """
    probs = np.full(n_bits, epsilon, dtype=np.float64)
    if isinstance(query_fp, (set, frozenset)):
        for b in query_fp:
            probs[int(b)] = 1.0 - epsilon
        return probs
    if isinstance(query_fp, dict):
        items = list(query_fp.items())
    elif isinstance(query_fp, (list, tuple)):
        if len(query_fp) == 0:
            return probs
        first = query_fp[0]
        if isinstance(first, (list, tuple)):
            items = [(int(b), float(p)) for b, p in query_fp]
        elif len(query_fp) == n_bits:
            return np.clip(np.asarray(query_fp, dtype=np.float64),
                           epsilon, 1.0 - epsilon)
        else:
            for b in query_fp:
                probs[int(b)] = 1.0 - epsilon
            return probs
    else:
        arr = np.asarray(query_fp, dtype=np.float64)
        if arr.shape != (n_bits,):
            raise ValueError(
                f"densify_query_fp: dense vector has shape {arr.shape}, "
                f"expected ({n_bits},)")
        return np.clip(arr, epsilon, 1.0 - epsilon)
    for b, p in items:
        b = int(b)
        if not 0 <= b < n_bits:
            raise ValueError(f"densify_query_fp: bit {b} out of range")
        probs[b] = min(max(float(p), epsilon), 1.0 - epsilon)
    return probs


def score_candidates(candidate_bits: list[set[int]], probs: np.ndarray,
                      score: str, *, n_bits: int = N_BITS) -> np.ndarray:
    """Score (higher is better) of every candidate under `score`."""
    if score not in SCORES:
        raise ValueError(f"score_candidates: unknown score {score!r}")
    if score == "tanimoto":
        query_on = set(int(i) for i in np.nonzero(probs >= 0.5)[0])
        out = np.empty(len(candidate_bits), dtype=np.float64)
        for i, cand in enumerate(candidate_bits):
            inter = len(cand & query_on)
            union = len(cand | query_on)
            out[i] = 1.0 if union == 0 else inter / union
        return out
    if score == "cosine":
        denom_base = float(np.dot(probs, probs))
        out = np.empty(len(candidate_bits), dtype=np.float64)
        for i, cand in enumerate(candidate_bits):
            if not cand or denom_base == 0.0:
                out[i] = 0.0
                continue
            dot = float(sum(probs[b] for b in cand))
            out[i] = dot / math.sqrt(len(cand) * denom_base)
        return out
    logp = np.log(probs)
    log1mp = np.log(1.0 - probs)
    base = float(log1mp.sum())
    gain = logp - log1mp
    out = np.empty(len(candidate_bits), dtype=np.float64)
    for i, cand in enumerate(candidate_bits):
        b = np.zeros(n_bits, dtype=np.float64)
        if cand:
            b[list(cand)] = 1.0
        out[i] = float(base + np.dot(b, gain))
    return out


def rerank(candidates, query_fp, score: str, *,
           n_bits: int = N_BITS, epsilon: float = 1e-4) -> list[int]:
    """Best-first order (indices into `candidates`) under `score`.

    `candidates` is a list of on-bit collections (sets, or sorted bit lists);
    `query_fp` is anything `densify_query_fp` accepts (sparse panel
    probabilities or exact on-bits). Ties keep the input (model rank) order:
    Python's sort is stable. Higher scores rank first.
    """
    if score not in SCORES:
        raise ValueError(f"rerank: unknown score {score!r}")
    cand_sets = [set(int(b) for b in c) for c in candidates]
    probs = densify_query_fp(query_fp, n_bits=n_bits, epsilon=epsilon)
    values = score_candidates(cand_sets, probs, score, n_bits=n_bits)
    return sorted(range(len(cand_sets)), key=lambda i: values[i], reverse=True)


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float]:
    """Wilson score interval for `k`/`n` at `z` (95% for z = 1.96)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_dump(path: Path) -> list[dict]:
    """Parse a `--dump-candidates` JSONL file (one query object per line)."""
    lines = path.read_text().splitlines()
    queries = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}")
        for field in ("target", "composition", "candidates"):
            if field not in obj:
                raise ValueError(f"{path}:{lineno}: line lacks {field!r}")
        for side in ("target",):
            for field in ("atoms", "bonds"):
                if field not in obj[side]:
                    raise ValueError(
                        f"{path}:{lineno}: target lacks {field!r}")
        for i, cand in enumerate(obj["candidates"]):
            for field in ("atoms", "bonds", "samples", "best_log_prob"):
                if field not in cand:
                    raise ValueError(
                        f"{path}:{lineno}: candidate {i} lacks {field!r}")
        if "source_index" in obj and not isinstance(obj["source_index"], int):
            raise ValueError(
                f"{path}:{lineno}: source_index is not an integer")
        queries.append(obj)
    if not queries:
        raise ValueError(f"{path}: dump holds no query lines")
    return queries


def load_panel(path: Path) -> list[dict]:
    """Parse a panel file (`fp_mist_panel.json` schema)."""
    doc = json.loads(path.read_text())
    molecules = doc.get("molecules")
    if not isinstance(molecules, list) or not molecules:
        raise ValueError(f"{path}: panel holds no molecules")
    for i, mol in enumerate(molecules):
        for field in ("atoms", "bonds", "fp_true", "fp_pred_mean"):
            if field not in mol:
                raise ValueError(f"{path}: panel molecule {i} lacks {field!r}")
    return molecules


def load_bits(path: Path) -> list[list[int]]:
    """Per-molecule exact on-bit lists from a `bits` sidecar."""
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        by_molecule = doc
    elif isinstance(doc, dict) and isinstance(doc.get("bits_by_molecule"), list):
        by_molecule = doc["bits_by_molecule"]
    else:
        raise ValueError(
            f"{path}: bits file has neither a plain list nor 'bits_by_molecule'")
    out = []
    for i, entry in enumerate(by_molecule):
        try:
            bits = sorted(int(b) for b in entry)
        except TypeError:
            raise ValueError(f"{path}: bits entry {i} is not a bit list")
        if any(b < 0 or b >= N_BITS for b in bits):
            raise ValueError(f"{path}: bits entry {i} holds an out-of-range bit")
        out.append(bits)
    return out


def match_panel_to_dump(dump_queries: list[dict],
                         panel_molecules: list[dict]) -> list[dict]:
    """Match every dump query to its panel molecule.

    When the dump lines carry `source_index` (written by the Rust driver),
    matching is positional: dump line `pos` uses panel entry `source_index`
    (out-of-range indices are loud errors). Otherwise both sides are keyed
    by the stereo-free canonical SMILES of the `to_rdkit` molecule (legacy
    dumps). Returns one dict per dump query with the panel position,
    `fp_pred_mean` and `fp_true`. Loud failures: a dump target with no panel
    molecule, or several panel molecules with disagreeing fingerprints.
    """
    if all(isinstance(q.get("source_index"), int) for q in dump_queries):
        matched = []
        for pos, query in enumerate(dump_queries):
            h = query["source_index"]
            if not 0 <= h < len(panel_molecules):
                raise ValueError(
                    f"query {pos}: source_index {h} is outside "
                    f"{len(panel_molecules)} panel molecules")
            matched.append({"panel_index": h,
                            "fp_pred_mean": panel_molecules[h]["fp_pred_mean"],
                            "fp_true": list(panel_molecules[h]["fp_true"]),
                            "panel_smiles_agree": True})
        return matched
    by_smiles: dict[str, list[int]] = {}
    panel_smiles: list[str] = []
    for i, mol in enumerate(panel_molecules):
        smi = canonical_smiles(mol_of(mol["atoms"], mol["bonds"]))
        panel_smiles.append(smi)
        by_smiles.setdefault(smi, []).append(i)
    matched = []
    for pos, query in enumerate(dump_queries):
        target = query["target"]
        smi = canonical_smiles(mol_of(target["atoms"], target["bonds"]))
        hits = by_smiles.get(smi, [])
        if not hits:
            raise ValueError(
                f"query {pos}: target graph matches no panel molecule "
                f"(count/identity mismatch between dump and panel)")
        if len(hits) > 1:
            pred_keys = {tuple(tuple(e) for e in panel_molecules[h]["fp_pred_mean"])
                         for h in hits}
            true_keys = {tuple(panel_molecules[h]["fp_true"]) for h in hits}
            if len(pred_keys) > 1 or len(true_keys) > 1:
                raise ValueError(
                    f"query {pos}: target graph matches {len(hits)} panel "
                    f"molecules with disagreeing fingerprints (ambiguous)")
        h = hits[0]
        matched.append({"panel_index": h,
                        "fp_pred_mean": panel_molecules[h]["fp_pred_mean"],
                        "fp_true": list(panel_molecules[h]["fp_true"]),
                        "panel_smiles_agree": panel_smiles[h] == smi})
    return matched


def check_bits_against_dump(dump_queries: list[dict],
                            by_molecule: list[list[int]]) -> list[list[int]]:
    """Match a bits sidecar to the dump: count then identity, both loud.

    The sidecar is aligned with the validation export order while the dump
    holds only evaluated queries, so a count mismatch means the two cannot be
    aligned. Identity is verified per line: the entry must equal the
    fingerprint computed from the dump target itself. When the dump lines
    carry `source_index`, entries are addressed by it (positional); legacy
    dumps without it use the dump order with a count check.
    """
    if all(isinstance(q.get("source_index"), int) for q in dump_queries):
        queries_bits = []
        for pos, query in enumerate(dump_queries):
            h = query["source_index"]
            if not 0 <= h < len(by_molecule):
                raise ValueError(
                    f"query {pos}: source_index {h} is outside "
                    f"{len(by_molecule)} bits entries")
            target = query["target"]
            computed = fingerprint_bits(
                canonical_smiles(mol_of(target["atoms"], target["bonds"])))
            if computed != by_molecule[h]:
                raise ValueError(
                    f"query {pos}: bits entry {h} disagrees with the "
                    f"fingerprint computed from the dump target "
                    f"(identity mismatch)")
            queries_bits.append(by_molecule[h])
        return queries_bits
    if len(by_molecule) != len(dump_queries):
        raise ValueError(
            f"bits file holds {len(by_molecule)} entries for "
            f"{len(dump_queries)} dump queries (count mismatch; the bits "
            f"sidecar is aligned with the validation export order, so an "
            f"exact-fingerprint dump must cover every export molecule)")
    queries_bits = []
    for pos, query in enumerate(dump_queries):
        target = query["target"]
        computed = fingerprint_bits(
            canonical_smiles(mol_of(target["atoms"], target["bonds"])))
        if computed != by_molecule[pos]:
            raise ValueError(
                f"query {pos}: bits entry disagrees with the fingerprint "
                f"computed from the dump target (identity mismatch)")
        queries_bits.append(by_molecule[pos])
    return queries_bits


def analyze(dump_queries: list[dict], query_fps: list,
            *, returned: int, epsilon: float, exact: bool) -> dict:
    """Aggregate re-ranking report over all dump queries.

    `query_fps[pos]` is the query fingerprint (sparse panel probabilities or
    exact on-bits) for dump line `pos`. Every dumped candidate is scored and
    re-ranked; top-k is read from the re-ranked list. The model's own order
    is reported as the baseline. `returned` is recorded as the generation
    shortlist cap but never truncates scoring. The denominator of every
    rate is the dump's total query count; queries with no candidates are
    misses everywhere (including 0.0 Tanimoto credit).
    """
    n = len(dump_queries)
    cand_counts: list[int] = []
    ceiling = 0
    top_hits = {ordering: {k: 0 for k in TOP_KS} for ordering in ORDERINGS}
    tanimoto_sums = {ordering: 0.0 for ordering in ORDERINGS}
    exact_counts: list[int] = []
    target_only_exact = 0
    per_query_target_rank = {ordering: [] for ordering in ORDERINGS}

    for pos, query in enumerate(dump_queries):
        target = query["target"]
        target_smi = canonical_smiles(mol_of(target["atoms"], target["bonds"]))
        target_bits = set(fingerprint_bits(target_smi))
        cands = list(query["candidates"])
        cand_counts.append(len(cands))
        cand_bits: list[set[int]] = []
        cand_smiles: list[str] = []
        for cand in cands:
            smi = canonical_smiles(mol_of(cand["atoms"], cand["bonds"]))
            cand_smiles.append(smi)
            cand_bits.append(set(fingerprint_bits(smi)))
        try:
            target_pos = cand_smiles.index(target_smi)
        except ValueError:
            target_pos = None

        orders: dict[str, list[int]] = {MODEL_ORDER: list(range(len(cands)))}
        if cands:
            for score in SCORES:
                orders[score] = rerank(cand_bits, query_fps[pos], score,
                                       epsilon=epsilon)
        else:
            for score in SCORES:
                orders[score] = []
        if target_pos is not None:
            ceiling += 1
        for ordering in ORDERINGS:
            order = orders[ordering]
            if target_pos is not None:
                rank = order.index(target_pos) + 1
                per_query_target_rank[ordering].append(rank)
                for k in TOP_KS:
                    if rank <= k:
                        top_hits[ordering][k] += 1
            else:
                per_query_target_rank[ordering].append(None)
            if order:
                top = cand_bits[order[0]]
                inter = len(top & target_bits)
                union = len(top | target_bits)
                tanimoto_sums[ordering] += 1.0 if union == 0 else inter / union
            # Empty queries contribute 0.0 (a miss earns no partial credit).

        if exact:
            query_on = set(int(b) for b in query_fps[pos])
            n_exact = sum(1 for c in cand_bits if c == query_on)
            exact_counts.append(n_exact)
            if target_pos is not None and n_exact == 1 \
                    and cand_bits[target_pos] == query_on:
                target_only_exact += 1

    orderings_report = {}
    for ordering in ORDERINGS:
        hits = top_hits[ordering]
        orderings_report[ordering] = {
            **{f"top{k}": hits[k] / n for k in TOP_KS},
            **{f"top{k}_count": hits[k] for k in TOP_KS},
            **{f"top{k}_wilson95": wilson(hits[k], n) for k in TOP_KS},
            "mean_tanimoto_top": tanimoto_sums[ordering] / n,
        }
    report: dict = {
        "n_queries": n,
        "n_with_candidates": sum(1 for c in cand_counts if c > 0),
        "mean_candidates_per_query": sum(cand_counts) / n,
        "ceiling_queries": ceiling,
        "ceiling_fraction": ceiling / n,
        "ceiling_wilson95": wilson(ceiling, n),
        "orderings": orderings_report,
        "target_identity": TARGET_IDENTITY_METHOD,
        "returned_cap": returned,
        "epsilon": epsilon,
        "denominator": "dump query lines (queries with no candidates are misses)",
        "empty_tanimoto_rule": "queries with no candidates contribute 0.0",
    }
    if exact:
        report["exact"] = {
            "n_exact_match_total": sum(exact_counts),
            "mean_exact_match_per_query": sum(exact_counts) / n,
            "target_only_exact_queries": target_only_exact,
            "target_only_exact_fraction": target_only_exact / n,
        }
    else:
        report["exact"] = None
    return report


def check_outside_repo(path: Path) -> Path:
    """Resolve `path` and refuse it when it lies inside the repository."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    normalized = absolute.resolve()
    try:
        normalized.relative_to(REPO)
    except ValueError:
        return normalized
    raise ValueError(
        f"refusing to write data-derived output inside the repository: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="--dump-candidates JSONL file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--panel", type=Path, default=None,
                       help="panel file with fp_pred_mean / fp_true")
    group.add_argument("--bits", type=Path, default=None,
                       help="bits sidecar for exact-fingerprint runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (must be outside the repository)")
    parser.add_argument("--returned", type=int, default=25,
                        help="generation shortlist cap (recorded, never truncates scoring: "
                             "every dumped candidate is re-ranked)")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="probability clip for scoring")
    args = parser.parse_args(argv)
    try:
        if args.returned < 1:
            raise ValueError("--returned must be >= 1")
        if not 0.0 < args.epsilon < 0.5:
            raise ValueError("--epsilon must lie in (0, 0.5)")
        out_path = check_outside_repo(args.out) if args.out is not None else None
        dump_queries = load_dump(args.candidates)
        if args.panel is not None:
            panel_molecules = load_panel(args.panel)
            matched = match_panel_to_dump(dump_queries, panel_molecules)
            query_fps = [m["fp_pred_mean"] for m in matched]
            fp_source = f"panel:{args.panel.name}"
            query_fp_mode = "predicted_sparse"
            exact = False
        else:
            by_molecule = load_bits(args.bits)
            query_bits = check_bits_against_dump(dump_queries, by_molecule)
            query_fps = query_bits
            fp_source = f"bits:{args.bits.name}"
            query_fp_mode = "exact"
            exact = True
        report = analyze(dump_queries, query_fps, returned=args.returned,
                         epsilon=args.epsilon, exact=exact)
        report["fp_source"] = fp_source
        report["query_fp_mode"] = query_fp_mode
        report["candidates_file"] = str(args.candidates)
        text = json.dumps(report, indent=2, sort_keys=True)
        if out_path is not None:
            if out_path.parent != Path("."):
                out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n")
        else:
            print(text)
        orderings = report["orderings"]
        print(f"queries={report['n_queries']} "
              f"with_candidates={report['n_with_candidates']} "
              f"ceiling={report['ceiling_queries']}")
        for ordering in ORDERINGS:
            o = orderings[ordering]
            print(f"  {ordering}: top1={o['top1']:.4f} "
                  f"top10={o['top10']:.4f} top25={o['top25']:.4f} "
                  f"mean_tanimoto_top={o['mean_tanimoto_top']:.4f}")
        return 0
    except (ValueError, OSError) as e:
        print(f"completion_fp_rerank: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
