"""Tests for completion_fp_rerank.py (task MC20c).

Plain functions + `main()`, no dataset: all fixtures are hand-written in the
exact `--dump-candidates` line format of `completion_experiment.rs`
(`{"target": {"atoms", "bonds"}, "composition", "candidates": [...]}` with
`[u16; 10]` compositions in C,H,N,O,F,P,S,Cl,Br,I order) and the panel/bits
sidecar formats of `export_fingerprints_mist.py`. Typed graphs use the
`ms2_reference.ATOM_TYPES` ids (C/3H=4, C/2H=3, O/1H=9).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import completion_fp_rerank as r

EPS = 1e-4

ETHANOL = {"atoms": [4, 3, 9], "bonds": [[0, 1, 1], [1, 2, 1]]}
PROPANE = {"atoms": [4, 3, 4], "bonds": [[0, 1, 1], [1, 2, 1]]}
METHANOL = {"atoms": [4, 9], "bonds": [[0, 1, 1]]}
ETHANE = {"atoms": [4, 4], "bonds": [[0, 1, 1]]}

COMP_ETHANOL = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0]
COMP_PROPANE = [3, 8, 0, 0, 0, 0, 0, 0, 0, 0]


def dump_line(target, composition, graphs_with_stats, source_index=None) -> dict:
    """One hand-written dump line in the exact Rust writer format."""
    line = {
        "target": {"atoms": list(target["atoms"]),
                   "bonds": [list(b) for b in target["bonds"]]},
        "composition": list(composition),
        "candidates": [
            {"atoms": list(g["atoms"]), "bonds": [list(b) for b in g["bonds"]],
             "samples": s, "best_log_prob": lp}
            for g, s, lp in graphs_with_stats
        ],
    }
    if source_index is not None:
        line["source_index"] = source_index
    return line


def target_bits(graph) -> set[int]:
    mol = r.mol_of(graph["atoms"], graph["bonds"])
    return set(r.fingerprint_bits(r.canonical_smiles(mol)))


def test_scores_by_hand() -> None:
    """The three scores against values computed by hand (tiny bit space)."""
    n = 8
    a, b, c = {0, 1}, {2}, {1, 2}
    probs = {0: 0.9, 1: 0.8, 2: 0.1}

    def hand_ll(cand: set[int]) -> float:
        total = 0.0
        for bit in range(n):
            p = probs.get(bit, EPS)
            total += math.log(p) if bit in cand else math.log(1.0 - p)
        return total

    got = r.score_candidates([a, b, c], r.densify_query_fp(
        probs, n_bits=n, epsilon=EPS), "log_likelihood", n_bits=n).tolist()
    want = [hand_ll(a), hand_ll(b), hand_ll(c)]
    assert all(abs(x - y) < 1e-9 for x, y in zip(got, want)), (got, want)

    pvec = [probs.get(i, EPS) for i in range(n)]
    sump2 = sum(p * p for p in pvec)

    def hand_cos(cand: set[int]) -> float:
        if not cand:
            return 0.0
        return sum(pvec[i] for i in cand) / math.sqrt(len(cand) * sump2)

    got = r.score_candidates([a, b, c, set()], r.densify_query_fp(
        probs, n_bits=n, epsilon=EPS), "cosine", n_bits=n).tolist()
    want = [hand_cos(a), hand_cos(b), hand_cos(c), 0.0]
    assert all(abs(x - y) < 1e-12 for x, y in zip(got, want)), (got, want)

    # Query bits at threshold 0.5: {0, 1}.
    got = r.score_candidates([a, b, c, set()], r.densify_query_fp(
        probs, n_bits=n, epsilon=EPS), "tanimoto", n_bits=n).tolist()
    assert got == [1.0, 0.0, 1.0 / 3.0, 0.0], got

    # Order checks: A matches the query best under every score.
    for score in r.SCORES:
        assert r.rerank([a, b, c], probs, score,
                        n_bits=n, epsilon=EPS)[0] == 0, score


def test_tie_keeps_model_order() -> None:
    """Tied scores are broken by the model's own rank (stable sort)."""
    assert r.rerank([{0}, {0}, {1}], {0: 0.9}, "log_likelihood",
                    n_bits=4, epsilon=EPS) == [0, 1, 2]
    assert r.rerank([{0}, {0}], {0: 0.9}, "cosine",
                    n_bits=4, epsilon=EPS) == [0, 1]
    assert r.rerank([{5}, {5}], {0: 0.9}, "tanimoto",
                    n_bits=8, epsilon=EPS) == [0, 1]
    assert r.rerank([], {0: 0.9}, "log_likelihood",
                    n_bits=4, epsilon=EPS) == []
    try:
        r.rerank([{0}], {0: 0.9}, "nope", n_bits=4, epsilon=EPS)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown score did not raise")


def test_rerank_moves_target_to_rank1() -> None:
    """Three-candidate query: exact query fp ranks the target first."""
    eth_bits = target_bits(ETHANOL)
    query = dump_line(ETHANOL, COMP_ETHANOL,
                      [(PROPANE, 30, -1.0), (METHANOL, 20, -2.0),
                       (ETHANOL, 5, -5.0)])
    assert r.TARGET_IDENTITY_METHOD.startswith("canonical SMILES")
    report = r.analyze([query], [sorted(eth_bits)], returned=25,
                       epsilon=EPS, exact=True)
    assert report["orderings"]["model"]["top1_count"] == 0
    for score in r.SCORES:
        assert report["orderings"][score]["top1_count"] == 1, score
    assert report["ceiling_queries"] == 1
    assert report["exact"]["target_only_exact_queries"] == 1
    assert report["exact"]["n_exact_match_total"] == 1


def test_empty_query_in_denominator() -> None:
    """Rates use the dump's total query count; empty queries are misses."""
    hit = dump_line(ETHANOL, COMP_ETHANOL,
                    [(PROPANE, 30, -1.0), (ETHANOL, 5, -5.0)])
    empty = dump_line(PROPANE, COMP_PROPANE, [])
    eth_bits = target_bits(ETHANOL)
    report = r.analyze([hit, empty], [sorted(eth_bits), sorted(eth_bits)],
                       returned=25, epsilon=EPS, exact=True)
    assert report["n_queries"] == 2
    assert report["n_with_candidates"] == 1
    assert report["mean_candidates_per_query"] == 1.0
    model = report["orderings"]["model"]
    assert model["top1_count"] == 0 and model["top1"] == 0.0
    assert model["top10_count"] == 1 and model["top10"] == 0.5
    assert model["top10_wilson95"][0] <= 0.5 <= model["top10_wilson95"][1]
    assert report["ceiling_queries"] == 1
    assert report["ceiling_fraction"] == 0.5
    # The empty query contributes 0.0 Tanimoto under every ordering.
    assert report["orderings"]["log_likelihood"]["mean_tanimoto_top"] <= 0.5


def test_exact_counting_branches() -> None:
    """Exact-match totals and the target-only condition."""
    eth_bits = sorted(target_bits(ETHANOL))
    dup = dump_line(ETHANOL, COMP_ETHANOL,
                    [(ETHANOL, 9, -1.0), (ETHANOL, 4, -2.0)])
    missed = dump_line(ETHANOL, COMP_ETHANOL, [(PROPANE, 9, -1.0)])
    report = r.analyze([dup, missed], [eth_bits, eth_bits], returned=25,
                       epsilon=EPS, exact=True)
    assert report["exact"]["n_exact_match_total"] == 2
    # Two exact matches: the target is not the *only* one.
    assert report["exact"]["target_only_exact_queries"] == 0
    assert report["exact"]["mean_exact_match_per_query"] == 1.0
    assert report["exact"]["target_only_exact_fraction"] == 0.0
    assert report["ceiling_queries"] == 1


def test_returned_cap() -> None:
    """Every dumped candidate is scored: no truncation to --returned."""
    query = dump_line(ETHANOL, COMP_ETHANOL,
                      [(PROPANE, 30, -1.0), (METHANOL, 20, -2.0),
                       (ETHANOL, 5, -5.0)])
    eth_bits = sorted(target_bits(ETHANOL))
    report = r.analyze([query], [eth_bits], returned=2, epsilon=EPS,
                       exact=True)
    # All three dumped candidates are scored despite returned=2; the target
    # (model rank 3) is still found and re-ranked first.
    assert report["mean_candidates_per_query"] == 3.0
    assert report["ceiling_queries"] == 1
    assert report["orderings"]["model"]["top1"] == 0.0
    assert report["returned_cap"] == 2
    for score in r.SCORES:
        assert report["orderings"][score]["top1_count"] == 1, score


def panel_molecule(graph, probs) -> dict:
    bits = sorted(target_bits(graph))
    return {"atoms": list(graph["atoms"]),
            "bonds": [list(b) for b in graph["bonds"]],
            "fp_true": bits,
            "fp_pred_mean": [[b, float(p)] for b, p in probs]}


def test_panel_match_and_mismatch() -> None:
    """Panel matching by target identity; mismatches raise loudly."""
    eth_bits = target_bits(ETHANOL)
    query = dump_line(ETHANOL, COMP_ETHANOL, [(ETHANOL, 5, -5.0)])
    panel = [panel_molecule(PROPANE, []),
             panel_molecule(ETHANOL, [(b, 1.0) for b in sorted(eth_bits)])]
    matched = r.match_panel_to_dump([query], panel)
    assert matched[0]["panel_index"] == 1
    assert set(matched[0]["fp_true"]) == eth_bits

    try:
        r.match_panel_to_dump([query], [panel_molecule(PROPANE, [])])
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched target did not raise")

    twin = panel_molecule(ETHANOL, [(b, 0.2) for b in sorted(eth_bits)])
    try:
        r.match_panel_to_dump([query], [panel[1], twin])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous fingerprints did not raise")


def test_positional_matching_by_source_index() -> None:
    """Dump lines with source_index match by position, not structure."""
    eth_bits = sorted(target_bits(ETHANOL))
    prop_bits = sorted(target_bits(PROPANE))
    # Two panel molecules sharing the target graph but with disagreeing
    # fingerprints: structure matching would raise ambiguous, positional does
    # not.
    twin_pred = [[b, 0.2] for b in eth_bits]
    panel = [panel_molecule(PROPANE, []),
             panel_molecule(ETHANOL, [(b, 1.0) for b in eth_bits]),
             panel_molecule(ETHANOL, twin_pred)]
    query = dump_line(ETHANOL, COMP_ETHANOL, [(ETHANOL, 5, -5.0)])
    query["source_index"] = 2
    matched = r.match_panel_to_dump([query], panel)
    assert matched[0]["panel_index"] == 2
    assert matched[0]["fp_pred_mean"] == twin_pred
    # Out-of-range source_index is a loud error.
    query["source_index"] = 7
    try:
        r.match_panel_to_dump([query], panel)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range source_index did not raise")
    # Bits sidecar addressed by source_index, with the identity check.
    q0 = dump_line(ETHANOL, COMP_ETHANOL, [(ETHANOL, 5, -5.0)])
    q0["source_index"] = 1
    by_mol = [prop_bits, eth_bits, prop_bits]
    assert r.check_bits_against_dump([q0], by_mol) == [eth_bits]
    q0["source_index"] = 5
    try:
        r.check_bits_against_dump([q0], by_mol)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range bits source_index did not raise")


def test_bits_match_and_mismatch() -> None:
    """Bits sidecar: count equality plus per-line identity, else raise."""
    q0 = dump_line(ETHANOL, COMP_ETHANOL, [(ETHANOL, 5, -5.0)])
    q1 = dump_line(PROPANE, COMP_PROPANE, [(ETHANE, 3, -1.0)])
    good = [sorted(target_bits(ETHANOL)), sorted(target_bits(PROPANE))]
    assert r.check_bits_against_dump([q0, q1], good) == good

    try:
        r.check_bits_against_dump([q0, q1], good[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("count mismatch did not raise")

    swapped = [good[1], good[0]]
    try:
        r.check_bits_against_dump([q0, q1], swapped)
    except ValueError:
        pass
    else:
        raise AssertionError("identity mismatch did not raise")


def test_out_inside_repo_refused() -> None:
    try:
        r.check_outside_repo(Path("tools/ms2/out.json"))
    except ValueError:
        pass
    else:
        raise AssertionError("in-repo --out was not refused")


TESTS = (
    test_scores_by_hand,
    test_tie_keeps_model_order,
    test_rerank_moves_target_to_rank1,
    test_empty_query_in_denominator,
    test_exact_counting_branches,
    test_returned_cap,
    test_positional_matching_by_source_index,
    test_panel_match_and_mismatch,
    test_bits_match_and_mismatch,
    test_out_inside_repo_refused,
)


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception as e:  # noqa: BLE001 - report, don't hide
            failures += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok {test.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
