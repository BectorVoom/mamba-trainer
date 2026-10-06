"""Tests for `tools/ms2/completion_evidence_ambiguity.py` (MC19 ambiguity ladder).

Runnable with (no dataset needed)::

    uv run --project /Users/ods/Documents/Enveda_CASMI python \\
        tools/ms2/test_completion_evidence_ambiguity.py

A tiny in-memory universe is written to a temporary parquet: the six
C4H10O isomers (four alcohols + two ethers), a C7H8 pair
(toluene / cycloheptatriene), and a fold-0 C4H10O ether that must never
be counted.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import completion_evidence_ambiguity as amb

BUTAN_1_OL = "CCCCO"
BUTAN_2_OL = "CCC(O)C"
METHYLPROPANOL = "CC(C)CO"  # 2-methylpropan-1-ol
TERT_BUTANOL = "CC(C)(O)C"
DIETHYL_ETHER = "CCOCC"
METHYL_PROPYL_ETHER = "CCCOC"
TOLUENE = "Cc1ccccc1"
CHT = "C1C=CC=CC=C1"  # cycloheptatriene
FOLD0_ETHER = "COC(C)C"  # methyl isopropyl ether, C4H10O, sealed fold 0

TINY = [
    (BUTAN_1_OL, 1),
    (BUTAN_2_OL, 1),
    (METHYLPROPANOL, 1),
    (TERT_BUTANOL, 1),
    (DIETHYL_ETHER, 1),
    (METHYL_PROPYL_ETHER, 1),
    (TOLUENE, 1),
    (CHT, 1),
    (FOLD0_ETHER, 0),
]


def _write_tiny_parquet(path: Path):
    """Tiny structures parquet with the columns the tool reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles, folds = zip(*TINY)
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
    table = pa.table(
        {
            "scorer_key": [f"KEY{i:04d}" for i in range(len(smiles))],
            "smiles": list(smiles),
            "fold_identity": list(folds),
            "identity_group": [3] * len(smiles),  # 3 % 3 == 0: all fold-1 are queries
            "scaffold": scaffolds,
        }
    )
    pq.write_table(table, path)


def _tiny_descs(workers: int = 1):
    loaded = load_universe_cached()
    return amb.compute_descriptors([r["smiles"] for r in loaded["rows"]], workers)


_CACHE: dict = {}


def load_universe_cached():
    assert "path" in _CACHE, "test setup: tiny parquet path not registered"
    if "loaded" not in _CACHE:
        _CACHE["loaded"] = amb.load_universe(_CACHE["path"])
    return _CACHE["loaded"]


def _queries(descs, rows):
    return [i for i, r in enumerate(rows) if r["fold_identity"] == 1 and r["identity_group"] % 3 == 0]


def test_fold0_never_counted():
    """Fold-0 rows are excluded: universe 8, butan-1-ol formula pool 6."""
    loaded = load_universe_cached()
    assert len(loaded["rows"]) == 8, len(loaded["rows"])
    assert all(r["fold_identity"] != 0 for r in loaded["rows"])
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    assert len(queries) == 8, queries
    # All six C4H10O isomers share one exact mass; the fold-0 ether must
    # not join the pool.
    pools = amb.evidence_pool_sizes(descs, queries, ["formula"], 5.0)
    assert pools[0] == 6, pools


def test_formula_fg_separates_alcohols_from_ethers():
    """Ertl/IFG: alcohols are [OH] (type CO), ethers [O] (type COC).

    So formula + fg gives pool 4 for each alcohol and pool 2 for each
    ether.
    """
    loaded = load_universe_cached()
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    pools = amb.evidence_pool_sizes(descs, queries, ["formula", "fg"], 5.0)
    assert pools[0:4] == [4, 4, 4, 4], pools
    assert pools[4:6] == [2, 2], pools


def test_carbon_types_pools_derived_by_hand():
    """Carbon-type multisets (all carbons SP3, non-aromatic; H counts):

    butan-1-ol      CH3-CH2-CH2-CH2-OH       {H3:1, H2:3}        -> pool 1
    butan-2-ol      CH3-CH2-CH(OH)-CH3       {H3:2, H2:1, H1:1} -> pool 2
    2-methyl-1-ol   (CH3)2CH-CH2-OH          {H3:2, H2:1, H1:1} -> pool 2
    tert-butanol    (CH3)3C-OH               {H3:3, H0:1}        -> pool 1

    butan-2-ol and 2-methylpropan-1-ol share the same multiset, so they
    stay pooled; the other two alcohols are unique. (fg is '[OH]' for all
    four, formula is C4H10O for all six, so carbon_types decides.)
    """
    loaded = load_universe_cached()
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    pools = amb.evidence_pool_sizes(descs, queries, ["formula", "fg", "carbon_types"], 5.0)
    assert pools[0:4] == [1, 2, 2, 1], pools


def test_ecfp4_gives_singletons():
    """Radius-2 Morgan count environments distinguish every test molecule."""
    loaded = load_universe_cached()
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    pools = amb.evidence_pool_sizes(descs, queries, ["formula", "ecfp4_counts"], 5.0)
    assert pools == [1] * 8, pools


def test_same_formula_pair_and_degree_sequence():
    """Toluene (no FG) vs cycloheptatriene (triene FG): formula pool 2,
    formula + fg pool 1. The two ethers share formula, fg and the heavy
    degree sequence (1,1,2,2,2), so they stay pooled at 2 there."""
    loaded = load_universe_cached()
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    assert amb.evidence_pool_sizes(descs, queries, ["formula"], 5.0)[6:8] == [2, 2]
    assert amb.evidence_pool_sizes(descs, queries, ["formula", "fg"], 5.0)[6:8] == [1, 1]
    pools = amb.evidence_pool_sizes(
        descs, queries, ["formula", "fg", "degree_sequence"], 5.0
    )
    assert pools[4:6] == [2, 2], pools


def test_mass_window_boundary():
    """Window |dm| <= ppm * m is inclusive at the boundary."""
    masses = [100.0, 100.0005, 100.001]  # sorted
    assert amb.count_within(masses, 100.0, 5.0) == 2  # tol = 0.0005, edge in
    assert amb.count_within(masses, 100.0, 4.999) == 1  # tol just below edge
    # Mass-only pools on the tiny universe: all C4H10O isomers share one
    # exact mass, so a generous window pools all six (fold-0 excluded).
    loaded = load_universe_cached()
    descs = _tiny_descs()
    queries = _queries(descs, loaded["rows"])
    pools = amb.evidence_pool_sizes(descs, queries, ["mass"], 5.0)
    assert pools[0] == 6, pools
    assert pools[6] == 2 and pools[7] == 2, pools  # C7H8 pair shares one mass


def test_determinism_across_worker_counts():
    """One worker and two workers give identical descriptors and pools."""
    one = amb.compute_descriptors([s for s, _ in TINY if True][:8], 1)
    two = amb.compute_descriptors([s for s, _ in TINY if True][:8], 2)
    assert one == two
    queries = list(range(8))
    for keys in (["formula", "fg"], ["formula", "fg", "carbon_types"]):
        assert amb.evidence_pool_sizes(one, queries, keys, 5.0) == amb.evidence_pool_sizes(
            two, queries, keys, 5.0
        )


def test_wilson_interval_sanity():
    """Interval contains the estimate, stays in [0, 1], mirrors, shrinks."""
    lo, hi = amb.wilson_interval(5, 10)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0, (lo, hi)
    assert amb.wilson_interval(0, 10)[0] == 0.0
    assert amb.wilson_interval(10, 10)[1] == 1.0
    # Symmetry: lo(k) + hi(n - k) == 1.
    lo1, hi1 = amb.wilson_interval(2, 10)
    lo2, hi2 = amb.wilson_interval(8, 10)
    assert abs(lo1 + hi2 - 1.0) < 1e-12, (lo1, hi2)
    assert abs(hi1 + lo2 - 1.0) < 1e-12, (hi1, lo2)
    # More data narrows the interval.
    _, hi_narrow = amb.wilson_interval(50, 100)
    assert (hi - lo) > (hi_narrow - amb.wilson_interval(50, 100)[0])
    assert amb.wilson_interval(0, 0) == (0.0, 1.0)


def test_cli_end_to_end(tmp=None):
    """Full CLI on the tiny parquet: JSON scope verbatim, no leak."""
    tmpdir = tempfile.TemporaryDirectory() if tmp is None else None
    base = Path(tmpdir.name) if tmpdir else Path(tmp)
    path = base / "tiny.parquet"
    out = base / "tiny.json"
    _write_tiny_parquet(path)
    _CACHE["path"] = str(path)
    _CACHE.pop("loaded", None)
    code = amb.main(
        ["--structures", str(path), "--out", str(out), "--workers", "2",
         "--limit-universe", "0", "--limit-queries", "0"]
    )
    assert code == 0
    document = json.loads(out.read_text())
    assert document["scope"] == amb.SCOPE_TEXT
    assert document["universe_size"] == 8, document["universe_size"]
    assert document["n_queries_all"] == 8
    assert document["n_parse_failures"] == 0
    # Tiny parquet scaffold column was written with the same recompute path,
    # so both rates are 1.0 here (on the real folds the exact-string rate
    # is lower: the column uses the canonical tautomer).
    assert document["scaffold_agreement"]["rate_exact"] == 1.0
    assert document["scaffold_agreement"]["rate_tautomer_canonicalised"] == 1.0
    blob = out.read_text()
    for smi in ("CCCCO", "CCOCC", "c1ccccc1", "KEY000"):
        assert smi not in blob, smi  # aggregates only: no SMILES, no keys
    names = [e["name"] for e in document["evidence"]]
    assert "formula + fg" in names
    assert "formula + fg + carbon_types" in names
    assert document["ladder"][0] == "formula + fg"
    # Tiny universe: formula + fg already pools everything <= 25, so the
    # le25 greedy correctly adds nothing; the le1 greedy must add keys.
    assert document["greedy_le25"]["base_frac_le25"] >= 0.99
    assert document["greedy_le25"]["order"] == []
    assert document["greedy_le1"]["order"], "le1 greedy must add at least one key"
    assert document["greedy_le1"]["fractions_le1_all"][-1] == 1.0
    if tmpdir:
        tmpdir.cleanup()


def main() -> None:
    """Direct runner (no pytest needed): plain asserts on a temp parquet."""
    tmpdir = tempfile.TemporaryDirectory()
    base = Path(tmpdir.name)
    path = base / "tiny.parquet"
    _write_tiny_parquet(path)
    _CACHE["path"] = str(path)
    test_fold0_never_counted()
    test_formula_fg_separates_alcohols_from_ethers()
    test_carbon_types_pools_derived_by_hand()
    test_ecfp4_gives_singletons()
    test_same_formula_pair_and_degree_sequence()
    test_mass_window_boundary()
    test_determinism_across_worker_counts()
    test_wilson_interval_sanity()
    test_cli_end_to_end(tmp=base)
    tmpdir.cleanup()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
