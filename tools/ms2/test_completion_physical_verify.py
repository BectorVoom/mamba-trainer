"""Tests for completion_physical_verify (plain functions + main).

Runs with and without pytest; deterministic with the fixed seed. Uses
hand-built molecules plus the committed tiny fixture only.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from completion_physical_verify import (
    ENERGY_COMPARABILITY,
    VerifyConfig,
    reperceive_stereo,
    structural_alerts,
    verify_candidate,
)
from completion_physical_verify import main as verify_main
from completion_stereo_check import (
    candidate_block_to_rdkit_block,
    typed_of_smiles,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "ms2" / "completion_tiny_responses.json"

CFG = VerifyConfig(conformers=4, max_iters=500, seed=20261005, timeout_s=60.0)

CONVERGED = "force_field_optimization_converged"


def smiles_graph(s: str):
    return typed_of_smiles(s)


def test_simple_molecules_converge():
    for s in ["CCO", "c1ccccc1", "CC(=O)O"]:
        atoms, bonds = smiles_graph(s)
        r = verify_candidate(atoms, bonds, config=CFG)
        assert r["status"] == CONVERGED, (s, r["status"], r["reason"])
        assert r["alerts"] == [], (s, r["alerts"])
        assert r["stability"] == {
            "status": "not_evaluated",
            "reason": "a converged force-field optimisation is not evidence "
                      "of thermodynamic stability, kinetic persistence or "
                      "synthesizability"}, s
        assert r["electronic_structure"] == {
            "status": "not_evaluated",
            "reason": "no quantum-chemistry program is installed"}, s
        assert r["energy_comparability"] == ENERGY_COMPARABILITY, s
        assert r["method"]["force_field"] in ("MMFF94", "UFF"), s
        assert r["protocol"] == "physical-verification-v1", s


def _trans_block(atoms, bonds, a, b, value):
    adj: dict[int, set[int]] = {}
    for x, y, _ in bonds:
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)
    ra = min(v for v in adj[a] if v != b)
    rb = min(v for v in adj[b] if v != a)
    return {"tetrahedral_centers": [],
            "double_bonds": [{"atoms": [a, b], "reference": [ra, rb]}],
            "stereoisomers": [{"tetrahedral": [], "double_bonds": [value]}]}


def _has(atoms, bonds, alert_id, stereo=None, isomer=None):
    return any(a["id"] == alert_id
               for a in structural_alerts(atoms, bonds, stereo, isomer))


# Each case: (graph, alert id, expect present). A graph is either a SMILES
# string, an (atoms, bonds) pair, or an (atoms, bonds, stereo, isomer)
# tuple for assignment-dependent alerts.
def _alert_cases():
    hep_atoms = [1, 2, 2, 3, 3, 3, 3]
    hep_bonds = [[0, 2, 2], [2, 1, 1], [0, 3, 1], [3, 4, 1], [4, 1, 1],
                 [0, 5, 1], [5, 6, 1], [6, 1, 1]]
    nor_atoms = [2, 2, 3, 3, 3, 2, 2]
    nor_bonds = [[0, 2, 1], [2, 1, 1], [0, 3, 1], [3, 4, 1], [4, 1, 1],
                 [0, 5, 1], [5, 6, 2], [6, 1, 1]]
    hex_atoms, hex_bonds = smiles_graph("C1CCC=CC1")
    oct_atoms, oct_bonds = smiles_graph("C1=CCCCCCC1")
    trans_hex = _trans_block(hex_atoms, hex_bonds, 3, 4, "trans")
    cis_hex = _trans_block(hex_atoms, hex_bonds, 3, 4, "cis")
    trans_oct = _trans_block(oct_atoms, oct_bonds, 0, 1, "trans")
    return [
        ("C1CC#CC1", "triple_bond_in_small_ring", True),
        ("C1CCCC#CCC1", "triple_bond_in_small_ring", False),
        ("C1=C=CCCC1", "cumulene_in_small_ring", True),
        ("C=C=C", "cumulene_in_small_ring", False),
        ("C1=CC1", "three_membered_ring_unsaturation", True),
        ("C1CC1", "three_membered_ring_unsaturation", False),
        ("C1CC2CCC12", "fused_small_rings", True),
        ("C1CCC1", "fused_small_rings", False),
        ("COOC", "peroxide", True),
        ("CCO", "peroxide", False),
        ("COOOC", "polyoxide_chain", True),
        ("COOC", "polyoxide_chain", False),
        ("NNN", "polynitrogen_chain", True),
        ("NN", "polynitrogen_chain", False),
        ("NCl", "n_halogen", True),
        ("NC", "n_halogen", False),
        ("OCl", "o_halogen", True),
        ("CCO", "o_halogen", False),
        ("C=CO", "enol", True),
        ("CCO", "enol", False),
        ("OCO", "geminal_diol", True),
        ("CCO", "geminal_diol", False),
        ("OCN", "hemiaminal", True),
        ("CCO", "hemiaminal", False),
        ("OCN", "geminal_amino_alcohol", True),
        ("CCO", "geminal_amino_alcohol", False),
        ("OS(=O)(=O)O", "hypervalent_sulfur_or_phosphorus", True),
        ("OP(=O)(O)O", "hypervalent_sulfur_or_phosphorus", True),
        ("CSC", "hypervalent_sulfur_or_phosphorus", False),
        ("C12C3C4C1C5C4C3C25", "cage_small_rings", True),
        ("C1CCCCC1", "cage_small_rings", False),
        ((hep_atoms, hep_bonds), "bridgehead_double_bond", True),
        ((nor_atoms, nor_bonds), "bridgehead_double_bond", False),
        ((hex_atoms, hex_bonds, trans_hex, 0),
         "trans_double_bond_in_small_ring", True),
        ((hex_atoms, hex_bonds, cis_hex, 0),
         "trans_double_bond_in_small_ring", False),
        ((oct_atoms, oct_bonds, trans_oct, 0),
         "trans_double_bond_in_small_ring", False),
    ]


def _unpack(graph):
    if isinstance(graph, str):
        atoms, bonds = smiles_graph(graph)
        return atoms, bonds, None, None
    if len(graph) == 2:
        return graph[0], graph[1], None, None
    return graph[0], graph[1], graph[2], graph[3]


def test_every_alert_positive_and_negative():
    checked = set()
    for graph, alert_id, expect in _alert_cases():
        atoms, bonds, stereo, isomer = _unpack(graph)
        got = _has(atoms, bonds, alert_id, stereo, isomer)
        assert got == expect, (alert_id, expect, got)
        checked.add(alert_id)
    expected_ids = {"trans_double_bond_in_small_ring",
                    "triple_bond_in_small_ring", "cumulene_in_small_ring",
                    "bridgehead_double_bond",
                    "three_membered_ring_unsaturation", "fused_small_rings",
                    "cage_small_rings", "peroxide", "polyoxide_chain",
                    "polynitrogen_chain", "n_halogen", "o_halogen", "enol",
                    "geminal_diol", "hemiaminal", "geminal_amino_alcohol",
                    "hypervalent_sulfur_or_phosphorus"}
    assert checked == expected_ids, checked ^ expected_ids
    # Alert records carry id, atoms, params and a one-line meaning.
    atoms, bonds = smiles_graph("COOC")
    for a in structural_alerts(atoms, bonds):
        assert set(a) == {"id", "atoms", "params", "meaning"}, set(a)
        assert isinstance(a["meaning"], str) and "\n" not in a["meaning"]


def test_cubane_converges_with_cage_alert():
    atoms, bonds = smiles_graph("C12C3C4C1C5C4C3C25")
    r = verify_candidate(atoms, bonds, config=CFG)
    assert r["status"] == CONVERGED, (r["status"], r["reason"])
    ids = {a["id"] for a in r["alerts"]}
    assert "cage_small_rings" in ids, ids


def test_trans_cyclohexene_truthful():
    atoms, bonds = smiles_graph("C1CCC=CC1")
    block = _trans_block(atoms, bonds, 3, 4, "trans")
    assert _has(atoms, bonds, "trans_double_bond_in_small_ring", block, 0)
    r = verify_candidate(atoms, bonds, block, 0, CFG)
    if r["status"] == CONVERGED:
        # Converged only counts if the coordinates really show trans.
        obs = r["diagnostics"]["stereo"]["double_bonds"][0]
        assert obs["requested"] == "trans" and obs["observed"] == "trans" \
            and obs["match"] is True, obs
    else:
        assert r["status"] in ("calculation_failed", "unsupported", "error"), \
            r["status"]
        assert r["reason"], "a non-converged status needs its reason recorded"
    return r["status"]


def _butanol_block():
    return {"tetrahedral_centers": [{"atom": 2, "ligands": [1, 3, 4, "H"]}],
            "double_bonds": [],
            "stereoisomers": [{"tetrahedral": ["ccw"], "double_bonds": []},
                              {"tetrahedral": ["cw"], "double_bonds": []}]}


def _butene_block():
    return {"tetrahedral_centers": [],
            "double_bonds": [{"atoms": [1, 2], "reference": [0, 3]}],
            "stereoisomers": [{"tetrahedral": [], "double_bonds": ["cis"]},
                              {"tetrahedral": [], "double_bonds": ["trans"]}]}


def test_enantiomers_converge_and_differ():
    atoms, bonds = smiles_graph("CCC(O)C")
    block = _butanol_block()
    seen = []
    for i in (0, 1):
        r = verify_candidate(atoms, bonds, block, i, CFG)
        assert r["status"] == CONVERGED, (i, r["status"], r["reason"])
        d = r["diagnostics"]["stereo"]["tetrahedral"][0]
        assert d["match"] is True, (i, d)
        seen.append((d["requested"], d["observed"]))
    assert seen == [("ccw", "ccw"), ("cw", "cw")], seen


def test_cis_trans_butene_converge_and_differ():
    atoms, bonds = smiles_graph("CC=CC")
    block = _butene_block()
    seen = []
    for i in (0, 1):
        r = verify_candidate(atoms, bonds, block, i, CFG)
        assert r["status"] == CONVERGED, (i, r["status"], r["reason"])
        d = r["diagnostics"]["stereo"]["double_bonds"][0]
        assert d["match"] is True, (i, d)
        seen.append((d["requested"], d["observed"]))
    assert seen == [("cis", "cis"), ("trans", "trans")], seen


def test_corrupted_assignment_detected():
    # Flip the request after the fact: the coordinate check must catch it.
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from completion_stereo_check import to_rdkit
    atoms, bonds = smiles_graph("CCC(O)C")
    internal = {"tetrahedral_centers": [{"atom": 2,
                                         "ligands": [1, 3, 4, "H"]}],
                "double_bonds": [], "isomers": [{"tetrahedral": [0],
                                                 "double_bonds": []}]}
    mol = to_rdkit(atoms, bonds, internal, 0)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = CFG.seed
    params.numThreads = 1
    assert list(AllChem.EmbedMultipleConfs(mol_h, numConfs=1, params=params))
    rep = reperceive_stereo(mol_h, 0, internal,
                            {"tetrahedral": ["cw"], "double_bonds": []})
    entry = rep["tetrahedral"][0]
    assert entry["requested"] == "cw" and entry["observed"] == "ccw" \
        and entry["match"] is False, entry


def test_no_force_field_parameters_unsupported():
    atoms, bonds = smiles_graph("FS(F)(F)(F)(F)F")
    r = verify_candidate(atoms, bonds, config=CFG)
    assert r["status"] == "unsupported", (r["status"], r["reason"])
    assert "no_force_field_parameters" in r["reason"], r["reason"]
    assert any(a["id"] == "hypervalent_sulfur_or_phosphorus"
               for a in r["alerts"]), [a["id"] for a in r["alerts"]]


def test_candidate_block_adapter():
    block = {"tetrahedral_centers": [{"atom": 2, "ligands": [1, 3, 4, "H"]}],
             "double_bonds": [{"atoms": [1, 2], "reference": [0, 3]}],
             "stereoisomers": [{"tetrahedral": ["cw"],
                                "double_bonds": ["trans"]}]}
    internal = candidate_block_to_rdkit_block(block)
    assert internal["isomers"] == [{"tetrahedral": [1], "double_bonds": [1]}]
    internal2 = candidate_block_to_rdkit_block(
        {"tetrahedral_centers": [], "double_bonds": [],
         "isomers": [{"tetrahedral": [], "double_bonds": []}]})
    assert internal2["isomers"] == [{"tetrahedral": [], "double_bonds": []}]


def test_fixture_end_to_end():
    docs = json.loads(FIXTURE.read_text())
    before = [(c.get("rank"), c["atoms"], c["bonds"]) for d in docs
              for c in d["candidates"]]
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "verified.json")
        rc = verify_main(["--in", str(FIXTURE), "--out", out,
                          "--conformers", "3", "--max-iters", "300",
                          "--seed", "20261005", "--stereo", "all"])
        assert rc == 0
        text = Path(out).read_text()
        after_docs = json.loads(text)
    assert len(after_docs) == len(docs)
    after = [(c.get("rank"), c["atoms"], c["bonds"]) for d in after_docs
             for c in d["candidates"]]
    assert after == before, "candidate order and count are unchanged"
    for d in after_docs:
        for c in d["candidates"]:
            pv = c["physical_verification"]
            assert pv["protocol"] == "physical-verification-v1"
            assert pv["constitution"]["status"] in (
                CONVERGED, "calculation_failed", "unsupported", "error")
        top = d["physical_verification"]
        assert top["protocol"] == "physical-verification-v1"
        assert top["stability"]["status"] == "not_evaluated"
        assert top["electronic_structure"]["status"] == "not_evaluated"
        assert "force-field energies compare conformers" in \
            top["energy_comparability"]
    for pattern in [r"\bstable\b", "verified stable", "physically valid"]:
        assert not re.search(pattern, text), pattern


def test_output_vocabulary():
    atoms, bonds = smiles_graph("CCO")
    text = json.dumps(verify_candidate(atoms, bonds, config=CFG))
    for pattern in [r"\bstable\b", "verified stable", "physically valid"]:
        assert not re.search(pattern, text), pattern


def test_aromatic_systems_have_no_reactivity_alerts():
    # Phenol must not fire enol; aromatic N-heterocycles must not fire
    # polynitrogen chains (shared aromaticity perception).
    assert not _has(*smiles_graph("Oc1ccccc1"), "enol")
    for s in ["c1n[nH]nc1", "c1nn[nH]n1"]:
        atoms, bonds = smiles_graph(s)
        assert not _has(atoms, bonds, "polynitrogen_chain"), s


def test_fused_systems_have_no_bridgehead_alert():
    # Fused (not bridged) systems: no bridgehead alert, and bounded time.
    import time
    for s in ["C1CC2=C(C1)CCCC2", "Oc1cccc2ccccc12"]:
        atoms, bonds = smiles_graph(s)
        t0 = time.time()
        got = _has(atoms, bonds, "bridgehead_double_bond")
        dt = time.time() - t0
        assert got is False, s
        assert dt < 10, (s, dt)


def test_cyclic_sulfone_is_not_a_cumulene():
    assert not _has(*smiles_graph("O=S1(=O)CCCC1"), "cumulene_in_small_ring")
    assert _has(*smiles_graph("C1=C=CCCC1"), "cumulene_in_small_ring")


def test_carbonyl_carbamates_have_no_hydration_alerts():
    assert not _has(*smiles_graph("NC(=O)O"), "hemiaminal")
    assert not _has(*smiles_graph("NC(=O)O"), "geminal_amino_alcohol")
    assert not _has(*smiles_graph("COC(=O)N"), "geminal_amino_alcohol")
    assert not _has(*smiles_graph("OC(=N)O"), "geminal_diol")
    # Saturated controls still fire; the ether/alcohol split is recorded.
    assert _has(*smiles_graph("OCO"), "geminal_diol")
    atoms, bonds = smiles_graph("OCN")
    got = {a["id"]: a for a in structural_alerts(atoms, bonds)}
    assert "hemiaminal" in got and "geminal_amino_alcohol" in got
    assert got["geminal_amino_alcohol"]["params"] == {"hydroxyl": True}


def test_reference_switched_ring_stereo_has_no_trans_alert():
    atoms, bonds = smiles_graph("C1CCC=CC1")
    mixed = {"tetrahedral_centers": [],
             "double_bonds": [{"atoms": [3, 4], "reference": ["H", 5]}],
             "stereoisomers": [{"tetrahedral": [], "double_bonds": ["trans"]}]}
    assert not _has(atoms, bonds, "trans_double_bond_in_small_ring",
                    mixed, 0)
    both_h = {"tetrahedral_centers": [],
              "double_bonds": [{"atoms": [3, 4], "reference": ["H", "H"]}],
              "stereoisomers": [{"tetrahedral": [], "double_bonds": ["trans"]}]}
    assert _has(atoms, bonds, "trans_double_bond_in_small_ring", both_h, 0)


def test_polynitrogen_occurrences_deduplicated():
    atoms, bonds = smiles_graph("NNN")
    got = [a for a in structural_alerts(atoms, bonds)
           if a["id"] == "polynitrogen_chain"]
    assert len(got) == 1, got


def test_embedding_reports_conformers_requested():
    atoms, bonds = smiles_graph("CCO")
    r = verify_candidate(atoms, bonds, config=CFG)
    assert r["status"] == CONVERGED
    emb = r["embedding"]
    assert "attempts" not in emb and "conformer_attempts" not in r["method"]
    assert emb["conformers_requested"] >= CFG.conformers, emb
    assert all("conformers_requested" in rd for rd in emb["rounds"]), emb
    assert r["method"]["conformers_requested"] == CFG.conformers


def test_summary_renamed_and_warnings_separate():
    import completion_physical_verify as v
    atoms, bonds = smiles_graph("CCO")
    cfg = VerifyConfig(conformers=2, max_iters=200, seed=20261005,
                       timeout_s=60.0)
    out, _ = v.verify_response(
        {"candidates": [{"atoms": atoms, "bonds": bonds}]}, cfg, "none",
        None, lambda a, b, s, i: verify_candidate(a, b, s, i, cfg))
    summary = out["candidates"][0]["physical_verification"]["summary"]
    assert "verified" not in summary, summary
    assert summary["converged_without_structural_alerts"] == 1, summary
    assert summary["diagnostic_warnings"] == [], summary


def test_timeout_terminates_worker_and_keeps_alerts():
    import multiprocessing
    import time
    import completion_physical_verify as v
    atoms, bonds = smiles_graph("COOC")  # peroxide alert expected
    cfg = VerifyConfig(conformers=2, max_iters=100, seed=20261005,
                       timeout_s=0)
    t0 = time.time()
    r = v.run_unit_in_worker(atoms, bonds, None, None, cfg)
    dt = time.time() - t0
    assert r["status"] == "calculation_failed", (r["status"], r["reason"])
    assert "timeout" in r["reason"], r["reason"]
    assert "terminated" in r["reason"] and "joined" in r["reason"], \
        r["reason"]
    assert any(a["id"] == "peroxide" for a in r["alerts"]), r["alerts"]
    assert dt < 30, dt
    assert not multiprocessing.active_children(), \
        multiprocessing.active_children()


def test_crash_keeps_alerts_without_inline_fallback():
    import multiprocessing
    import completion_physical_verify as v
    atoms, bonds = smiles_graph("COOC")
    cfg = VerifyConfig(conformers=2, max_iters=100, seed=20261005,
                       timeout_s=10.0)
    orig = v._verify_unit_worker
    try:
        try:
            fork = multiprocessing.get_context("fork")
        except ValueError:
            fork = None
        if fork is None:
            return
        def boom(payload):
            raise RuntimeError("injected crash")
        v._verify_unit_worker = boom
        r = v.run_unit_in_worker(atoms, bonds, None, None, cfg,
                                 _context=fork)
    finally:
        v._verify_unit_worker = orig
    assert r["status"] == "error", (r["status"], r["reason"])
    assert "worker_failed" in r["reason"], r["reason"]
    assert "RuntimeError" in r["reason"] and "injected crash" in r["reason"]
    assert any(a["id"] == "peroxide" for a in r["alerts"]), r["alerts"]
    assert not multiprocessing.active_children()


def test_repeated_runs_agree_modulo_timing():
    atoms, bonds = smiles_graph("CCO")
    first = verify_candidate(atoms, bonds, config=CFG)
    second = verify_candidate(atoms, bonds, config=CFG)
    norm = lambda r: {k: v for k, v in r.items() if k != "elapsed_s"}
    assert norm(first) == norm(second)
    assert first["diagnostics"]["warnings"] == []
    assert first["diagnostics"]["failed"] == []


def main() -> int:
    test_simple_molecules_converge()
    print("test_simple_molecules_converge ok")
    test_every_alert_positive_and_negative()
    print("test_every_alert_positive_and_negative ok")
    test_cubane_converges_with_cage_alert()
    print("test_cubane_converges_with_cage_alert ok")
    status = test_trans_cyclohexene_truthful()
    print(f"test_trans_cyclohexene_truthful ok (final status: {status})")
    test_enantiomers_converge_and_differ()
    print("test_enantiomers_converge_and_differ ok")
    test_cis_trans_butene_converge_and_differ()
    print("test_cis_trans_butene_converge_and_differ ok")
    test_corrupted_assignment_detected()
    print("test_corrupted_assignment_detected ok")
    test_no_force_field_parameters_unsupported()
    print("test_no_force_field_parameters_unsupported ok")
    test_candidate_block_adapter()
    print("test_candidate_block_adapter ok")
    test_fixture_end_to_end()
    print("test_fixture_end_to_end ok")
    test_output_vocabulary()
    print("test_output_vocabulary ok")
    test_aromatic_systems_have_no_reactivity_alerts()
    print("test_aromatic_systems_have_no_reactivity_alerts ok")
    test_fused_systems_have_no_bridgehead_alert()
    print("test_fused_systems_have_no_bridgehead_alert ok")
    test_cyclic_sulfone_is_not_a_cumulene()
    print("test_cyclic_sulfone_is_not_a_cumulene ok")
    test_carbonyl_carbamates_have_no_hydration_alerts()
    print("test_carbonyl_carbamates_have_no_hydration_alerts ok")
    test_reference_switched_ring_stereo_has_no_trans_alert()
    print("test_reference_switched_ring_stereo_has_no_trans_alert ok")
    test_polynitrogen_occurrences_deduplicated()
    print("test_polynitrogen_occurrences_deduplicated ok")
    test_embedding_reports_conformers_requested()
    print("test_embedding_reports_conformers_requested ok")
    test_summary_renamed_and_warnings_separate()
    print("test_summary_renamed_and_warnings_separate ok")
    test_timeout_terminates_worker_and_keeps_alerts()
    print("test_timeout_terminates_worker_and_keeps_alerts ok")
    test_crash_keeps_alerts_without_inline_fallback()
    print("test_crash_keeps_alerts_without_inline_fallback ok")
    test_repeated_runs_agree_modulo_timing()
    print("test_repeated_runs_agree_modulo_timing ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
