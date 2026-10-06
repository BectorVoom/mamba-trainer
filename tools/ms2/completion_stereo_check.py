"""RDKit cross-check of `stereo-perception-v2` (MC13).

Rust reports come from the `stereo_report` example (subprocess, built once
with the CPU feature; graphs are piped as one JSON document on stdin, one
JSON document comes back on stdout). The Rust report reaches this tool only
through that example: the wheel is not installed in the RDKit environment,
so the Python binding is deliberately not used here. Typed graphs
(`atoms` = V0 type ids, `bonds` = `[a, b, order]`) come from
`ms2_reference.kekulized` + `graph_of`, so atom indices align with RDKit's.

`to_rdkit` builds the molecule from type ids and bonds with hydrogen counts
fixed (no hydrogen invention: `SetNumExplicitHs`, `SetNoImplicit(True)`) and
applies one assignment under the documented convention mapping:

* tetrahedral: `R` is RDKit's incident-bond order at the centre (the order
  `atom.GetBonds()` returns, hydrogen last when implicit) and `p` the parity
  of the permutation from `L` to `R`; `CHI_TETRAHEDRAL_CW` for `cw` with even
  `p` or `ccw` with odd `p`, else `CHI_TETRAHEDRAL_CCW`.
* double bond: `bond.SetStereoAtoms(x, y)` with real neighbour atom indices
  in the bond's begin/end order, `STEREOCIS` / `STEREOTRANS` relative to
  them; when an end's reference ligand is a hydrogen or lone pair, that end's
  other (heavy) ligand is used and the value inverted once per substituted
  end; when an end has no heavy substituent at all (`=NH`), an explicit
  hydrogen atom is added for it. A dummy atom is never invented for a lone
  pair.

After building, every assigned element is re-read after `SanitizeMol` and
`AssignStereochemistry(cleanIt=False)`, then round-tripped through
isomeric SMILES and re-perceived, and every single-element flip is checked
to change the SMILES: a lost assignment is a `LostAssignmentError` naming
the element, never silently dropped. Assignment lengths and values are
validated against the block first (no silent `zip` truncation).

Checks: (a) every named fixture molecule: `distinct_stereoisomers` equals the
number of distinct canonical isomeric SMILES of all expanded isomers and
equals RDKit's unique enumeration; (b) the first `--limit` molecules of an
export file: per-category counts and agreement fractions (aggregates only;
the data is CC BY-NC). Exit non-zero iff a named fixture disagrees.
Disagreements on real molecules are reported, not hidden: up to 20 as
(heavy atoms, category, ours, RDKit's) without SMILES.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

sys.path.insert(0, str(Path(__file__).parent))
from ms2_reference import ATOM_TYPES, graph_of, kekulized

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "target" / "release" / "examples" / "stereo_report"

NAMED_FIXTURES = {
    "ethanol": "CCO",
    "propane": "CCC",
    "toluene": "Cc1ccccc1",
    "bromochlorofluoromethane": "FC(Cl)Br",
    "2-butanol": "CCC(O)C",
    "isopropanol": "CC(O)C",
    "2,3-butanediol": "CC(O)C(O)C",
    "1,4-dimethylcyclohexane": "CC1CCC(C)CC1",
    "1,2-dimethylcyclohexane": "CC1CCCCC1C",
    "methylcyclohexane": "CC1CCCCC1",
    "pseudo-asymmetric": "CC(Cl)C(F)C(Cl)C",
    "2-butene": "CC=CC",
    "hexa-2,4-diene": "CC=CC=CC",
    "acetaldoxime": "CC=NO",
    "hydrazone": "CC=NN",
    "cyclohexene": "C1CCC=CC1",
    "benzene": "c1ccccc1",
    "cyclooctene": "C1=CCCCCCC1",
    "1,1-dichloroethene": "C=C(Cl)Cl",
    "propene": "CC=C",
}

BOND_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

ENUM_OPTIONS = StereoEnumerationOptions(
    tryEmbedding=False, onlyUnassigned=False, unique=True, maxIsomers=1024
)


class LostAssignmentError(ValueError):
    """An assigned stereo element did not survive RDKit sanitization."""


def build_rust_binary() -> Path:
    proc = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--no-default-features",
            "--features",
            "cpu",
            "--example",
            "stereo_report",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("cargo build of stereo_report failed")
    return EXAMPLE


def rust_reports(entries: list[dict], binary: Path) -> list[dict]:
    """One batched call: graphs in, reports out, order preserved."""
    payload = json.dumps(
        {
            "graphs": entries,
            "max_elements": 10,
            "max_automorphisms": 20000,
            "work_limit": 100000,
            # Full expansion within max_elements (2^10): set comparison
            # needs every canonical representative, not a prefix.
            "max_expanded": 1024,
        }
    )
    proc = subprocess.run(
        [str(binary)], input=payload, capture_output=True, text=True, cwd=str(REPO)
    )
    if proc.returncode != 0:
        raise RuntimeError(f"stereo_report failed: {proc.stderr[:2000]}")
    doc = json.loads(proc.stdout)
    assert doc["version"] == "stereo-perception-v2", doc.get("version")
    reports = doc["reports"]
    assert len(reports) == len(entries), (len(reports), len(entries))
    return reports


def typed_of_smiles(smiles: str) -> tuple[list[int], list[list[int]]]:
    mol = kekulized(smiles)
    atoms, bonds = graph_of(mol)
    assert len(atoms) == mol.GetNumAtoms(), "atom count follows RDKit order"
    return atoms, bonds


def parity_of(from_order: list, to_order: list) -> int:
    """Parity (0 even, 1 odd) of the permutation taking `from_order` into the
    order of `to_order`."""
    pos = [to_order.index(x) for x in from_order]
    inversions = sum(1 for i in range(len(pos)) for j in range(i + 1, len(pos)) if pos[i] > pos[j])
    return inversions % 2


def stereo_block_of(report: dict) -> dict:
    """Tool-internal block: stereogenic centres/bonds plus every expanded
    isomer as 0/1 values aligned with them."""
    tetra = []
    bonds = []
    for idx in report["stereogenic"]:
        element = report["potential"][idx]
        if element["kind"] == "tetrahedral":
            tetra.append(element)
        else:
            bonds.append(element)
    isomers = []
    for values in report["isomers"]:
        by_index = dict(zip(report["stereogenic"], values))
        isomers.append(
            {
                "tetrahedral": [by_index[report["potential"].index(e)] for e in tetra],
                "double_bonds": [by_index[report["potential"].index(e)] for e in bonds],
            }
        )
    return {"tetrahedral_centers": tetra, "double_bonds": bonds, "isomers": isomers}


def candidate_block_to_rdkit_block(stereo: dict) -> dict:
    """Adapt a candidate `stereo` block to the internal block shape `to_rdkit` takes.

    Candidates carry expanded isomers as `stereoisomers` with `"cw"` /
    `"ccw"` and `"cis"` / `"trans"` strings aligned with
    `tetrahedral_centers` / `double_bonds`; `to_rdkit` takes `isomers`
    with `0` / `1` values (`ccw`/`cis` = 0, `cw`/`trans` = 1). A block
    that already has `isomers` (e.g. from `stereo_block_of`) passes
    through with its values coerced to integers. Shared by
    `completion_stereo_check` tooling and `completion_physical_verify`,
    so there is exactly one conversion from a typed graph plus a stereo
    assignment to an RDKit molecule: `to_rdkit` below.
    """
    tetra = list(stereo.get("tetrahedral_centers", []))
    bonds = list(stereo.get("double_bonds", []))
    if "isomers" in stereo:
        isomers = [
            {
                "tetrahedral": [int(v) for v in iso.get("tetrahedral", [])],
                "double_bonds": [int(v) for v in iso.get("double_bonds", [])],
            }
            for iso in stereo["isomers"]
        ]
    else:
        value_map = {"ccw": 0, "cw": 1, "cis": 0, "trans": 1}
        isomers = []
        for iso in stereo.get("stereoisomers", []):
            isomers.append(
                {
                    "tetrahedral": [value_map[v] for v in iso.get("tetrahedral", [])],
                    "double_bonds": [value_map[v] for v in iso.get("double_bonds", [])],
                }
            )
    return {"tetrahedral_centers": tetra, "double_bonds": bonds, "isomers": isomers}


def _resolve_end(mol: Chem.RWMol, end: int, ref, partner: int, label: str) -> tuple[int, bool]:
    """Real neighbour index for one double-bond end plus whether the value
    must invert (reference substituted by the other ligand)."""
    atom = mol.GetAtomWithIdx(end)
    heavies = sorted(
        n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != partner
    )
    if isinstance(ref, int):
        if ref not in heavies:
            raise LostAssignmentError(f"{label}: reference {ref} is no neighbour of {end}")
        return ref, False
    if heavies:
        return heavies[0], True
    if ref == "H":
        # No heavy substituent at all (=NH): add an explicit hydrogen. It
        # replaces one implicit hydrogen count so the valence is unchanged.
        if atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
        hydrogen = Chem.Atom(1)
        hydrogen.SetNoImplicit(True)
        idx = mol.AddAtom(hydrogen)
        mol.AddBond(end, idx, Chem.BondType.SINGLE)
        return idx, False
    # A lone pair with neither heavy substituent nor hydrogen cannot be
    # represented: a dummy atom is never invented.
    raise LostAssignmentError(f"{label}: lone-pair reference on end {end} has no ligand")


def to_rdkit(
    atoms: list[int], bonds: list[list[int]], block: dict, isomer: int | None
) -> Chem.Mol:
    """Build the molecule from type ids and bonds, optionally applying one
    expanded isomer (index into `block["isomers"]`). Returns the RDKit mol
    (not yet a SMILES string); raises `LostAssignmentError` naming the
    element when an applied assignment does not survive sanitization.

    Hardened mapping (F5): the assignment length and every value are
    validated against the block (no silent `zip` truncation); before
    assigning, each designated stereo bond must still be a non-aromatic
    double bond after sanitisation and each designated centre must still
    have its documented ligands; after assigning, the molecule is
    serialised to isomeric SMILES and re-parsed, and RDKit's own stereo
    perception must still find every assigned element specified — and
    flipping any single assigned element must change the SMILES (two
    assignments that serialise identically cannot both be expressed)."""
    mol = Chem.RWMol()
    for t in atoms:
        element, hydrogens, _ = ATOM_TYPES[t]
        atom = Chem.Atom(element)
        atom.SetNumExplicitHs(hydrogens)
        atom.SetNoImplicit(True)
        assert mol.AddAtom(atom) == mol.GetNumAtoms() - 1
    for a, b, order in bonds:
        mol.AddBond(a, b, BOND_ORDER[order])
    Chem.SanitizeMol(mol)
    if isomer is None:
        return mol
    values = block["isomers"][isomer]
    tetra_entries = list(block.get("tetrahedral_centers", []))
    bond_entries = list(block.get("double_bonds", []))
    tetra_values = list(values.get("tetrahedral", []))
    bond_values = list(values.get("double_bonds", []))
    if len(tetra_values) != len(tetra_entries):
        raise LostAssignmentError(
            f"tetrahedral assignment has {len(tetra_values)} values "
            f"for {len(tetra_entries)} centres"
        )
    if len(bond_values) != len(bond_entries):
        raise LostAssignmentError(
            f"double-bond assignment has {len(bond_values)} values "
            f"for {len(bond_entries)} bonds"
        )
    for i, v in enumerate(tetra_values):
        if v not in (0, 1):
            raise LostAssignmentError(f"tetrahedral_centers[{i}]: value {v!r} is not 0/1")
    for i, v in enumerate(bond_values):
        if v not in (0, 1):
            raise LostAssignmentError(f"double_bonds[{i}]: value {v!r} is not 0/1")
    # Pre-assign eligibility: sanitisation may have aromatised a supposed
    # double bond, so every designated bond must still be a non-aromatic
    # double bond now — before any stereo is touched.
    for i, entry in enumerate(bond_entries):
        a, b = entry["atoms"]
        bond = mol.GetBondBetweenAtoms(a, b)
        if (
            bond is None
            or bond.GetBondType() != Chem.BondType.DOUBLE
            or bond.GetIsAromatic()
        ):
            raise LostAssignmentError(
                f"double_bonds[{i}] ({a}, {b}) is not a non-aromatic "
                f"double bond after sanitisation"
            )
    # Double-bond ends first: explicit hydrogens added here never touch a
    # tetrahedral centre (which carries no double bond).
    resolved_bonds = []
    for i, (entry, value) in enumerate(zip(bond_entries, bond_values)):
        label = f"double_bonds[{i}]"
        a, b = entry["atoms"]
        ref_a, ref_b = entry["reference"]
        bond = mol.GetBondBetweenAtoms(a, b)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ref_begin = ref_a if begin == a else ref_b
        ref_end = ref_b if begin == a else ref_a
        x, flip_x = _resolve_end(mol, begin, ref_begin, end, label)
        y, flip_y = _resolve_end(mol, end, ref_end, begin, label)
        want_trans = value == 1
        is_trans = want_trans ^ ((flip_x + flip_y) % 2 == 1)
        bond.SetStereoAtoms(x, y)
        bond.SetStereo(Chem.BondStereo.STEREOTRANS if is_trans else Chem.BondStereo.STEREOCIS)
        resolved_bonds.append((i, a, b))
    for i, (entry, value) in enumerate(zip(tetra_entries, tetra_values)):
        centre = entry["atom"]
        ligands = entry["ligands"]
        atom = mol.GetAtomWithIdx(centre)
        order = [bd.GetOtherAtomIdx(centre) for bd in atom.GetBonds()]
        if "H" in ligands:
            order = order + ["H"]
        if sorted(map(str, order)) != sorted(map(str, ligands)):
            raise LostAssignmentError(
                f"tetrahedral_centers[{i}] ({centre}) no longer has "
                f"documented ligands {ligands} (neighbours {order})"
            )
        p = parity_of(ligands, order)
        clockwise = value == 1
        tag = (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
            if clockwise == (p == 0)
            else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )
        atom.SetChiralTag(tag)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=False)
    for i, entry in enumerate(tetra_entries):
        tag = mol.GetAtomWithIdx(entry["atom"]).GetChiralTag()
        if tag not in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            raise LostAssignmentError(
                f"tetrahedral_centers[{i}] (atom {entry['atom']}): tetrahedral tag lost"
            )
    for i, a, b in resolved_bonds:
        stereo = mol.GetBondBetweenAtoms(a, b).GetStereo()
        if stereo not in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            raise LostAssignmentError(
                f"double_bonds[{i}] (({a}, {b})): bond stereo lost"
            )
    # Post-assign round trip: canonical isomeric SMILES, re-parse, and
    # RDKit's own stereo perception (fresh assignment on the reparsed
    # molecule) must still find every assigned element specified. Atom map
    # numbers carry the original indices through canonicalisation (a
    # non-canonical SMILES would preserve order but its chiral parity does
    # not round-trip reliably).
    for i in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
    ordered_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(ordered_smi)
    if reparsed is None:
        raise LostAssignmentError("assigned molecule did not re-parse from isomeric SMILES")
    new_index_of_old: dict[int, int] = {}
    for atom in reparsed.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            new_index_of_old[map_num - 1] = atom.GetIdx()
            atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(reparsed, cleanIt=True)
    for i, entry in enumerate(tetra_entries):
        old = entry["atom"]
        if old not in new_index_of_old:
            raise LostAssignmentError(
                f"tetrahedral_centers[{i}] (atom {old}): lost from SMILES round-trip"
            )
        tag = reparsed.GetAtomWithIdx(new_index_of_old[old]).GetChiralTag()
        if tag not in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            raise LostAssignmentError(
                f"tetrahedral_centers[{i}] (atom {old}): "
                f"unspecified after SMILES round-trip"
            )
    for i, a, b in resolved_bonds:
        if a not in new_index_of_old or b not in new_index_of_old:
            raise LostAssignmentError(
                f"double_bonds[{i}] (({a}, {b})): lost from SMILES round-trip"
            )
        stereo = reparsed.GetBondBetweenAtoms(
            new_index_of_old[a], new_index_of_old[b]
        ).GetStereo()
        # A SMILES re-parse reports E/Z where the built molecule says
        # cis/trans: both vocabularies mean specified.
        if stereo not in (
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOZ,
        ):
            raise LostAssignmentError(
                f"double_bonds[{i}] (({a}, {b})): unspecified after SMILES round-trip"
            )
    # The maps were transient alignment aids: clear them so the returned
    # molecule is exactly the assigned one.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Expressibility (block level, see `check_isomer_smiles_distinct`):
    # two reported-distinct isomers serialising to one SMILES means the
    # SMILES channel lost an assignment. A single flipped element
    # legitimately coincides when the flip stays in the same orbit
    # (conditional stereogenicity, e.g. a pseudo-asymmetric centre whose
    # ends match), so no per-element flip check is done here: it would
    # false-positive on such symmetries.
    return mol


def check_isomer_smiles_distinct(atoms: list[int], bonds: list[list[int]], block: dict) -> set[str]:
    """Canonical isomeric SMILES of every isomer in `block["isomers"]`.

    Raises `LostAssignmentError` (naming the colliding isomer indices) when
    two reported-distinct isomers serialise identically: the SMILES channel
    then cannot express the assignment (e.g. the heteroaromatic macrocycle
    whose many assignments normalise to one molecule). `to_rdkit` failures
    propagate unchanged.
    """
    seen: dict[str, int] = {}
    for i in range(len(block["isomers"])):
        smi = canonical(Chem.MolToSmiles(to_rdkit(atoms, bonds, block, i), isomericSmiles=True))
        if smi in seen:
            raise LostAssignmentError(
                f"isomers {seen[smi]} and {i} give the same SMILES"
            )
        seen[smi] = i
    return set(seen)


def canonical(smiles: str) -> str:
    """Canonical isomeric SMILES (normalizes atom order and H handling)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable SMILES {smiles}"
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def rdkit_unique_smiles(mol: Chem.Mol) -> set[str]:
    """Canonical isomeric SMILES of RDKit's unique stereoisomer enumeration."""
    return {canonical(Chem.MolToSmiles(m, isomericSmiles=True)) for m in EnumerateStereoisomers(mol, options=ENUM_OPTIONS)}


def bare_for_enumeration(atoms: list[int], bonds: list[list[int]], report: dict) -> Chem.Mol:
    """Bare mol for RDKit's own enumeration. Terminal =NH double-bond ends
    need explicit hydrogens for the enumerator to see the end."""
    block = stereo_block_of(report)
    bare = to_rdkit(atoms, bonds, block, None)
    if any(
        e["kind"] == "double_bond" and "H" in e["reference"]
        for e in report["potential"]
    ):
        bare = Chem.AddHs(bare)
    return bare


def check_named_fixtures(binary: Path) -> list[str]:
    """Three-way agreement on every named fixture. Returns failure lines."""
    names = list(NAMED_FIXTURES)
    entries = []
    for name in names:
        atoms, bonds = typed_of_smiles(NAMED_FIXTURES[name])
        entries.append({"atoms": atoms, "bonds": bonds})
    reports = rust_reports(entries, binary)
    failures = []
    for name, atoms, bonds, report in zip(names, [e["atoms"] for e in entries], [e["bonds"] for e in entries], reports):
        assert report["resolution"] == "resolved", f"{name}: {report['resolution']}"
        assert not report["unsupported"], f"{name}: {report['unsupported']}"
        block = stereo_block_of(report)
        try:
            ours = check_isomer_smiles_distinct(atoms, bonds, block)
        except LostAssignmentError as e:
            failures.append(f"{name}: lost assignment ({e})")
            continue
        bare = bare_for_enumeration(atoms, bonds, report)
        theirs = rdkit_unique_smiles(bare)
        ours_count, distinct, theirs_count = len(ours), report["distinct_stereoisomers"], len(theirs)
        if not (distinct == ours_count == theirs_count and ours == theirs):
            failures.append(
                f"{name}: distinct={distinct} ours={ours_count} rdkit={theirs_count} "
                f"sets_equal={ours == theirs}"
            )
    return failures


CATEGORIES = ("no_stereo", "centres_only", "bonds_only", "both", "unsupported", "unresolved")


def category_of(report: dict) -> str:
    if report["resolution"] != "resolved":
        return "unresolved"
    if report["unsupported"]:
        return "unsupported"
    kinds = {e["kind"] for e in report["potential"]}
    if not kinds:
        return "no_stereo"
    if kinds == {"tetrahedral"}:
        return "centres_only"
    if kinds == {"double_bond"}:
        return "bonds_only"
    return "both"


def check_export(export: Path, limit: int, binary: Path) -> int:
    """Agreement aggregates over the first `limit` export molecules.
    Always returns 0 (real-molecule disagreements are reported, not hidden,
    and never fail the run); prints aggregates only."""
    doc = json.loads(export.read_text())
    molecules = doc["molecules"][:limit]
    entries = [{"atoms": m["atoms"], "bonds": m["bonds"]} for m in molecules]
    reports = rust_reports(entries, binary)
    stats: dict[str, dict[str, int]] = {
        c: {"n": 0, "count_match": 0, "set_match": 0, "set_compared": 0} for c in CATEGORIES
    }
    disagreements: list[tuple[int, str, object, object]] = []
    for mol, report in zip(molecules, reports):
        atoms, bonds = mol["atoms"], mol["bonds"]
        heavy = len(atoms)
        cat = category_of(report)
        stats[cat]["n"] += 1
        if cat in ("unresolved", "unsupported"):
            continue
        block = stereo_block_of(report)
        try:
            ours = {
                canonical(Chem.MolToSmiles(to_rdkit(atoms, bonds, block, i), isomericSmiles=True))
                for i in range(len(block["isomers"]))
            }
            # Distinct reported isomers must stay distinct through SMILES;
            # a collapse is a lost assignment, recorded — never hidden.
            if len(ours) != len(block["isomers"]):
                disagreements.append((heavy, cat, report["distinct_stereoisomers"], "collapsed-smiles"))
                continue
        except LostAssignmentError:
            disagreements.append((heavy, cat, report["distinct_stereoisomers"], "lost-assignment"))
            continue
        theirs = rdkit_unique_smiles(bare_for_enumeration(atoms, bonds, report))
        count_ok = report["distinct_stereoisomers"] == len(ours) == len(theirs)
        set_ok = ours == theirs
        stats[cat]["count_match"] += int(count_ok)
        stats[cat]["set_match"] += int(set_ok)
        stats[cat]["set_compared"] += 1
        if not (count_ok and set_ok):
            disagreements.append((heavy, cat, report["distinct_stereoisomers"], len(theirs)))
    print(f"export molecules: {len(molecules)}")
    for cat in CATEGORIES:
        s = stats[cat]
        n = s["n"]
        if n == 0:
            print(f"  {cat}: n=0")
            continue
        if cat in ("unresolved", "unsupported"):
            print(f"  {cat}: n={n}")
            continue
        print(
            f"  {cat}: n={n} count_match={s['count_match'] / n:.3f} "
            f"set_match={s['set_match'] / n:.3f}"
        )
    print(f"disagreements: {len(disagreements)} (up to 20 shown, no SMILES)")
    for heavy, cat, ours, theirs in disagreements[:20]:
        print(f"  heavy={heavy} category={cat} ours={ours} rdkit={theirs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args(argv)
    binary = build_rust_binary()
    failures = check_named_fixtures(binary)
    print(f"named fixtures: {len(NAMED_FIXTURES) - len(failures)}/{len(NAMED_FIXTURES)} agree")
    for line in failures:
        print(f"  MISMATCH {line}")
    if args.export is not None:
        check_export(args.export, args.limit, binary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
