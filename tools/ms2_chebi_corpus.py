"""ChEBI 3-star SDF ingestion for the database-first retrieval harness.

Implements the ChEBI arm of docs/MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md
Experiment A: one pinned 3-star structure export, standardized under the same
versioned chemistry domain as the synthetic and MassSpecGym runs, queried
with controlled open-subgraph patterns (parent-relative H counts).

Run:   python3 tools/ms2_chebi_corpus.py --sdf data/pinned/chebi_3_stars.sdf.gz \\
           --max-records 15000 --max-queries 100
Test:  python3 -m unittest tools.ms2_chebi_corpus

Format notes (V2000 molfiles, parsed with stdlib only):
- Bond type 4 (aromatic) is rejected as aromatic_bond: no silent
  kekulization, same boundary as the SMILES front end.
- Bond types 5-8 (query/ambiguous orders) are rejected as
  unsupported_bond_type: an unknown order is not guessed.
- Any nonzero charge (atom-block code, M CHG, or ... ) is rejected as
  unsupported_charge; doublet radicals as unsupported_radical.
- Any nonzero mass difference or M ISO entry is rejected as
  unsupported_isotope. Explicit H/D/T atoms are rejected (explicit_H).
- R-groups / pseudoatoms (R, *, A, Q, L, X) are rejected as
  unsupported_element.
- Disconnected molfiles (salts, mixtures) are rejected as
  disconnected_or_salt by the shared standardization.
- Property fields (formula, charge, monoisotopic mass, InChIKey) are
  retained as metadata and cross-checked, never trusted blindly: the
  neutral mass used for filtering is computed from the validated graph
  with the repository integer arithmetic.
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import time
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_database_retrieval import (
    DatabaseIndex,
    WorkCounters,
    formula_key,
    rank_baselines,
    run_query,
    standardize_record,
)
from tools.ms2_msgym_corpus import (
    SmilesUnsupported,
    extract_patterns,
    parse_formula,
    typed_from_parsed,
)

# V2000 atom-block charge codes -> formal charge (4 = doublet radical).
CHARGE_CODE = {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}


def parse_mol_block(lines):
    """Parse V2000 mol lines -> (atoms, bonds) or raise SmilesUnsupported.

    atoms: dicts compatible with typed_from_parsed. bonds: (a, b, order)
    with order in (1, 2, 3) or the raw ambiguous code (rejected downstream).
    """
    if len(lines) < 4:
        raise SmilesUnsupported("mol_too_short")
    # Fixed-width V2000 counts (atom/bond totals can exceed 3 digits and
    # run together, e.g. "114127" = 114 atoms, 127 bonds).
    try:
        natoms = int(lines[3][0:3])
        nbonds = int(lines[3][3:6])
    except ValueError:
        raise SmilesUnsupported("bad_counts_line")
    if len(lines) < 4 + natoms + nbonds:
        raise SmilesUnsupported("mol_truncated")
    atoms = []
    for i in range(natoms):
        line = lines[4 + i]
        # Fixed-width V2000 fields: 3-digit indices run together on large
        # molecules (e.g. "97100" = atoms 97 and 100), so split() fails.
        symbol = line[31:34].strip() or (line.split()[3:4] or [""])[0]
        try:
            mass_diff = int(line[34:36]) if line[34:36].strip() else 0
            charge_code = int(line[36:39]) if line[36:39].strip() else 0
        except ValueError:
            parts = line.split()
            try:
                mass_diff = int(parts[4]) if len(parts) > 4 else 0
                charge_code = int(parts[5]) if len(parts) > 5 else 0
            except ValueError:
                raise SmilesUnsupported("bad_atom_line")
        if not symbol:
            raise SmilesUnsupported("bad_atom_line")
        if mass_diff != 0:
            raise SmilesUnsupported("unsupported_isotope")
        if charge_code == 4:
            raise SmilesUnsupported("unsupported_radical")
        if charge_code not in CHARGE_CODE:
            raise SmilesUnsupported(f"bad_charge_code:{charge_code}")
        if symbol in ("D", "T"):
            raise SmilesUnsupported("unsupported_isotope")
        if symbol == "H" and CHARGE_CODE.get(charge_code, 0) != 0:
            raise SmilesUnsupported("unsupported_charge")
        valence = {"C": 4, "N": 3, "O": 2, "F": 1, "Cl": 1, "Br": 1,
                   "I": 1, "S": 2}.get(symbol, 0)
        atoms.append({"element": symbol, "aromatic": False, "bracket": False,
                      "hydrogens": 0, "charge": CHARGE_CODE[charge_code],
                      "valence": valence})
    bonds = []
    for i in range(nbonds):
        line = lines[4 + natoms + i]
        parsed = None
        if len(line) >= 9 and line[0:9].strip():
            try:
                parsed = (int(line[0:3]), int(line[3:6]), int(line[6:9]))
            except ValueError:
                parsed = None
        if parsed is None:
            parts = line.split()
            if len(parts) < 3:
                raise SmilesUnsupported("bad_bond_line")
            try:
                parsed = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                raise SmilesUnsupported("bad_bond_line")
        a, b, t = parsed[0] - 1, parsed[1] - 1, parsed[2]
        if not (0 <= a < natoms and 0 <= b < natoms):
            raise SmilesUnsupported("bad_bond_index")
        from tools.ms2_database_retrieval import AROM_ORDER

        bonds.append((a, b, AROM_ORDER if t == 4 else t))
    # M-block extensions: CHG (charge) and ISO (isotope veto).
    for line in lines[4 + natoms + nbonds:]:
        if line.startswith("M  CHG"):
            parts = line.split()
            try:
                n = int(parts[2])
            except (IndexError, ValueError):
                raise SmilesUnsupported("bad_m_chg")
            for k in range(n):
                try:
                    ai = int(parts[3 + 2 * k]) - 1
                    ch = int(parts[4 + 2 * k])
                except (IndexError, ValueError):
                    raise SmilesUnsupported("bad_m_chg")
                if not (0 <= ai < natoms):
                    raise SmilesUnsupported("bad_m_chg_index")
                atoms[ai]["charge"] = ch
        elif line.startswith("M  ISO"):
            raise SmilesUnsupported("unsupported_isotope")
        elif line.startswith("M  END"):
            break
    return _fold_explicit_hydrogens(atoms, bonds)


def _fold_explicit_hydrogens(atoms, bonds):
    """Fold explicit H into heavy-atom valence; D/T rejected as isotopes.

    Total H per heavy atom is valence - heavy bond-order sum, which holds
    regardless of the explicit/implicit mix, so explicit H only needs
    validation (order-1, exactly one heavy neighbor, uncharged). Nothing
    is ever dropped silently: unbonded or H-H-bonded H raises.
    """
    for a in atoms:
        if a["element"] in ("D", "T"):
            raise SmilesUnsupported("unsupported_isotope")
        if a["element"] == "H" and a["charge"] != 0:
            raise SmilesUnsupported("unsupported_charge")
    h_bonds = {}
    for a, b, o in bonds:
        ea, eb = atoms[a]["element"], atoms[b]["element"]
        if ea == "H" or eb == "H":
            h, other = (a, b) if ea == "H" else (b, a)
            if o != 1 or atoms[other]["element"] == "H":
                raise SmilesUnsupported("explicit_h_mismatch")
            h_bonds[h] = h_bonds.get(h, 0) + 1
    for i, a in enumerate(atoms):
        if a["element"] == "H" and h_bonds.get(i, 0) != 1:
            raise SmilesUnsupported("explicit_h_mismatch")
    keep = {}
    new_atoms = []
    for i, a in enumerate(atoms):
        if a["element"] == "H":
            continue
        keep[i] = len(new_atoms)
        new_atoms.append(a)
    new_bonds = [(keep[a], keep[b], o) for a, b, o in bonds
                 if atoms[a]["element"] != "H" and atoms[b]["element"] != "H"]
    return new_atoms, new_bonds


def iter_sdf_records(handle):
    """Yield (mol_lines, properties) per `$$$$`-delimited record."""
    mol_lines, props, current = [], {}, None
    for raw in handle:
        line = raw.rstrip("\n")
        if line.strip() == "$$$$":
            yield mol_lines, props
            mol_lines, props, current = [], {}, None
            continue
        if line.startswith("> <"):
            name = line[3:].rstrip(">")
            props[name] = []
            current = name
        elif line.startswith("M  END"):
            mol_lines.append(line)
            current = None
        elif current is not None:
            if line == "":
                continue
            props[current].append(line)
        else:
            mol_lines.append(line)
    if mol_lines or props:
        yield mol_lines, props


def first_prop(props, *names):
    for name in names:
        if name in props and props[name]:
            return props[name][0].strip()
    return ""


def standardize_chebi_record(mol_lines, props, mid):
    """Standardize one ChEBI record. Returns (record, exclusion_reason)."""
    try:
        atoms, bonds = parse_mol_block(mol_lines)
    except SmilesUnsupported as exc:
        return None, exc.reason
    typed, result = typed_from_parsed(atoms, bonds)
    if typed is None:
        return None, result
    # Database-pass size bound (distinct from any generation domain): giants
    # stay countable for mass/formula stages but out of exact matching.
    if len(typed) > 100:
        return None, "above_size_cap"
    raw = {"id": mid, "source": "chebi-3star", "family": "chebi",
           "atom_types": typed, "edges": result, "charge": 0,
           "original": {
               "chebi_id": first_prop(props, "ChEBI ID"),
               "name": first_prop(props, "ChEBI NAME"),
               "smiles": first_prop(props, "SMILES"),
               "inchikey": first_prop(props, "INCHIKEY"),
               "formula_property": first_prop(props, "FORMULA"),
               "mass_property": first_prop(props, "MONOISOTOPIC_MASS"),
               "star": first_prop(props, "STAR"),
               # Literature-frequency prior for the cheap ranker: synonym
               # and cross-reference counts (database presence, not a
               # stability or correctness claim about the structure).
               "n_synonyms": sum(len(v.split(";")) for v in props.get("SYNONYM", [])),
               "n_xrefs": sum(len(v.split(";")) for k, v in props.items()
                              if ("Links" in k or "Numbers" in k)
                              for v in v),
           }}
    record, reason = standardize_record(raw)
    if record is None:
        return None, reason
    if raw["original"]["star"] not in ("3", ""):
        return None, f"star_mismatch:{raw['original']['star']}"
    # Formula cross-check against the ChEBI Formula property (independent
    # column validates the implicit-H reconstruction). Dot-formulas name
    # multi-component materials (salts, co-crystals, hydrates), correctly
    # outside the single-molecule domain.
    fprop = raw["original"]["formula_property"]
    if fprop:
        if "." in fprop:
            return None, "multicomponent_formula"
        try:
            expected = parse_formula(fprop)
        except Exception:
            return None, "formula_unparseable"
        if formula_key(record["formula"]) != formula_key(expected):
            return None, "formula_mismatch"
    record["family"] = "chebi"
    return record, None


def run_chebi_experiment(sdf_path, max_records, max_queries):
    """Capped ChEBI run: full-index in-pool queries + out-of-pool slice."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    opener = gzip.open if sdf_path.endswith(".gz") else open
    records, excluded = [], Counter()
    seen_skeletons = {}
    dup_inchikey = 0
    n_scanned = 0
    with opener(sdf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for mol_lines, props in iter_sdf_records(handle):
            if n_scanned >= max_records:
                break
            n_scanned += 1
            chebi_id = first_prop(props, "ChEBI ID") or f"CHEBI-ROW{n_scanned}"
            key = first_prop(props, "INCHIKEY")[:14]
            if key and key in seen_skeletons:
                dup_inchikey += 1
                continue
            rec, reason = standardize_chebi_record(mol_lines, props, chebi_id)
            if rec is None:
                excluded[reason.split(":")[0]] += 1
                continue
            if key:
                seen_skeletons[key] = chebi_id
            try:
                from decimal import Decimal

                # Theoretical-vs-theoretical: only property display rounding
                # applies (4-5 decimals), so 500 uDa is generous slack that
                # can never mask a real composition error (mDa-scale+).
                obs_mu = int(Decimal(rec["original"]["mass_property"]) * 1_000_000)
                du = abs(obs_mu - rec["mass"])
            except Exception:
                du = None
            rec["mass_vs_property_uda"] = du
            if du is not None and du > 500:
                excluded["mass_property_mismatch"] += 1
                continue
            records.append(rec)
    index = DatabaseIndex(records)
    queries = []
    for rec in records:
        if len(queries) >= max_queries:
            break
        single, pair = extract_patterns(rec["atom_types"], rec["edges"])
        queries.append((rec["id"], rec,
                        [], [single] if single else [],
                        [single, pair] if pair else ([single] if single else [])))
    rows = []
    import sys as _sys

    for qi, (tid, target, p0, p1, p2) in enumerate(queries):
        for qname, pats in (("mass_only", p0), ("single_bond", p1),
                            ("bonded_pair", p2)):
            counters = WorkCounters(limit=100_000)
            res = run_query(index, target, pats, "unknown", None,
                            counters=counters)
            rank, in_s3 = rank_baselines(res["s3"], tid, index)
            rows.append({
                "query": f"{tid}:{qname}",
                "n_index": len(index),
                "n_s1": len(res["s1"]), "n_s2": len(res["s2"]),
                "n_s3": len(res["s3"]),
                "recall_s1": res["recall"]["s1"], "recall_s3": res["recall"]["s3"],
                "zero_status": res["zero_status"],
                "top1_uniform": rank.get("uniform@1"),
                "top1_fingerprint": rank.get("fingerprint@1"),
                "heavy": target["heavy"],
                "truncated": res["truncated"],
                "embedding_nodes": counters.embedding_nodes,
            })
        if (qi + 1) % 20 == 0 or qi + 1 == len(queries):
            _sys.stderr.write(f"# progress {qi + 1}/{len(queries)}\n")
            _sys.stderr.flush()
    # Out-of-pool slice: first min(20, n) queries rerun with target removed.
    oop = []
    for tid, target, p0, p1, p2 in queries[:20]:
        sub = DatabaseIndex([r for r in records if r["id"] != tid])
        for qname, pats in (("mass_only", p0), ("bonded_pair", p2)):
            counters = WorkCounters(limit=100_000)
            res = run_query(sub, target, pats, "unknown", None,
                            counters=counters)
            oop.append({"query": f"{tid}:{qname}|oop",
                        "n_s3": len(res["s3"]),
                        "zero_status": res["zero_status"],
                        "truncated": res["truncated"]})
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return {
        "rows": rows, "oop_rows": oop,
        "n_scanned": n_scanned, "n_index": len(records),
        "n_queries": len(queries), "dup_inchikey": dup_inchikey,
        "excluded": dict(excluded), "wall_s": wall, "cpu_s": cpu,
    }


def main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--max-records", type=int, default=15000)
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    result = run_chebi_experiment(args.sdf, args.max_records, args.max_queries)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
    import json

    from collections import Counter as _Counter

    oop_stat = _Counter((o["query"].rsplit(":", 1)[1].split("|")[0], o["zero_status"])
                        for o in result["oop_rows"])
    summary = {k: v for k, v in result.items() if k not in ("rows", "oop_rows")}
    summary["oop_status_counts"] = {f"{a}/{s}": n for (a, s), n in sorted(oop_stat.items())}
    summary["n_oop"] = len(result["oop_rows"])
    sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------
# Tests (hand-written mol blocks; no external data).
# --------------------------------------------------------------------------

ETHANOL_MOL = """ethanol
  test
  test
  3  2  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END""".splitlines()


def _props(**kw):
    return {k: [v] for k, v in kw.items()}


class ChebiTests(unittest.TestCase):
    def test_ethanol_mol(self):
        atoms, bonds = parse_mol_block(ETHANOL_MOL)
        self.assertEqual(len(atoms), 3)
        rec, reason = standardize_chebi_record(
            ETHANOL_MOL, _props(**{"FORMULA": "C2H6O", "STAR": "3"}), "CHEBI:X")
        self.assertIsNotNone(rec, reason)
        self.assertEqual(rec["formula"]["H"], 6)

    def test_explicit_h_folded(self):
        lines = [
            "x", "x", "x",
            "  3  2  0  0  0  0  0  0  0  0  0  0",
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
            "    1.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
            "    2.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
            "  1  2  1  0  0  0  0",
            "  2  3  1  0  0  0  0",
            "M  END",
        ]
        rec, reason = standardize_chebi_record(
            lines, _props(**{"FORMULA": "CH4O", "STAR": "3"}), "CHEBI:X")
        self.assertIsNotNone(rec, reason)
        f = rec["formula"]
        self.assertEqual((f["C"], f["H"], f["O"]), (1, 4, 1))

    def test_dangling_h_rejected(self):
        lines = [
            "x", "x", "x",
            "  2  0  0  0  0  0  0  0  0  0  0  0",
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
            "    5.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
            "M  END",
        ]
        try:
            parse_mol_block(lines)
            self.fail("expected SmilesUnsupported")
        except Exception as exc:
            self.assertEqual(getattr(exc, "reason", ""), "explicit_h_mismatch")

    def test_aromatic_bond_rejected(self):
        lines = [l.replace("  1  2  1", "  1  2  4") if l.startswith("  1  2") else l
                 for l in ETHANOL_MOL]
        rec, reason = standardize_chebi_record(
            lines, _props(**{"FORMULA": "C2H6O"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "aromatic_bond")

    def test_charge_rejected(self):
        lines = [l[:30] + " N   0  3" if l.split()[3:4] == ["O"] else l
                 for l in ETHANOL_MOL]
        rec, reason = standardize_chebi_record(
            lines, _props(**{"FORMULA": "C2H6ON"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "unsupported_charge")

    def test_m_chg_rejected(self):
        lines = [l for l in ETHANOL_MOL if not l.startswith("M  END")]
        lines = lines + ["M  CHG  1   3   1", "M  END"]
        rec, reason = standardize_chebi_record(
            lines, _props(**{"FORMULA": "C2H6O"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "unsupported_charge")

    def test_m_iso_rejected(self):
        lines = [l for l in ETHANOL_MOL if not l.startswith("M  END")]
        lines = lines + ["M  ISO  1   1  13", "M  END"]
        rec, reason = standardize_chebi_record(
            lines, _props(**{"FORMULA": "C2H6O"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "unsupported_isotope")

    def test_formula_mismatch(self):
        rec, reason = standardize_chebi_record(
            ETHANOL_MOL, _props(**{"FORMULA": "C2H6O2"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "formula_mismatch")

    def test_record_splitting(self):
        doc = "> <ChEBI ID>\nCHEBI:1\n\n$$$$\n> <ChEBI ID>\nCHEBI:2\n\n$$$$\n"
        import io

        out = list(iter_sdf_records(io.StringIO(doc)))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1]["ChEBI ID"], ["CHEBI:1"])

    def test_three_digit_bond_indices(self):
        # Fixed-width "%3d%3d%3d": "97100  1" = atoms 97 and 100, order 1.
        atoms = [{"element": "C", "aromatic": False, "bracket": False,
                  "hydrogens": 0, "charge": 0} for _ in range(100)]
        lines = ["h", "h", "h", "100  1  0  0  0  0            999 V2000"]
        lines += ["    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0"
                  for _ in range(100)]
        lines += [" 97100  1  0  0  0  0", "M  END"]
        pa, pb = parse_mol_block(lines)
        self.assertEqual(len(pa), 100)
        self.assertEqual(pb[-1], (96, 99, 1))

    def test_multicomponent_formula_labeled(self):
        rec, reason = standardize_chebi_record(
            ETHANOL_MOL, _props(**{"FORMULA": "C2H6O.H2O", "STAR": "3"}), "CHEBI:X")
        self.assertIsNone(rec)
        self.assertEqual(reason, "multicomponent_formula")

    def test_hf_admitted_atomic_f_rejected(self):
        hf = [
            "x", "x", "x",
            "  2  1  0  0  0  0  0  0  0  0  0  0",
            "    0.0000    0.0000    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0",
            "    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
            "  1  2  1  0  0  0  0",
            "M  END",
        ]
        rec, reason = standardize_chebi_record(
            hf, _props(**{"FORMULA": "HF", "STAR": "3"}), "CHEBI:29228")
        self.assertIsNotNone(rec, reason)
        bare = [l for l in hf if " H " not in l]
        bare[3] = "  1  0  0  0  0  0  0  0  0  0  0  0"
        rec2, reason2 = standardize_chebi_record(
            bare, _props(**{"FORMULA": "F", "STAR": "3"}), "CHEBI:24061")
        self.assertIsNone(rec2)
        self.assertEqual(reason2, "formula_mismatch")

    def test_hydrogen_chloride_admitted(self):
        hcl = [
            "x", "x", "x",
            "  2  1  0  0  0  0  0  0  0  0  0  0",
            "    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0",
            "    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
            "  1  2  1  0  0  0  0",
            "M  END",
        ]
        rec, reason = standardize_chebi_record(
            hcl, _props(**{"FORMULA": "HCl", "STAR": "3"}), "CHEBI:X")
        self.assertIsNotNone(rec, reason)
        self.assertEqual(rec["formula"]["Cl"], 1)


if __name__ == "__main__":
    main(sys.argv[1:])
