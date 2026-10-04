"""Tests for `export_msgym.py --split msgym-split-v1` (tasks D3+D4).

Runnable with:
    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy --with pyarrow \
        python tools/ms2/test_export_msgym.py

The synthetic TSV is built from real in-domain SMILES taken from the first
rows of the pinned TSV (keys varied, one key shared by two SMILES, one
test-fold row). Plain asserts only. Synthetic identity blocks are 14
uppercase letters (valid normalised identities). Test-fold rows are never
read for contents: the synthetic test row uses a fixed key and is only
checked for absence.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIN = REPO / "data/pinned/MassSpecGym1.5.tsv"
TOOL = Path(__file__).resolve().parent / "export_msgym.py"
TABLE_TOOL = Path(__file__).resolve().parent / "formula_table_msgym.py"


def real_smiles(n):
    """First n distinct in-domain SMILES of the pinned TSV with their fields."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rdkit import Chem, RDLogger  # noqa: E402
    import ms2_reference as ref  # noqa: E402
    RDLogger.DisableLog("rdApp.*")
    order, fields = [], {}
    with open(PIN) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            smi, ad = p[col["smiles"]], p[col["adduct"]]
            if smi in fields:
                continue
            if ad not in ("[M+H]+", "[M-H]-"):
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                Chem.Kekulize(m, clearAromaticFlags=True)
            except Exception:
                continue
            if ref.classify(m):
                continue
            order.append(smi)
            fields[smi] = (p[col["mzs"]], p[col["intensities"]], p[col["precursor_mz"]],
                           ad, p[col["instrument_type"]], p[col["collision_energy"]])
            if len(order) >= n:
                break
    assert len(order) >= n, "pinned TSV has too few in-domain SMILES"
    return order, fields


def normalize_block(raw):
    """Independent normalisation (no import from the tools)."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    block = text.split("-", 1)[0].strip().upper()
    if len(block) != 14:
        return None
    if any(not ("A" <= c <= "Z") for c in block):
        return None
    return block


def expected_part(key14, fold):
    """Independent recomputation of the split rule (normalised, no import)."""
    block = normalize_block(key14)
    assert block is not None, key14
    h = int.from_bytes(hashlib.sha256(block.encode()).digest()[:8], "big")
    if fold == "train":
        return "rank" if h % 5 == 0 else "fit"
    assert fold == "val", fold
    return "calibration" if h % 2 == 0 else "report"


def _letters(n, length):
    s = ""
    for _ in range(length):
        s = chr(65 + n % 26) + s
        n //= 26
    return s


def pick_keys(n, fold):
    prefix = "SYT" if fold == "train" else "SYV"
    keys, i = [], 0
    while len(keys) < n:
        k = prefix + _letters(i, 11)
        assert len(k) == 14 and normalize_block(k) == k, k
        keys.append(k)
        i += 1
    parts = {expected_part(k, fold) for k in keys}
    want = {"fit", "rank"} if fold == "train" else {"calibration", "report"}
    while not want.issubset(parts) and len(keys) < n + 200:
        k = prefix + _letters(i, 11)
        i += 1
        keys.append(k)
        parts.add(expected_part(k, fold))
    return keys


def write_synth_tsv(path, order, fields):
    train_smi = order[:6]
    val_smi = order[6:10]
    test_smi = order[10:11]
    train_keys = pick_keys(6, "train")
    val_keys = pick_keys(4, "val")
    train_keys[1] = train_keys[0]  # two SMILES sharing one InChIKey block
    with open(PIN) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    with open(path, "w") as out:
        out.write("\t".join(header) + "\n")
        rid = 0

        def row(smi, key, fold):
            nonlocal rid
            mzs, its, prec, ad, inst, ce = fields[smi]
            out.write("\t".join([f"SYN{rid:05d}", mzs, its, smi, key, "", "",
                                  "", prec, ad, inst, ce, fold, "False"]) + "\n")
            rid += 1

        for smi, k in zip(train_smi, train_keys):
            row(smi, k, "train")
        row(train_smi[0], train_keys[0], "train")
        for smi, k in zip(val_smi, val_keys):
            row(smi, k, "val")
        row(test_smi[0], "SYXTESTFOLDXXXX", "test")
    return {"train_smi": train_smi, "val_smi": val_smi, "test_smi": test_smi,
            "train_keys": train_keys, "val_keys": val_keys}


def run_export(tsv, out_dir, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    r = subprocess.run([sys.executable, str(TOOL), "--tsv", str(tsv),
                        "--out-dir", str(out_dir), *extra],
                       capture_output=True, text=True, cwd=str(REPO), env=env)
    assert r.returncode == 0, f"export failed: {r.stderr[-3000:]}"
    return r


def run_table(tsv, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    r = subprocess.run([sys.executable, str(TABLE_TOOL), "--tsv", str(tsv), *extra],
                       capture_output=True, text=True, cwd=str(REPO), env=env)
    return r


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def molecules_sha(doc):
    return hashlib.sha256(
        json.dumps(doc["molecules"], separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    order, fields = real_smiles(14)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tsv = tmp / "synth.tsv"
        meta = write_synth_tsv(tsv, order, fields)

        split_dir = tmp / "split1"
        run_export(tsv, split_dir, "--name", "t", "--split", "msgym-split-v1",
                   "--fit-molecules", "100", "--rank-molecules", "100",
                   "--calibration-molecules", "100", "--report-molecules", "100",
                   "--spectra-per-molecule", "1")
        parts = {}
        for p in ("fit", "rank", "calibration", "report"):
            fp = split_dir / f"t_{p}.json"
            assert fp.exists(), f"missing {fp}"
            doc = json.loads(fp.read_text())
            assert doc["subset"] == p, (fp, doc["subset"])
            assert doc.get("split") == "msgym-split-v1", fp
            assert "sha256(inchikey14)" in doc.get("split_rule", ""), fp
            assert "fold_conflict_identities" in doc, fp
            assert "invalid_inchikey" in doc, fp
            parts[p] = doc

        mols = {p: {m["smiles"]: m for m in doc["molecules"]} for p, doc in parts.items()}
        keys = {p: {m["key"] for m in doc["molecules"]} for p, doc in parts.items()}

        # 1. parts are disjoint by InChIKey block (and by molecule).
        for a in keys:
            for b in keys:
                if a < b:
                    assert not keys[a] & keys[b], (a, b, keys[a] & keys[b])
        seen = set()
        for p, d in mols.items():
            assert not (set(d) & seen), p
            seen |= set(d)
        print("PASS parts disjoint by InChIKey block")

        # 2. two SMILES with the same key share a part.
        shared = meta["train_keys"][0]
        holders = [p for p in mols
                   if meta["train_smi"][0] in mols[p] and meta["train_smi"][1] in mols[p]]
        assert len(holders) == 1, holders
        assert shared in keys[holders[0]]
        print(f"PASS same-key SMILES share part {holders[0]}")

        # 3. the 80/20 and 50/50 rule matches an independent recomputation.
        key_fold = {}
        for smi, k in zip(meta["train_smi"], meta["train_keys"]):
            key_fold[smi] = (k, "train")
        for smi, k in zip(meta["val_smi"], meta["val_keys"]):
            key_fold[smi] = (k, "val")
        for p, d in mols.items():
            for smi in d:
                k, fold = key_fold[smi]
                assert expected_part(k, fold) == p, (smi, k, fold, p)
        train_parts = [expected_part(k, "train") for k in meta["train_keys"]]
        val_parts = [expected_part(k, "val") for k in meta["val_keys"]]
        assert set(train_parts) == {"fit", "rank"}, train_parts
        assert set(val_parts) == {"calibration", "report"}, val_parts
        print("PASS 80/20 and 50/50 rule matches independent recomputation")

        # 4. a test-fold row never appears (neither its SMILES nor its key).
        for p, d in mols.items():
            assert meta["test_smi"][0] not in d, p
            assert "SYXTESTFOLDXXXX" not in keys[p], p
        print("PASS test-fold row never appears")

        # 5. determinism: two runs are byte-equal.
        split_dir2 = tmp / "split2"
        run_export(tsv, split_dir2, "--name", "t", "--split", "msgym-split-v1",
                   "--fit-molecules", "100", "--rank-molecules", "100",
                   "--calibration-molecules", "100", "--report-molecules", "100",
                   "--spectra-per-molecule", "1")
        for p in ("fit", "rank", "calibration", "report"):
            a = (split_dir / f"t_{p}.json").read_bytes()
            b = (split_dir2 / f"t_{p}.json").read_bytes()
            assert a == b, p
        print("PASS determinism (two split runs byte-equal)")

        # 6. --split absent still exports train/validation with the new
        # provenance keys (molecules arrays unchanged by the D4 refactor on
        # conflict-free valid inputs).
        nosplit_dir = tmp / "nosplit"
        r = run_export(tsv, nosplit_dir, "--name", "base",
                       "--train-molecules", "100", "--validation-molecules", "100",
                       "--spectra-per-molecule", "1")
        for subset in ("train", "validation"):
            doc = json.loads((nosplit_dir / f"base_{subset}.json").read_text())
            assert doc["subset"] == subset
            assert doc["fold_conflict_molecules"] == 0, doc
            assert doc["fold_conflict_identities"] == 0, doc
            assert doc["invalid_inchikey"] == 0, doc
            assert len(doc["molecules"]) > 0, subset
        # determinism of the no-split path (molecules arrays byte-equal).
        nosplit_dir2 = tmp / "nosplit2"
        run_export(tsv, nosplit_dir2, "--name", "base",
                   "--train-molecules", "100", "--validation-molecules", "100",
                   "--spectra-per-molecule", "1")
        for subset in ("train", "validation"):
            a = json.loads((nosplit_dir / f"base_{subset}.json").read_text())
            b = json.loads((nosplit_dir2 / f"base_{subset}.json").read_text())
            assert molecules_sha(a) == molecules_sha(b), subset
        print("PASS no-split exports with identity provenance")

        # 7. E1: two SMILES variants of one identity block in train and val
        # are both excluded (fold conflict by identity, not by SMILES).
        with open(PIN) as fh:
            header = fh.readline().rstrip("\n").split("\t")
        conflict_tsv = tmp / "conflict.tsv"
        conflict_block = "CONFLICTBLOCKA"  # 14 letters
        assert normalize_block(conflict_block) == conflict_block
        with open(conflict_tsv, "w") as out:
            out.write("\t".join(header) + "\n")
            rid = 0

            def crow(smi, key, fold):
                nonlocal rid
                mzs, its, prec, ad, inst, ce = fields[smi]
                out.write("\t".join([f"CFL{rid:05d}", mzs, its, smi, key, "", "",
                                      "", prec, ad, inst, ce, fold, "False"]) + "\n")
                rid += 1

            # distinct SMILES sharing one block across folds
            smi_a, smi_b = order[0], order[1]
            assert smi_a != smi_b
            crow(smi_a, conflict_block, "train")
            crow(smi_b, conflict_block, "val")
            # controls that must survive: one train-only and one val-only block
            crow(order[2], pick_keys(1, "train")[0], "train")
            crow(order[3], pick_keys(1, "val")[0], "val")
        cdir = tmp / "conflict"
        run_export(conflict_tsv, cdir, "--name", "c", "--split", "msgym-split-v1",
                   "--fit-molecules", "100", "--rank-molecules", "100",
                   "--calibration-molecules", "100", "--report-molecules", "100",
                   "--spectra-per-molecule", "1")
        all_mols = {}
        for p in ("fit", "rank", "calibration", "report"):
            doc = json.loads((cdir / f"c_{p}.json").read_text())
            for m in doc["molecules"]:
                all_mols[m["smiles"]] = p
            assert doc["fold_conflict_identities"] == 1, (p, doc)
        assert smi_a not in all_mols, smi_a
        assert smi_b not in all_mols, smi_b
        assert order[2] in all_mols and order[3] in all_mols
        print("PASS cross-fold same-block variants both excluded")

        # 8. E1+E3: a full 27-char InChIKey and its 14-char block land in the
        # same part (both tools), case-insensitively; invalid blocks raise.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import export_msgym as ex  # noqa: E402
        import formula_table_msgym as ft  # noqa: E402
        block = pick_keys(1, "train")[0]
        full_a = f"{block}-AAAAAAAAAA-N"
        full_b = f"{block.lower()}-0000000002-n"
        for mod in (ex, ft):
            assert mod.part_of(full_a, "train") == mod.part_of(block, "train")
            assert mod.part_of(full_b, "train") == mod.part_of(block, "train")
            assert mod.normalize_identity(full_a) == block
            assert mod.normalize_identity(block.lower()) == block
            for bad in ("", "SHORT", "12345678901234", "ABC-DEF", "SYT0000000000",
                        "TOOLONGIDENTITYBLOCK", "ABCDEFGHIJKLM-EXTRA"):
                assert mod.normalize_identity(bad) is None, (mod, bad)
                try:
                    mod.part_of(bad, "train")
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{mod} accepted {bad!r}")
        print("PASS full InChIKey and block share a part; invalid rejected")

        # 9. E1+E3: rows with invalid blocks are skipped and counted.
        bad_tsv = tmp / "bad.tsv"
        with open(bad_tsv, "w") as out:
            out.write("\t".join(header) + "\n")
            rid = 0

            def brow(smi, key, fold):
                nonlocal rid
                mzs, its, prec, ad, inst, ce = fields[smi]
                out.write("\t".join([f"BAD{rid:05d}", mzs, its, smi, key, "", "",
                                      "", prec, ad, inst, ce, fold, "False"]) + "\n")
                rid += 1

            brow(order[0], pick_keys(1, "train")[0], "train")
            brow(order[1], "NOTVALID", "train")
            brow(order[2], "", "train")
            brow(order[3], "12345678901234", "val")
        bdir = tmp / "badexp"
        r = run_export(bad_tsv, bdir, "--name", "b",
                       "--train-molecules", "100", "--validation-molecules", "100",
                       "--spectra-per-molecule", "1")
        doc = json.loads((bdir / "b_train.json").read_text())
        assert doc["invalid_inchikey"] == 3, doc
        got = {m["smiles"] for m in doc["molecules"]}
        assert order[0] in got, got
        assert order[1] not in got and order[2] not in got, got
        print("PASS invalid InChIKey rows skipped and counted")

        # 10. E2: only fit/rank may supply table rows; calibration/report are
        # refused; conflicting aliases are an error. Every CLI choice tested.
        rep_fit = tmp / "rep_fit.json"
        tab_fit = tmp / "tab_fit.json"
        r = run_table(tsv, "--out", str(rep_fit), "--table-out", str(tab_fit),
                      "--split", "msgym-split-v1", "--part", "fit")
        assert r.returncode == 0, r.stderr[-2000:]
        assert tab_fit.exists() and len(json.loads(tab_fit.read_text())["rows"]) > 0
        rep_rank = tmp / "rep_rank.json"
        r = run_table(tsv, "--out", str(rep_rank),
                      "--split", "msgym-split-v1", "--part", "rank")
        assert r.returncode == 0, r.stderr[-2000:]
        # alias equivalence: --formula-table-from fit == --part fit rows
        rep_alias = tmp / "rep_alias.json"
        tab_alias = tmp / "tab_alias.json"
        r = run_table(tsv, "--out", str(rep_alias), "--table-out", str(tab_alias),
                      "--split", "msgym-split-v1", "--formula-table-from", "fit")
        assert r.returncode == 0, r.stderr[-2000:]
        assert (json.loads(tab_alias.read_text())["rows"]
                == json.loads(tab_fit.read_text())["rows"])
        print("PASS table fit/rank supply rows; alias matches")
        for bad_part in ("calibration", "report"):
            r = run_table(tsv, "--out", str(tmp / f"rep_{bad_part}.json"),
                          "--split", "msgym-split-v1", "--part", bad_part)
            assert r.returncode != 0, bad_part
            assert "evaluation data" in r.stderr, (bad_part, r.stderr[-1000:])
            r = run_table(tsv, "--out", str(tmp / f"rep_{bad_part}b.json"),
                          "--split", "msgym-split-v1",
                          "--formula-table-from", bad_part)
            assert r.returncode != 0, bad_part
            assert "evaluation data" in r.stderr, (bad_part, r.stderr[-1000:])
        print("PASS table calibration/report refused as evaluation data")
        # conflicting aliases
        r = run_table(tsv, "--out", str(tmp / "rep_conf.json"),
                      "--split", "msgym-split-v1",
                      "--part", "fit", "--formula-table-from", "rank")
        assert r.returncode != 0, "conflicting aliases accepted"
        assert "conflict" in r.stderr.lower(), r.stderr[-1000:]
        print("PASS table conflicting aliases are an error")
        # missing / dangling options
        r = run_table(tsv, "--out", str(tmp / "rep_nopart.json"),
                      "--split", "msgym-split-v1")
        assert r.returncode != 0, "split without part accepted"
        r = run_table(tsv, "--out", str(tmp / "rep_nosplit.json"), "--part", "fit")
        assert r.returncode != 0, "part without split accepted"
        r = run_table(tsv, "--out", str(tmp / "rep_nosplit2.json"),
                      "--formula-table-from", "fit")
        assert r.returncode != 0, "alias without split accepted"
        print("PASS table split/part alias requirements enforced")
        # no-split table still works (legacy behaviour on valid inputs)
        r = run_table(tsv, "--out", str(tmp / "rep_legacy.json"))
        assert r.returncode == 0, r.stderr[-2000:]
        print("PASS table no-split legacy path works")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
