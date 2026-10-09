"""Motif sequences for a decoder that emits whole ring systems and groups.

The completion model builds a molecule one atom at a time. This tool defines
the coarser alphabet a motif decoder uses, and converts molecules to and from
it.

Motifs. A molecule (kekulized, heavy atoms only) is cut into

* ring systems: the connected components of its ring bonds, so fused,
  bridged and spiro rings stay in one piece;
* acyclic groups: connected sets of non-ring heteroatoms, of non-ring carbons
  that carry a multiple bond, and of non-ring carbons bonded to two or more
  heteroatoms (the core of Ertl's functional-group rule, restricted to
  non-ring atoms);
* single atoms: every other non-ring carbon.

Every bond between two motifs is then a non-ring bond, so the motifs form a
tree. A motif's identity is the canonical SMILES of the piece with each cut
bond replaced by hydrogens; `free[i]` is the number of hydrogens atom `i` of
that capped piece carries, which is exactly how much bond order the atom can
still spend on attachments.

Sequence. With `MOTIF m`, `ATOM a`, `BOND o` and `END` tokens:

    molecule := MOTIF m  body  END
    body     := ( ATOM a  BOND o  MOTIF m'  ATOM b  body  END )*

`ATOM a BOND o MOTIF m' ATOM b` attaches a new motif `m'` by a bond of order
`o` from atom `a` of the motif on top of the stack to atom `b` of the new
one, and pushes it; `END` pops. Atom numbers are those of the motif's
reference molecule (the parse of its identity SMILES). A sequence is valid
when every attachment finds `free >= o` at both ends (`tools/ms2` keeps this
rule in one place, `MotifMachine`, which the Rust decoder mask and the Lean
model `lean/MotifDecoder` mirror).

The sequence of a molecule is made canonical: the root is a centre of the
motif tree, children are ordered by their tokens, and a motif's own
symmetries are resolved by taking the smallest result over its automorphisms.
A molecule with a motif of `MAX_AUTOMORPHISMS` or more automorphisms is left
out, because that minimum could not be taken over all of them. The sequence
does not record stereochemistry: stereoisomers share one.

    python tools/ms2/motif_tokens.py build \
        --export data/ms2/specgen/msgym_train.json --export data/ms2/specgen/extra163k_structures.json \
        --apply data/ms2/specgen/msgym_validation.json --out data/ms2/specgen/motif

writes `<out>/vocab.json` (motifs seen in the `--export` files) and one
`<name>.motif.jsonl` per file with `{"index", "key", "tokens" | "skip"}`,
and reports vocabulary size, coverage and sequence lengths. A molecule is
kept only if assembling its tokens gives back the same molecule.

`prepare` then cuts the vocabulary at `--min-count` and writes, per export,
the rows `examples/ms2_motif_decoder.rs` trains on (token ids, formula, true
fingerprint bits, spectrum ids).
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MAX_MOTIF_ATOMS = 32
MAX_AUTOMORPHISMS = 2000
ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


class Skip(Exception):
    """A molecule this alphabet does not cover; the message is the reason."""


def plain(smiles: str) -> Chem.Mol:
    """Kekulized heavy-atom molecule without stereo marks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Skip("unparseable")
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0 or atom.GetIsotope() != 0:
            raise Skip("charged, radical or isotopic atom")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def identity(mol: Chem.Mol) -> str:
    """Stereo-free canonical SMILES."""
    copy = Chem.Mol(mol)
    Chem.SanitizeMol(copy)
    return Chem.MolToSmiles(copy, isomericSmiles=False)


def partition(mol: Chem.Mol) -> list[list[int]]:
    """Atom sets of the motifs (see the module docstring)."""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    in_ring = [a.IsInRing() for a in mol.GetAtoms()]
    marked = [False] * n
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if in_ring[i]:
            continue
        if atom.GetAtomicNum() != 6:
            marked[i] = True
            continue
        multiple = any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())
        hetero = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 6)
        marked[i] = multiple or hetero >= 2
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.IsInRing() or (marked[a] and marked[b] and not in_ring[a] and not in_ring[b]):
            parent[find(a)] = find(b)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def capped(mol: Chem.Mol, atoms: list[int]) -> tuple[Chem.Mol, list[int]]:
    """The motif on `atoms` with every cut bond replaced by hydrogens.

    Returns the sanitized piece (atoms in the order of `atoms`) and each
    atom's hydrogen count there."""
    index = {a: i for i, a in enumerate(atoms)}
    rw = Chem.RWMol()
    free = []
    for a in atoms:
        atom = mol.GetAtomWithIdx(a)
        cut = sum(ORDER[b.GetBondType()] for b in atom.GetBonds() if b.GetOtherAtomIdx(a) not in index)
        new = Chem.Atom(atom.GetAtomicNum())
        new.SetNumExplicitHs(atom.GetTotalNumHs() + cut)
        new.SetNoImplicit(True)
        rw.AddAtom(new)
        free.append(atom.GetTotalNumHs() + cut)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in index and b in index:
            rw.AddBond(index[a], index[b], bond.GetBondType())
    piece = rw.GetMol()
    try:
        Chem.SanitizeMol(piece)
    except Exception as error:  # a piece RDKit cannot represent on its own
        raise Skip(f"motif does not sanitize: {type(error).__name__}")
    return piece, free


_REFERENCE: dict[str, tuple[Chem.Mol, list[int], list[int]]] = {}


def reference(smiles: str) -> tuple[Chem.Mol, list[int], list[int]]:
    """Reference molecule of a motif identity, its free valences and elements."""
    hit = _REFERENCE.get(smiles)
    if hit is None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise Skip("motif identity does not parse")
        hit = (mol, [a.GetTotalNumHs() for a in mol.GetAtoms()], [a.GetAtomicNum() for a in mol.GetAtoms()])
        _REFERENCE[smiles] = hit
    return hit


def numberings(piece: Chem.Mol, free: list[int], smiles: str) -> list[tuple[int, ...]]:
    """Every map from the piece's atoms to the reference numbering that keeps
    elements, bonds and free valences: one per automorphism of the motif."""
    ref, ref_free, _ = reference(smiles)
    if ref.GetNumAtoms() != piece.GetNumAtoms():
        raise Skip("motif identity does not round-trip")
    out = []
    # match[i] is the reference atom that piece atom i lands on
    matches = ref.GetSubstructMatches(piece, uniquify=False, maxMatches=MAX_AUTOMORPHISMS)
    if len(matches) >= MAX_AUTOMORPHISMS:
        # The smallest sequence over a truncated list would depend on the
        # input's atom order, so the molecule is left out rather than given a
        # sequence that is not canonical.
        raise Skip("motif has too many symmetries")
    for match in matches:
        inverse = [0] * len(match)
        for piece_atom, ref_atom in enumerate(match):
            inverse[piece_atom] = ref_atom
        if all(ref_free[inverse[i]] == free[i] for i in range(len(free))):
            out.append(tuple(inverse))
    if not out:
        raise Skip("motif does not match its own identity")
    return out


def decompose(smiles: str) -> list:
    """Canonical token list of a molecule: `["M", identity]`, `["A", a]`,
    `["B", o]` and `["E"]` entries (motif ids are assigned by the vocabulary)."""
    mol = plain(smiles)
    if mol.GetNumAtoms() == 0:
        raise Skip("no heavy atom")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise Skip("disconnected")
    parts = partition(mol)
    owner = {}
    for m, atoms in enumerate(parts):
        if len(atoms) > MAX_MOTIF_ATOMS:
            raise Skip("motif too large")
        for a in atoms:
            owner[a] = m
    names, maps = [], []
    for atoms in parts:
        piece, free = capped(mol, atoms)
        name = Chem.MolToSmiles(piece)
        names.append(name)
        local = numberings(piece, free, name)
        maps.append([{atom: numbering[i] for i, atom in enumerate(atoms)} for numbering in local])
    links: dict[int, list[tuple[int, int, int, int]]] = collections.defaultdict(list)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if owner[a] != owner[b]:
            order = ORDER[bond.GetBondType()]
            links[owner[a]].append((a, order, owner[b], b))
            links[owner[b]].append((b, order, owner[a], a))
    if sum(len(v) for v in links.values()) != 2 * (len(parts) - 1):
        raise Skip("motifs do not form a tree")

    def body(m: int, came_from: int, entry: int | None):
        """Smallest `(entry number, tokens)` of motif `m` over its automorphisms."""
        children = []
        for a, order, other, b in links[m]:
            if other != came_from:
                children.append((a, order, other, b))
        subtrees = [body(other, m, b) for _, _, other, b in children]
        best = None
        for numbering in maps[m]:
            items = sorted(
                (numbering[a], order, names[other], sub_entry, sub_tokens)
                for (a, order, other, _), (sub_entry, sub_tokens) in zip(children, subtrees)
            )
            head = -1 if entry is None else numbering[entry]
            candidate = (head, items)
            if best is None or candidate < best:
                best = candidate
        head, items = best
        tokens = []
        for a, order, name, sub_entry, sub_tokens in items:
            tokens += [["A", a], ["B", order], ["M", name], ["A", sub_entry]] + sub_tokens + [["E"]]
        return head, tokens

    # Root at a centre of the motif tree (one or two candidates).
    degree = {m: len(links[m]) for m in range(len(parts))}
    alive = set(range(len(parts)))
    while len(alive) > 2:
        leaves = [m for m in alive if degree[m] <= 1]
        for m in leaves:
            alive.discard(m)
            for _, _, other, _ in links[m]:
                if other in alive:
                    degree[other] -= 1
    best = None
    for root in sorted(alive):
        _, tokens = body(root, -1, None)
        candidate = [["M", names[root]]] + tokens + [["E"]]
        key = json.dumps(candidate)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1]


class MotifMachine:
    """The stack machine that reads a token sequence and builds the molecule.

    The one place the validity rule lives on the Python side. `step` returns
    an error string instead of raising, so a caller can use it as a mask."""

    def __init__(self) -> None:
        self.rw = Chem.RWMol()
        self.stack: list[int] = []  # first atom of each open motif
        self.sizes: list[int] = []  # atoms of each open motif
        self.free: list[int] = []
        self.pending: tuple | None = None
        self.done = False

    def _add(self, smiles: str) -> int:
        ref, ref_free, _ = reference(smiles)
        start = self.rw.GetNumAtoms()
        for atom, h in zip(ref.GetAtoms(), ref_free):
            new = Chem.Atom(atom.GetAtomicNum())
            new.SetIsAromatic(atom.GetIsAromatic())
            new.SetNumExplicitHs(h)
            new.SetNoImplicit(True)
            self.rw.AddAtom(new)
            self.free.append(h)
        for bond in ref.GetBonds():
            self.rw.AddBond(start + bond.GetBeginAtomIdx(), start + bond.GetEndAtomIdx(), bond.GetBondType())
        return start

    def run(self, tokens: list) -> str | None:
        """Feed a whole sequence; `None` when it is valid and complete."""
        i = 0
        if not tokens or tokens[0][0] != "M":
            return "does not start with a motif"
        start = self._add(tokens[0][1])
        self.stack.append(start)
        self.sizes.append(self.rw.GetNumAtoms() - start)
        i = 1
        while i < len(tokens):
            if self.done:
                return "tokens after the end"
            if tokens[i][0] == "E":
                self.stack.pop()
                self.sizes.pop()
                self.done = not self.stack
                i += 1
                continue
            group = tokens[i : i + 4]
            if [t[0] for t in group] != ["A", "B", "M", "A"]:
                return "malformed attachment"
            a, order, name, b = group[0][1], group[1][1], group[2][1], group[3][1]
            if order not in TYPE or not 0 <= a < self.sizes[-1] or b < 0:
                return "attachment atom or bond out of range"
            source = self.stack[-1] + a
            start = self._add(name)
            size = self.rw.GetNumAtoms() - start
            if b >= size:
                return "entry atom out of range"
            target = start + b
            if self.free[source] < order or self.free[target] < order:
                return "no free valence"
            for atom in (source, target):
                self.free[atom] -= order
                self.rw.GetAtomWithIdx(atom).SetNumExplicitHs(self.free[atom])
            self.rw.AddBond(source, target, TYPE[order])
            self.stack.append(start)
            self.sizes.append(size)
            i += 4
        return None if self.done else "not closed"

    def molecule(self) -> Chem.Mol:
        mol = self.rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol


def assemble(tokens: list) -> str:
    """Stereo-free canonical SMILES of a token sequence (`Skip` if invalid)."""
    machine = MotifMachine()
    error = machine.run(tokens)
    if error:
        raise Skip(error)
    try:
        mol = machine.molecule()
    except Exception as failure:
        raise Skip(f"assembled molecule does not sanitize: {type(failure).__name__}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def convert(smiles: str) -> dict:
    """Tokens of one molecule, or the reason it is skipped."""
    try:
        tokens = decompose(smiles)
        if assemble(tokens) != identity(plain(smiles)):
            return {"skip": "round trip differs"}
        return {"tokens": tokens}
    except Skip as skip:
        return {"skip": str(skip)}
    except Exception as failure:  # counted, never silently dropped
        return {"skip": f"error: {type(failure).__name__}"}


def _convert_file(path: Path, workers: int) -> list[dict]:
    molecules = json.loads(path.read_text())["molecules"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(convert, [m["smiles"] for m in molecules], chunksize=256))
    for i, (row, molecule) in enumerate(zip(rows, molecules)):
        row["index"] = i
        row["key"] = molecule["key"]
    return rows


# Token ids of the sequence model's output side (`src/models/ms2/motif.rs`
# holds the same numbers): 0 is padding, then END, the three bond orders, 32
# attachment atoms, then the motifs in vocabulary order.
END_ID = 1
BOND_BASE = 1  # BOND o is 1 + o
ATOM_BASE = 5
MOTIF_BASE = 37
ELEMENTS = [6, 1, 7, 8, 9, 15, 16, 17, 35, 53]  # C H N O F P S Cl Br I


def token_ids(tokens: list) -> list[int]:
    out = []
    for token in tokens:
        if token[0] == "E":
            out.append(END_ID)
        elif token[0] == "B":
            out.append(BOND_BASE + token[1])
        elif token[0] == "A":
            out.append(ATOM_BASE + token[1])
        else:
            out.append(MOTIF_BASE + token[1])
    return out


def formula_counts(smiles: str) -> list[int] | None:
    """Element counts in `ELEMENTS` order, or `None` outside that alphabet."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = collections.Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetAtomicNum()] += 1
        counts[1] += atom.GetTotalNumHs()
    if any(z not in ELEMENTS for z in counts):
        return None
    return [counts[z] for z in ELEMENTS]


def prepare(args) -> int:
    """Write what the Rust driver reads: the vocabulary cut at `--min-count`
    and one row per molecule with output-side token ids, formula, true bits
    and spectrum ids."""
    vocab = json.loads((args.motif_dir / "vocab.json").read_text())["motifs"]
    keep = [i for i, m in enumerate(vocab) if m["count"] >= args.min_count]
    remap = {old: new for new, old in enumerate(keep)}
    for i in keep:
        if any(z not in ELEMENTS for z in vocab[i]["elements"]):
            raise SystemExit(f"motif {vocab[i]['smiles']} holds an element outside {ELEMENTS}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lm_vocab.json").write_text(json.dumps({
        "format": "motif_lm_vocab_v1",
        "min_count": args.min_count,
        "elements": ELEMENTS,
        "layout": {"end": END_ID, "bond_base": BOND_BASE, "atom_base": ATOM_BASE, "motif_base": MOTIF_BASE},
        "motifs": [vocab[i] for i in keep],
    }))
    report = {"motifs": len(keep), "files": {}}
    for export_path, fp_path in zip(args.export, args.fp):
        molecules = json.loads(export_path.read_text())["molecules"]
        bits = json.loads(fp_path.read_text())["bits_by_molecule"]
        rows = [json.loads(line) for line in (args.motif_dir / f"{export_path.stem}.motif.jsonl").open()]
        if not (len(molecules) == len(bits) == len(rows)):
            raise SystemExit(f"{export_path}: {len(molecules)} molecules, {len(bits)} fingerprints, {len(rows)} motif rows")
        # The three files are joined by position: a reordered sidecar must be
        # an error, not a silent misattribution.
        sidecar_keys = json.loads(fp_path.read_text()).get("keys_by_molecule")
        if sidecar_keys is None or len(sidecar_keys) != len(molecules):
            raise SystemExit(f"{fp_path}: needs one keys_by_molecule entry per molecule to check the join")
        for i, (molecule, row) in enumerate(zip(molecules, rows)):
            if row["key"] != molecule["key"] or row["index"] != i:
                raise SystemExit(f"{export_path}: motif row {i} is {row['key']}, the export holds {molecule['key']}")
            if sidecar_keys[i] != f"{molecule['key']}|{molecule['identity_group']}":
                raise SystemExit(f"{fp_path}: entry {i} is {sidecar_keys[i]}, the export holds {molecule['key']}")
        usable = 0
        with (args.out / f"{export_path.stem}.lm.jsonl").open("w") as stream:
            for molecule, on, row in zip(molecules, bits, rows):
                tokens = None
                if "tokens" in row and all(t[0] != "M" or t[1] in remap for t in row["tokens"]):
                    tokens = token_ids([[t[0], remap[t[1]]] if t[0] == "M" else t for t in row["tokens"]])
                    usable += 1
                stream.write(json.dumps({
                    "index": row["index"],
                    "key": molecule["key"],
                    "tokens": tokens,
                    "formula": formula_counts(molecule["smiles"]),
                    "bits": on,
                    "spectra": [s["spectrum_id"] for s in molecule.get("spectra", [])],
                }, separators=(",", ":")) + "\n")
        report["files"][export_path.name] = {"molecules": len(molecules), "with_tokens": usable}
    (args.out / "lm_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "prepare"])
    ap.add_argument("--motif-dir", type=Path, default=None, help="prepare: the directory `build` wrote")
    ap.add_argument("--fp", type=Path, action="append", default=[], help="prepare: one fingerprint sidecar per --export")
    ap.add_argument("--export", type=Path, action="append", required=True, help="files whose motifs form the vocabulary")
    ap.add_argument("--apply", type=Path, action="append", default=[], help="files converted with that vocabulary")
    ap.add_argument("--min-count", type=int, default=1, help="a motif enters the vocabulary at this many molecules")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        if args.motif_dir is None or len(args.fp) != len(args.export):
            raise SystemExit("prepare needs --motif-dir and one --fp per --export")
        return prepare(args)
    args.out.mkdir(parents=True, exist_ok=True)

    converted = {path: _convert_file(path, args.workers) for path in args.export + args.apply}
    counts: collections.Counter = collections.Counter()
    for path in args.export:
        for row in converted[path]:
            for name in {t[1] for t in row.get("tokens", []) if t[0] == "M"}:
                counts[name] += 1
    names = [name for name, count in counts.most_common() if count >= args.min_count]
    vocab = []
    for name in names:
        ref, free, elements = reference(name)
        kekule = Chem.Mol(ref)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        vocab.append({
            "smiles": name,
            "count": counts[name],
            "elements": elements,
            "free": free,
            "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), ORDER[b.GetBondType()]] for b in kekule.GetBonds()],
            "ring": ref.GetRingInfo().NumRings() > 0,
        })
    index = {name: i for i, name in enumerate(names)}
    (args.out / "vocab.json").write_text(json.dumps({"format": "motif_vocab_v1", "motifs": vocab}))

    report = {"vocabulary": len(vocab), "ring_motifs": sum(v["ring"] for v in vocab), "files": {}}
    for path, rows in converted.items():
        skips: collections.Counter = collections.Counter()
        lengths, motifs, kept = [], [], 0
        with (args.out / f"{path.stem}.motif.jsonl").open("w") as stream:
            for row in rows:
                out = {"index": row["index"], "key": row["key"]}
                if "tokens" not in row:
                    out["skip"] = row["skip"]
                elif any(t[0] == "M" and t[1] not in index for t in row["tokens"]):
                    out["skip"] = "motif outside the vocabulary"
                else:
                    out["tokens"] = [[t[0], index[t[1]]] if t[0] == "M" else t for t in row["tokens"]]
                    lengths.append(len(out["tokens"]))
                    motifs.append(sum(1 for t in out["tokens"] if t[0] == "M"))
                    kept += 1
                if "skip" in out:
                    skips[out["skip"]] += 1
                stream.write(json.dumps(out, separators=(",", ":")) + "\n")
        ordered = sorted(lengths)
        report["files"][path.name] = {
            "molecules": len(rows),
            "kept": kept,
            "skipped": dict(skips),
            "tokens_mean": sum(lengths) / max(kept, 1),
            "tokens_p99": ordered[int(0.99 * (len(ordered) - 1))] if ordered else 0,
            "tokens_max": ordered[-1] if ordered else 0,
            "motifs_mean": sum(motifs) / max(kept, 1),
            "in_vocabulary_source": path in args.export,
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
