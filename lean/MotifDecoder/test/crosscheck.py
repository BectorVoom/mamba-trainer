"""Differential test of `motifcheck` against an independent Python port.

    python3 test/crosscheck.py --vocab <vocab.json> --seqs <x.motif.jsonl> [--seqs ...] \
        --work <scratch dir> [--bin .lake/build/bin/motifcheck] [--limit N] [--mutants K]

Reads a `motif_vocab_v1` vocabulary and `*.motif.jsonl` sequence files (as
written by tools/ms2/motif_tokens.py), writes them in motifcheck's text
format, adds K seeded random corruptions of every sequence, and compares
motifcheck's output line by line with `run` below, which is written from the
prose specification of the token-level machine and shares no code with the
Lean model.  It then repeats the run with a budget per sequence (the true
formula of the uncorrupted molecule, or that formula with H off by one).
Standard library only, unless `--machine tools/ms2/motif_tokens.py` is given:
then the same sequences also go through that file's `MotifMachine` (needs
RDKit) and its accept/reject verdict, atoms, free valences and bond endpoints
are compared with motifcheck's.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path


def run(vocab, tokens, budget=None):
    """Output line of the specification's machine for one token list."""
    elements, free, bonds, stack = [], [], [], []
    phase = ("start",)
    count = collections.Counter()

    def fits(m):
        if budget is None:
            return True
        need = collections.Counter(vocab[m]["elements"])
        return all(need[e] + count[e] <= budget[1].get(e, 0) for e in set(need) | set(count) | set(budget[1]))

    def add(m):
        base = len(elements)
        motif = vocab[m]
        elements.extend(motif["elements"])
        free.extend(motif["free"])
        count.update(motif["elements"])
        bonds.extend((base + i, base + j, o) for i, j, o in motif["bonds"])
        return base

    for index, (kind, value) in enumerate(tokens):
        ok = False
        if phase[0] == "start":
            if kind == "M" and value < len(vocab) and fits(value):
                base = add(value)
                stack.append((base, len(vocab[value]["elements"])))
                phase = ("body",)
                ok = True
        elif phase[0] == "body":
            if kind == "E":
                last = len(stack) == 1
                if budget is None or not last or (
                    all(count[e] == budget[1].get(e, 0) for e in set(count) | set(budget[1]))
                    and sum(free) == budget[0]
                ):
                    stack.pop()
                    phase = ("done",) if not stack else ("body",)
                    ok = True
            elif kind == "A":
                start, size = stack[-1]
                if value < size and free[start + value] >= 1:
                    phase = ("afterAtom", value)
                    ok = True
        elif phase[0] == "afterAtom":
            start, _ = stack[-1]
            if kind == "B" and 1 <= value <= 3 and free[start + phase[1]] >= value:
                phase = ("afterBond", phase[1], value)
                ok = True
        elif phase[0] == "afterBond":
            if kind == "M" and value < len(vocab) and any(f >= phase[2] for f in vocab[value]["free"]) and fits(value):
                phase = ("afterMotif", phase[1], phase[2], value)
                ok = True
        elif phase[0] == "afterMotif":
            _, a, o, m = phase
            motif = vocab[m]
            if kind == "A" and value < len(motif["elements"]) and motif["free"][value] >= o:
                source = stack[-1][0] + a
                base = add(m)
                bonds.append((source, base + value, o))
                free[source] -= o
                free[base + value] -= o
                assert free[source] >= 0 and free[base + value] >= 0
                stack.append((base, len(motif["elements"])))
                phase = ("body",)
                ok = True
        if not ok:
            return f"ERR {index}"
    if phase[0] != "done":
        return f"ERR {len(tokens)}"
    join = lambda xs: " ".join(str(x) for x in xs)
    flat = [x for bond in bonds for x in bond]
    return f"OK {len(elements)} {len(bonds)} | {join(elements)} | {join(free)} | {join(flat)}"


def machine_line(module, vocab, tokens):
    """`MotifMachine` on one token list: None when rejected, else
    (elements, free, bond endpoints, bond orders or None where aromatic)."""
    named = []
    for kind, value in tokens:
        if kind == "M":
            if value >= len(vocab):
                return None
            named.append(["M", vocab[value]["smiles"]])
        elif kind == "E":
            named.append(["E"])
        else:
            named.append([kind, value])
    machine = module.MotifMachine()
    if machine.run(named) is not None:
        return None
    order = {1.0: 1, 2.0: 2, 3.0: 3}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order.get(b.GetBondTypeAsDouble())) for b in machine.rw.GetBonds()]
    return [a.GetAtomicNum() for a in machine.rw.GetAtoms()], list(machine.free), bonds


def same_as_machine(line, result):
    if result is None:
        return line.startswith("ERR")
    if not line.startswith("OK"):
        return False
    _, els, free, flat = line.split(" | ")
    flat = [int(x) for x in flat.split()]
    lean_bonds = [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]
    elements, machine_free, bonds = result
    if [int(x) for x in els.split()] != elements or [int(x) for x in free.split()] != machine_free:
        return False
    if len(lean_bonds) != len(bonds):
        return False
    return all(x[:2] == y[:2] and (y[2] is None or x[2] == y[2]) for x, y in zip(lean_bonds, bonds))


def mutate(rng, tokens, vocab_size):
    out = [list(t) for t in tokens]
    choice = rng.randrange(7)
    i = rng.randrange(len(out))
    if choice == 0:
        del out[i]
    elif choice == 1:
        out = out[: rng.randrange(len(out) + 1)]
    elif choice == 2:
        out.insert(i, rng.choice([["E", 0], ["A", rng.randrange(8)], ["B", rng.randrange(5)], ["M", rng.randrange(vocab_size + 2)]]))
    elif choice == 3:
        j = rng.randrange(len(out))
        out[i], out[j] = out[j], out[i]
    elif choice == 4:
        kind = out[i][0]
        if kind == "A":
            out[i][1] = rng.randrange(40)
        elif kind == "B":
            out[i][1] = rng.randrange(5)
        elif kind == "M":
            out[i][1] = rng.randrange(min(vocab_size, 400))
        else:
            out[i] = ["A", rng.randrange(6)]
    elif choice == 5:
        out.append(rng.choice([["E", 0], ["A", 0], ["M", 0]]))
    else:
        atoms = [k for k, t in enumerate(out) if t[0] == "A"]
        if atoms:
            k = rng.choice(atoms)
            out[k][1] = max(0, out[k][1] + rng.choice([-1, 1]))
    return out


def text(tokens):
    return " ".join("E" if k == "E" else f"{k}{v}" for k, v in tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--seqs", type=Path, action="append", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--bin", type=Path, default=Path(__file__).resolve().parent.parent / ".lake/build/bin/motifcheck")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mutants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", type=Path, help="tools/ms2/motif_tokens.py, to compare its MotifMachine too")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    vocab = json.loads(args.vocab.read_text())["motifs"]
    lines = [str(len(vocab))]
    for motif in vocab:
        row = [len(motif["elements"]), *motif["elements"], *motif["free"], len(motif["bonds"])]
        for bond in motif["bonds"]:
            row += bond
        lines.append(" ".join(str(x) for x in row))
    (args.work / "vocab.txt").write_text("\n".join(lines) + "\n")

    sequences, budgets = [], []
    real = 0
    for path in args.seqs:
        with path.open() as stream:
            for raw in stream:
                row = json.loads(raw)
                if "tokens" not in row:
                    continue
                tokens = [[t[0], t[1] if len(t) > 1 else 0] for t in row["tokens"]]
                real += 1
                reference = run(vocab, tokens)
                formula = None
                if reference.startswith("OK"):
                    _, els, free, _ = reference.split(" | ")
                    formula = (sum(int(x) for x in free.split()), dict(collections.Counter(int(x) for x in els.split())))
                variants = [tokens] + [mutate(rng, tokens, len(vocab)) for _ in range(args.mutants)]
                for k, variant in enumerate(variants):
                    sequences.append(variant)
                    if formula is None:
                        budgets.append(None)
                    elif k == 1:
                        budgets.append((formula[0] + 1, formula[1]))
                    else:
                        budgets.append(formula)
                if args.limit and real >= args.limit:
                    break
        if args.limit and real >= args.limit:
            break
    (args.work / "sequences.txt").write_text("".join(text(s) + "\n" for s in sequences))
    (args.work / "budgets.txt").write_text(
        "".join(("-" if b is None else " ".join([str(b[0])] + [f"{e} {c}" for e, c in sorted(b[1].items())])) + "\n" for b in budgets)
    )

    failures = 0
    for label, extra, use in (("no budget", [], False), ("budget", [str(args.work / "budgets.txt")], True)):
        got = subprocess.run(
            [str(args.bin), str(args.work / "vocab.txt"), str(args.work / "sequences.txt"), *extra],
            check=True, capture_output=True, text=True,
        )
        if got.stderr:
            sys.stderr.write(got.stderr)
        lean = got.stdout.split("\n")
        if lean and lean[-1] == "":
            lean.pop()
        want = [run(vocab, s, b if use else None) for s, b in zip(sequences, budgets)]
        wrong = [i for i, (x, y) in enumerate(zip(lean, want)) if x != y]
        if len(lean) != len(want):
            print(f"{label}: {len(lean)} lines from motifcheck, {len(want)} expected")
            failures += 1
        for i in wrong[:5]:
            print(f"{label}: line {i}: {text(sequences[i])}\n  lean   {lean[i]}\n  python {want[i]}")
        failures += len(wrong)
        ok = sum(1 for x in want if x.startswith("OK"))
        print(f"{label}: {len(want)} sequences ({real} real), {ok} OK, {len(want) - ok} ERR, {len(wrong)} mismatches")
        if args.machine and not use:
            import importlib.util

            sys.dont_write_bytecode = True  # leave no __pycache__ next to the tool
            spec = importlib.util.spec_from_file_location("motif_tokens", args.machine)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results = [machine_line(module, vocab, s) for s in sequences]
            differ = [i for i, r in enumerate(results) if not same_as_machine(lean[i], r)]
            for i in differ[:5]:
                print(f"MotifMachine: line {i}: {text(sequences[i])}\n  lean {lean[i]}")
            failures += len(differ)
            accepted = sum(r is not None for r in results)
            print(f"MotifMachine: {len(sequences)} sequences, {accepted} accepted, {len(differ)} disagreements with motifcheck")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
