# MotifDecoder: Lean 4 model and proofs of the motif stack machine

The motif decoder (`src/models/ms2/motif.rs`, `examples/ms2_motif_decoder.rs`)
writes a molecule as a sequence of whole ring systems, groups and single
atoms. A stack machine reads the tokens, assembles the graph and decides which
token may come next. This project states that machine in Lean 4 and proves
what it guarantees.

**What is and is not verified.** The proofs are about the Lean definitions in
`MotifDecoder/Basic.lean`. The Rust and Python implementations are not
modelled in Lean; they are *tested* against the Lean executable (same
sequences; what is compared for each is listed under "Tests against the
implementations"). The neural network is not involved: the
theorems hold for every token sequence the mask lets through, whatever
produced it.

All seven properties below are proven. The project builds with zero errors,
zero warnings, no `sorry`, no `axiom`, no `native_decide`.

- **Lean**: `Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`, pinned in `lean-toolchain`.
- **Dependencies**: none (core Lean only; neither Batteries nor Mathlib).
- **Size**: about 2,000 lines; a clean build takes about 6 s.

## Layout

| File | Content |
|---|---|
| `MotifDecoder/Basic.lean` | Vocabulary, tokens, state, budget, `allowed`, `apply`, `step`, `run`, `check`. Everything the executable runs. |
| `MotifDecoder/ListLemmas.lean` | Facts about `nth`, `subAt`, `shiftBond` and the two budget tests. |
| `MotifDecoder/Step.lean` | Relational machine `Step`, reachability `ReachBy`, equivalence with `allowed`/`apply`/`run`, correctness of `check`. |
| `MotifDecoder/Invariant.lean` | Item 1. |
| `MotifDecoder/Accounting.lean` | Items 2, 3, 4. |
| `MotifDecoder/Mask.lean` | Item 5, plus ghost-counter faithfulness and progress. |
| `MotifDecoder/Connectivity.lean` | Item 6. |
| `MotifDecoder/Grammar.lean` | Item 7. |
| `MotifDecoder/Bridge.lean` | Extra: every accepted sequence is a sentence of the grammar. |
| `MotifDecoder/Axioms.lean` | `#print axioms` for every main theorem (built with the library). |
| `Main.lean` | `motifcheck`; imports only `MotifDecoder/Basic.lean`. |
| `test/` | Tiny vocabulary, sequences, budgets and the executable's output on them (`expected*.txt`, which `tests/ms2_motif.rs` holds the Rust machine to); `crosscheck.py` (differential test). |

## The machine

A motif has atoms with elements, a `free` value per atom (how much bond order
the atom can still spend on attachments) and its own bonds. Tokens are
`motif m`, `atom a`, `bond o`, `end_`. The state holds the atoms and bonds
added so far, a stack of open motifs and a phase:

- `start`: `motif m` adds the root and pushes it.
- `body`: `end_` pops (the sequence is done when the stack empties);
  `atom a` needs `a` inside the top motif with `free ≥ 1`.
- `afterAtom a`: `bond o` needs `1 ≤ o ≤ 3` and `free ≥ o` at that atom.
- `afterBond a o`: `motif m` needs some atom of `m` with `free ≥ o`.
- `afterMotif a o m`: `atom b` needs `free[b] ≥ o` in `m`; the motif is added,
  the bond `(top.start + a, new.start + b, o)` is added, `o` is subtracted
  from `free` at both ends, and the motif is pushed.

With a budget (a target formula), a motif must also fit the element counts
still missing, and the `end_` that empties the stack needs every element
count met and `sum free` equal to the target's hydrogens.

## How "reachable" is stated

`ReachBy V B ts s`: reading the tokens `ts` from `State.init`, each one
allowed in turn, leaves the machine in `s`. `V` is the vocabulary, `B` an
optional budget.

```lean
theorem step_iff : Step V B s t s' ↔ allowed V B s t = true ∧ s' = apply V s t
theorem reachBy_iff_run : ReachBy V B ts s ↔ run V B State.init ts = some s
theorem reachable_iff : Reachable V B s ↔ ∃ ts, run V B State.init ts = some s
```

`AcceptedAs V B ts s` is `run V B State.init ts = some s ∧ s.phase = .done`.

Hypotheses on the vocabulary:

- `VocabWF V`: every motif has `0 < size`, `free.length = size`, and every
  bond `i < size`, `j < size`, `i ≠ j`, `1 ≤ o ≤ 3`.
- `VocabConnected V`: in every motif, every atom is connected to atom 0.

## Results

### 1. Well-formedness invariant — proven

```lean
structure State.WF (s : State) : Prop where
  len : s.elements.length = s.free.length
  frames : ∀ f ∈ s.stack, f.1 + f.2 ≤ s.elements.length
  bonds : ∀ b ∈ s.bonds, b.1 < s.elements.length ∧ b.2.1 < s.elements.length ∧ b.1 ≠ b.2.1 ∧
    1 ≤ b.2.2 ∧ b.2.2 ≤ 3

theorem ReachBy.wf (hV : VocabWF V) (h : ReachBy V B ts s) : s.WF
```

### 2. Valence accounting — proven

No atom spends more bond order than it has: the two subtractions of an
attachment are exact, and the total is conserved.

```lean
theorem nth_attach_src (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (hal : allowed V B s (.atom b) = true) :
    nth (apply V s (.atom b)).free (s.top.1 + a) + o = nth s.free (s.top.1 + a)
theorem nth_attach_dst (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (hal : allowed V B s (.atom b) = true) :
    nth (apply V s (.atom b)).free (s.natoms + b) + o = nth (motifAt V m).free b
theorem ReachBy.valence (hV : VocabWF V) (h : ReachBy V B ts s) :
    s.free.sum + 2 * s.attachOrder = (s.added.map (fun m => (motifAt V m).free.sum)).sum
theorem ReachBy.valence_total (hV : VocabWF V) (h : ReachBy V B ts s) :
    s.free.sum + 2 * orderSum s.bonds =
      (s.added.map (fun m => (motifAt V m).free.sum + 2 * orderSum (motifAt V m).bonds)).sum
theorem ReachBy.free_le_orig (h : ReachBy V B ts s) :
    s.free.length = (origFree V s).length ∧ ∀ i, nth s.free i ≤ nth (origFree V s) i
```

`s.attachOrder` and `s.added` are ghost fields (the mask never reads them).
`ReachBy.orderSum_bonds`, `AcceptedAs.added_eq` and
`AcceptedAs.attachOrder_eq` tie them to the bonds and to the tokens.
`valence_total` removes `attachOrder`; its right-hand side still ranges over
`s.added`, which for an accepted sequence is the list of its `motif` tokens.

### 3. Counting: atoms and bonds add up — proven

```lean
theorem ReachBy.natoms_eq (h : ReachBy V B ts s) :
    s.elements.length = (s.added.map (fun m => (motifAt V m).size)).sum
theorem ReachBy.nbonds_eq (h : ReachBy V B ts s) (hne : s.phase ≠ .start) :
    s.bonds.length + 1 = (s.added.map (fun m => (motifAt V m).bonds.length)).sum + s.added.length
theorem ReachBy.cyclomatic_int (h : ReachBy V B ts s) (hne : s.phase ≠ .start) :
    (s.bonds.length : Int) + 1 - (s.elements.length : Int) =
      (s.added.map (fun m => ((motifAt V m).bonds.length : Int) + 1 - ((motifAt V m).size : Int))).sum
```

This is an identity between counts: `bonds + 1 − atoms` of the result is the
sum of the same quantity over the motifs. For a connected graph that
quantity is its cycle rank, so together with item 6 the result has no more
independent cycles than its motifs bring. Cycles themselves are **not**
formalised: there is no theorem saying that each individual cycle of the
result lies inside one motif. The natural-number form (`ReachBy.cyclomatic`)
needs `size ≤ bonds + 1` per motif as a hypothesis; that this follows from
connectivity is also not proven (the integer form does not need it).

### 4. Composition and exact formula — proven

```lean
theorem ReachBy.count_eq (h : ReachBy V B ts s) (e : Nat) :
    s.elements.count e = (s.added.map (fun m => (motifAt V m).elements.count e)).sum
theorem AcceptedAs.budget (h : AcceptedAs V (some b) ts s) :
    (∀ e, s.elements.count e = b.count e) ∧ s.free.sum = b.hydrogens
```

An accepted sequence under a budget has exactly the target's element counts
and hydrogens. `ReachBy.fits_iff` and `finished_some_iff` show the mask's two
finite tests are the "for every element" conditions.

### 5. Mask facts — proven

```lean
theorem end_allowed_in_body (hp : s.phase = .body) : allowed V none s .end_ = true
theorem entry_exists (hp : s.phase = .afterBond a o) (h : allowed V B s (.motif m) = true) :
    ∃ b, allowed V B (apply V s (.motif m)) (.atom b) = true
theorem ReachBy.bond_one_allowed (h : ReachBy V B ts s) (hp : s.phase = .afterAtom a) :
    allowed V B s (.bond 1) = true
theorem Accepted.length_eq (h : Accepted V B ts) : ts.length = 5 * (tokenMotifs ts).length - 3
theorem done_nothing_allowed (hp : s.phase = .done) (t : Token) : allowed V B s t = false
theorem ReachBy.progress (hV : VocabWF V) (hne : 0 < V.size) (h : ReachBy V none ts s)
    (hnd : s.phase ≠ .done) : ∃ t, allowed V none s t = true
```

`progress`: for a well-formed, non-empty vocabulary and no budget, the mask
has no dead end in any phase. It says a next token exists, not that every
choice of tokens terminates.

### 6. Connectivity — proven

```lean
def Adj (bs : List Bond) (i j : Nat) : Prop := ∃ o, (i, j, o) ∈ bs ∨ (j, i, o) ∈ bs
inductive Conn (bs : List Bond) : Nat → Nat → Prop
  | refl (i : Nat) : Conn bs i i
  | tail {i j k : Nat} : Conn bs i j → Adj bs j k → Conn bs i k

theorem ReachBy.connected (hC : VocabConnected V) (h : ReachBy V B ts s) :
    ∀ i, i < s.natoms → Conn s.bonds 0 i
theorem ReachBy.connected_pair (hC : VocabConnected V) (h : ReachBy V B ts s)
    (hi : i < s.natoms) (hj : j < s.natoms) : Conn s.bonds i j
```

### 7. Unambiguous grammar — proven

```lean
inductive MotifTree where
  | node (m : Nat) (children : List (Nat × Nat × Nat × MotifTree))   -- child = (a, o, b, subtree)

theorem parse_serialize (t : MotifTree) : parse (serialize t) = some t
theorem serialize_injective (h : serialize t₁ = serialize t₂) : t₁ = t₂
theorem parse_eq_some_iff (ts : List Token) (t : MotifTree) : parse ts = some t ↔ serialize t = ts
theorem Accepted.exists_tree (h : Accepted V B ts) : ∃ t, serialize t = ts
```

Every accepted sequence is the serialization of exactly one motif tree.

### Not proven

- **That the Rust and Python machines are the Lean machine.** They are
  tested against it (below), not modelled. The Rust machine also differs on
  purpose: it uses bounded integers and refuses a motif that would take the
  graph past 256 atoms (`MAX_GRAPH_ATOMS`), keeps cached sums
  (`counts`, `free_sum`, `max_free`) where Lean recomputes from lists, and
  its `run` corresponds to Lean's `check` (it rejects an unfinished
  sequence). Because of the bound, the progress theorem (item 5) does not
  carry over to the Rust machine as stated.
- **The search.** Candidate selection, scores, the cache reorder and the
  pruning rule `MotifMachine::dead` have no Lean counterpart. `dead` flags a
  state only when every heavy atom of the budget is placed (so no motif fits
  any more) and either an attachment is open or the hydrogen count is wrong;
  that argument is in the Rust documentation and a test, not in Lean.
- **Training and search conditioning.** Nothing here concerns the network.
- **Canonical form.** That a molecule gets one sequence whatever its atom
  order is checked by construction and tests, not proven; the grammar result
  (item 7) is about ordered motif trees, and stereoisomers share a sequence.
- **Budgeted completability.** With a budget, a state that passed the mask
  can be impossible to complete (see `test/` line 3 with the target C1 O1 H4:
  every token passes until the last `end_`). The decoder's search drops the
  states it can recognise as dead (`MotifMachine::dead`); that pruning rule is
  not in the Lean model.
- **The converse of `Accepted.exists_tree`**: that the serialization of every
  valence-respecting tree is accepted.
- **`size ≤ bonds + 1` from connectivity** (item 3, natural-number form only).
- **The executable's text parsing and printing** (`Main.lean`); only `check`,
  which it calls, is covered (`check_ok_iff`, `check_error`).
- **Chemistry.** `free` is an input. That a motif's `free` values are the
  right ones, that the canonical sequence of a molecule rebuilds that
  molecule, and that a rebuilt graph is a sensible molecule are checked by
  running the converter on the data (below), not proven.

## `#print axioms`

`lake build` prints, for the 43 main theorems (`MotifDecoder/Axioms.lean`),
only Lean's three standard axioms: `propext`, `Classical.choice`,
`Quot.sound`. There is no `sorryAx` and no `Lean.ofReduceBool`. Verbatim for
the headline theorems:

```
'MotifDecoder.ReachBy.wf' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.ReachBy.valence' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.ReachBy.valence_total' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.ReachBy.free_le_orig' depends on axioms: [propext, Quot.sound]
'MotifDecoder.ReachBy.natoms_eq' depends on axioms: [propext, Quot.sound]
'MotifDecoder.ReachBy.nbonds_eq' depends on axioms: [propext, Quot.sound]
'MotifDecoder.ReachBy.cyclomatic_int' depends on axioms: [propext, Quot.sound]
'MotifDecoder.ReachBy.count_eq' depends on axioms: [propext, Quot.sound]
'MotifDecoder.AcceptedAs.budget' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.Accepted.length_eq' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.ReachBy.progress' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.ReachBy.connected' depends on axioms: [propext, Classical.choice, Quot.sound]
'MotifDecoder.parse_serialize' depends on axioms: [propext, Quot.sound]
'MotifDecoder.parse_eq_some_iff' depends on axioms: [propext, Quot.sound]
'MotifDecoder.Accepted.exists_tree' depends on axioms: [propext, Classical.choice, Quot.sound]
```

## Build and run

```sh
cd lean/MotifDecoder
~/.elan/bin/lake build                      # library, axiom listing, executable
~/.elan/bin/lake exe motifcheck test/vocab.txt test/sequences.txt [test/budgets.txt]
```

`motifcheck <vocab-file> <sequences-file> [<budgets-file>]` runs
`MotifDecoder.check`, a loop over `allowed` and `apply`:

- vocab file: `N`, then per motif `n e_0 … e_{n-1} f_0 … f_{n-1} k i_1 j_1 o_1 …`.
- sequences file: one sequence per line, tokens `M<id>`, `A<index>`, `B<order>`, `E`.
- budgets file (optional): per sequence `H e_1 c_1 …` or `-`.
- output: `OK <atoms> <bonds> | <elements> | <free> | <bonds>` or
  `ERR <index of the first rejected token>` (the length if it ended early).

## Tests against the implementations (tests, not proofs)

The script `test/crosscheck.py` writes the full motif vocabulary (16,460
motifs) and the training and validation sequences (19,334) in `motifcheck`'s
format, adds four seeded random corruptions of each, and compares outputs.
The comparisons are not all of the same strength: the Rust machine and an
independent Python port of the specification are compared on whole output
lines (accepted graph, or index of the first rejected token) with and
without budgets; the RDKit machine of the converter is compared without
budgets only, on acceptance and on the accepted graph's atoms, free valences
and bond endpoints (bond orders where RDKit's bond is not aromatic), not on
rejection indices.

| Compared with the Lean executable | Sequences | Result |
|---|---|---|
| Independent Python port of the specification, no budget | 96,670 (29,757 accepted) | 0 differing lines |
| The same port with a budget per sequence | 96,670 (25,949 accepted) | 0 differing lines |
| `MotifMachine` of `tools/ms2/motif_tokens.py` (RDKit) | 96,670 | 0 disagreements on accept/reject, elements, free valences, bonds |
| Rust `MotifMachine` (`examples/ms2_motif_check.rs`), no budget | 96,670 | 0 differing lines |
| Rust `MotifMachine`, with budgets | 96,670 | 0 differing lines |

What these tests do not exercise: intermediate states and allowed-token sets
(only final graphs and rejection indices are compared), graphs near the Rust
machine's 256-atom bound, budgets taken from an independent source (they are
derived from the rebuilt graph), and aromatic bond orders in the RDKit
comparison.

The real vocabulary satisfies `VocabWF`, and every motif in it is connected,
so the hypotheses of items 1 to 6 hold for it; the Rust loader checks both
and refuses a vocabulary that fails either. Independently of Lean, the
converter rebuilt 180,610 of the 180,614 training-pool molecules from their
sequences exactly (stereo-free canonical SMILES) in the run that produced
the training data; it has since been changed to leave out molecules with a
motif of 2,000 or more automorphisms, which are five of those (none in the
vocabulary the decoder uses). The Rust driver refuses
to train if the machine rejects any training sequence for its own formula
(0 rejected of 169,668).

```sh
D=data/ms2/specgen/motif        # from the repository root
data/ms2/specgen/venv/bin/python -B lean/MotifDecoder/test/crosscheck.py --vocab $D/vocab.json \
    --seqs $D/msgym_validation.motif.jsonl --seqs $D/msgym_train.motif.jsonl --work <dir> \
    --bin lean/MotifDecoder/.lake/build/bin/motifcheck --machine tools/ms2/motif_tokens.py
lean/MotifDecoder/.lake/build/bin/motifcheck <dir>/vocab.txt <dir>/sequences.txt > lean.txt
target/cpu-ms2/release/examples/ms2_motif_check <dir>/vocab.txt <dir>/sequences.txt > rust.txt
diff lean.txt rust.txt
```

## Where the Lean model differs in form from the prose machine

None of these changes behaviour on a reachable state.

1. A motif's atom count is `elements.length`; `free.length = n` and `n ≥ 1`
   are part of `VocabWF`, not of the type.
2. Lookups are total: `nth l i` is 0 out of range, the top of an empty stack
   is `(0, 0)`, motif `m ≥ vocab.size` is the empty motif. The invariants show
   reachable states never read out of range.
3. Subtraction is truncating natural-number subtraction; item 2 proves it
   never truncates.
4. `apply` is total and does not check the mask: it is the identity on a
   token of the wrong kind for the phase, but it would still pop on an `end_`
   that only the budget refuses. `step` and `run` test `allowed` first, so a
   refused token never reaches `apply`; the Rust `apply` tests it itself and
   returns an error without changing the machine.
5. The budget target is a list of `(atomic number, count)` pairs (first entry
   for a key wins, absent keys count 0).
6. `added` and `attachOrder` are ghost fields the mask never reads.
