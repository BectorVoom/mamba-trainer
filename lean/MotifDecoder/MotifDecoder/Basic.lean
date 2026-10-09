/-
The motif stack machine: vocabulary, tokens, state, the mask `allowed` and the
transition `apply`.  Everything the executable `motifcheck` runs is defined in
this file; every theorem of the project is about these definitions.
-/
namespace MotifDecoder

/-- A bond `(i, j, order)`. -/
abbrev Bond := Nat × Nat × Nat

/-- A prefabricated piece.  Its number of atoms is `elements.length`. -/
structure Motif where
  /-- atomic numbers, one per atom -/
  elements : List Nat
  /-- bond order each atom can still spend on attachments -/
  free : List Nat
  /-- bonds between the motif's own atoms (local numbering) -/
  bonds : List Bond
  deriving Repr

/-- Number of atoms of a motif (the `n` of the specification). -/
def Motif.size (m : Motif) : Nat := m.elements.length

def Motif.empty : Motif := ⟨[], [], []⟩

abbrev Vocab := Array Motif

/-- Motif `m` of the vocabulary (the empty motif when `m` is out of range; the
mask never lets such an `m` through). -/
def motifAt (V : Vocab) (m : Nat) : Motif := V.getD m Motif.empty

/-- A bond of an `n`-atom graph is in range, is not a loop and has order 1..3. -/
def BondOK (n : Nat) (b : Bond) : Prop :=
  b.1 < n ∧ b.2.1 < n ∧ b.1 ≠ b.2.1 ∧ 1 ≤ b.2.2 ∧ b.2.2 ≤ 3

instance (n : Nat) (b : Bond) : Decidable (BondOK n b) := by
  unfold BondOK; infer_instance

/-- The constraints the specification lists under "Vocabulary". -/
structure Motif.WF (m : Motif) : Prop where
  pos : 0 < m.size
  free_len : m.free.length = m.size
  bonds_ok : ∀ b ∈ m.bonds, BondOK m.size b

instance (m : Motif) : Decidable m.WF :=
  decidable_of_iff (0 < m.size ∧ m.free.length = m.size ∧ ∀ b ∈ m.bonds, BondOK m.size b)
    ⟨fun ⟨a, b, c⟩ => ⟨a, b, c⟩, fun ⟨a, b, c⟩ => ⟨a, b, c⟩⟩

/-- Every motif of the vocabulary is well-formed. -/
def VocabWF (V : Vocab) : Prop := ∀ m, m < V.size → (motifAt V m).WF

instance (V : Vocab) : Decidable (VocabWF V) := by
  unfold VocabWF; infer_instance

inductive Token where
  | motif (m : Nat)
  | atom (a : Nat)
  | bond (o : Nat)
  | end_
  deriving DecidableEq, Repr

inductive Phase where
  | start
  | body
  | afterAtom (a : Nat)
  | afterBond (a o : Nat)
  | afterMotif (a o m : Nat)
  | done
  deriving DecidableEq, Repr

/-- Machine state.  `added` and `attachOrder` are ghost counters: the mask never
reads them (see `allowed`), they only record what was done. -/
structure State where
  /-- atomic number of every atom added so far -/
  elements : List Nat
  /-- remaining bond order of every atom added so far -/
  free : List Nat
  /-- all bonds, global numbering, in the order they were added -/
  bonds : List Bond
  /-- frames `(start, size)` of the open motifs, top first -/
  stack : List (Nat × Nat)
  phase : Phase
  /-- ids of the motifs added so far, in order (ghost) -/
  added : List Nat
  /-- sum of the orders of the attachment bonds added so far (ghost) -/
  attachOrder : Nat
  deriving Repr

def State.init : State :=
  { elements := [], free := [], bonds := [], stack := [], phase := .start, added := [],
    attachOrder := 0 }

/-- Current atom count. -/
def State.natoms (s : State) : Nat := s.elements.length

/-- Top frame (`(0, 0)` on an empty stack, which admits no atom). -/
def State.top (s : State) : Nat × Nat := s.stack.headD (0, 0)

def State.setPhase (s : State) (p : Phase) : State := { s with phase := p }

/-- `l[i]`, 0 when out of range (`nth_eq_getD`). -/
def nth : List Nat → Nat → Nat
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => nth xs i

/-- Subtract `o` from `l[i]` (truncating; `nth_attach_src`/`nth_attach_dst`
show the machine never truncates). -/
def subAt : List Nat → Nat → Nat → List Nat
  | [], _, _ => []
  | x :: xs, 0, o => (x - o) :: xs
  | x :: xs, i + 1, o => x :: subAt xs i o

def shiftBond (n : Nat) (b : Bond) : Bond := (n + b.1, n + b.2.1, b.2.2)

/-! ### Budget layer -/

/-- Target formula: a finite map atomic number ↦ count (first entry for a key
wins, absent keys count 0) and a hydrogen count. -/
structure Budget where
  target : List (Nat × Nat)
  hydrogens : Nat
  deriving Repr

def targetCount : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | (k, c) :: rest, e => if k = e then c else targetCount rest e

def Budget.count (b : Budget) (e : Nat) : Nat := targetCount b.target e

/-- Adding motif `mt` to atoms `els` keeps every element within the target.
Only the motif's own elements are tested; `fits_some_iff` shows this is the
"for every element" condition on reachable states. -/
def fits (B : Option Budget) (els : List Nat) (mt : Motif) : Bool :=
  match B with
  | none => true
  | some b => mt.elements.all (fun e => decide (mt.elements.count e + els.count e ≤ b.count e))

/-- The element counts equal the target and the free valences sum to `H`
(`finished_some_iff`). -/
def finished (B : Option Budget) (s : State) : Bool :=
  match B with
  | none => true
  | some b =>
    (s.elements ++ b.target.map Prod.fst).all (fun e => s.elements.count e == b.count e)
      && s.free.sum == b.hydrogens

/-! ### Mask and transition -/

/-- Some atom `b < n` of the motif has `free[b] ≥ o`. -/
def hasEntry (mt : Motif) (o : Nat) : Bool :=
  (List.range mt.size).any (fun b => decide (o ≤ nth mt.free b))

/-- The mask. -/
def allowed (V : Vocab) (B : Option Budget) (s : State) (t : Token) : Bool :=
  match s.phase, t with
  | .start, .motif m => decide (m < V.size) && fits B s.elements (motifAt V m)
  | .body, .end_ => !s.stack.tail.isEmpty || finished B s
  | .body, .atom a => decide (a < s.top.2) && decide (1 ≤ nth s.free (s.top.1 + a))
  | .afterAtom a, .bond o =>
      decide (1 ≤ o) && decide (o ≤ 3) && decide (o ≤ nth s.free (s.top.1 + a))
  | .afterBond _ o, .motif m =>
      decide (m < V.size) && hasEntry (motifAt V m) o && fits B s.elements (motifAt V m)
  | .afterMotif _ o m, .atom b =>
      decide (b < (motifAt V m).size) && decide (o ≤ nth (motifAt V m).free b)
  | _, _ => false

/-- First motif: append it and push its frame. -/
def startMotif (V : Vocab) (s : State) (m : Nat) : State :=
  let mt := motifAt V m
  { s with
    elements := s.elements ++ mt.elements
    free := s.free ++ mt.free
    bonds := s.bonds ++ mt.bonds.map (shiftBond s.natoms)
    stack := (s.natoms, mt.size) :: s.stack
    phase := .body
    added := s.added ++ [m] }

/-- `end_`: pop the top frame. -/
def popFrame (s : State) : State :=
  { s with stack := s.stack.tail, phase := if s.stack.tail.isEmpty then .done else .body }

/-- Attach motif `m` by a bond of order `o` from atom `a` of the top frame to
its atom `b`. -/
def attach (V : Vocab) (s : State) (a o m b : Nat) : State :=
  let mt := motifAt V m
  let n := s.natoms
  let src := s.top.1 + a
  let dst := n + b
  { elements := s.elements ++ mt.elements
    free := subAt (subAt (s.free ++ mt.free) src o) dst o
    bonds := s.bonds ++ mt.bonds.map (shiftBond n) ++ [(src, dst, o)]
    stack := (n, mt.size) :: s.stack
    phase := .body
    added := s.added ++ [m]
    attachOrder := s.attachOrder + o }

/-- The transition (meaningful when `allowed`). -/
def apply (V : Vocab) (s : State) (t : Token) : State :=
  match s.phase, t with
  | .start, .motif m => startMotif V s m
  | .body, .end_ => popFrame s
  | .body, .atom a => s.setPhase (.afterAtom a)
  | .afterAtom a, .bond o => s.setPhase (.afterBond a o)
  | .afterBond a o, .motif m => s.setPhase (.afterMotif a o m)
  | .afterMotif a o m, .atom b => attach V s a o m b
  | _, _ => s

/-- One masked step: `none` when the token is rejected. -/
def step (V : Vocab) (B : Option Budget) (s : State) (t : Token) : Option State :=
  if allowed V B s t then some (apply V s t) else none

/-- Feed a token list; `none` as soon as a token is rejected. -/
def run (V : Vocab) (B : Option Budget) : State → List Token → Option State
  | s, [] => some s
  | s, t :: ts =>
    match step V B s t with
    | some s' => run V B s' ts
    | none => none

/-- `ts` is accepted and leaves the machine in `s`. -/
def AcceptedAs (V : Vocab) (B : Option Budget) (ts : List Token) (s : State) : Prop :=
  run V B State.init ts = some s ∧ s.phase = .done

def Accepted (V : Vocab) (B : Option Budget) (ts : List Token) : Prop :=
  ∃ s, AcceptedAs V B ts s

/-- What `motifcheck` prints: the final state, or the index of the first
rejected token (the sequence length when it ends before `done`). -/
def checkFrom (V : Vocab) (B : Option Budget) : State → Nat → List Token → Except Nat State
  | s, i, [] => if s.phase = .done then .ok s else .error i
  | s, i, t :: ts =>
    if allowed V B s t then checkFrom V B (apply V s t) (i + 1) ts else .error i

def check (V : Vocab) (B : Option Budget) (ts : List Token) : Except Nat State :=
  checkFrom V B State.init 0 ts

end MotifDecoder
