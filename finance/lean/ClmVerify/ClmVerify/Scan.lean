/-!
# C7: the selective scan as an affine recurrence

Element-wise (one `b, d, n` lane) the Mamba-1 scan is `h_t = a_t * h_{t-1} + u_t`
from `h_{-1} = h0`.  Everything here is stated over an arbitrary commutative
semiring (`Lean.Grind.CommSemiring`, the core class; `Int`, `Rat`, `Nat` are
instances).  The executable `clmcheck` runs these definitions at `Int`.

Indexing: a list `ps = [p_0, …, p_{n-1}]` of pairs `p_t = (a_t, u_t)`;
`seqScan h0 ps = [h_0, …, h_{n-1}]`.

This file holds only definitions (the executable imports it); the theorems are
in `ScanProofs.lean`.
-/
namespace ClmVerify
open Lean.Grind

/-- An affine map `h ↦ a * h + u`. -/
structure Aff (α : Type) where
  a : α
  u : α
deriving Repr

section
variable {α : Type} [CommSemiring α]

/-- `p.comb q`: apply `p`, then `q`.  `(a1,u1) ⊕ (a2,u2) = (a2*a1, a2*u1 + u2)`. -/
def Aff.comb (p q : Aff α) : Aff α := ⟨q.a * p.a, q.a * p.u + q.u⟩

/-- The identity map `(1, 0)`. -/
def Aff.one : Aff α := ⟨1, 0⟩

def Aff.apply (p : Aff α) (h : α) : α := p.a * h + p.u

/-- Sequential scan: `[h_0, …, h_{n-1}]` with `h_t = a_t * h_{t-1} + u_t`, `h_{-1} = h`. -/
def seqScan (h : α) : List (Aff α) → List α
  | [] => []
  | p :: ps => p.apply h :: seqScan (p.apply h) ps

/-- The state after the whole list (the last `h`, or `h` itself for `[]`). -/
def final (h : α) (ps : List (Aff α)) : α := ps.foldl (fun h p => p.apply h) h

/-- Product of the `a`s of a chunk, accumulated left to right as the kernel does. -/
def prodA : List (Aff α) → α
  | [] => 1
  | p :: ps => prodA ps * p.a

/-! ### Chunked forward scan (the GPU kernel's three phases) -/

/-- `chunksAux L fuel xs`: consecutive chunks of length `L` (the last may be shorter). -/
def chunksAux {β : Type} (L : Nat) : Nat → List β → List (List β)
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, x :: xs => (x :: xs).take L :: chunksAux L n ((x :: xs).drop L)

/-- Split time into chunks of `L` steps (`L ≥ 1`; the last chunk may be shorter). -/
def chunksOf {β : Type} (L : Nat) (xs : List β) : List (List β) := chunksAux L xs.length xs

/-- Phase 1, per chunk: the product of `a` and the final state of a local scan from `0`. -/
def summarize (c : List (Aff α)) : Aff α := ⟨prodA c, (seqScan 0 c).getLastD 0⟩

/-- Phase 2: one sequential pass over the chunk summaries; element `k` is the state
carried into chunk `k` (`h0` for the first chunk). -/
def carries (h : α) : List (Aff α) → List α
  | [] => []
  | s :: ss => h :: carries (s.apply h) ss

/-- The chunked scan: phase 1 (`summarize`), phase 2 (`carries`), phase 3 (each chunk
re-scanned locally from its carried-in state), outputs concatenated in time order. -/
def chunkedScan (L : Nat) (h0 : α) (ps : List (Aff α)) : List α :=
  let cs := chunksOf L ps
  let hin := carries h0 (cs.map summarize)
  (List.zipWith (fun c h => seqScan h c) cs hin).flatten

/-! ### Readout `y_t = c_t * h_t + d * x_t` -/

/-- One time step of one lane: decay `a`, input `u` (`= dt*B*x`), readout `c`, skip input `x`. -/
structure Inp (α : Type) where
  a : α
  u : α
  c : α
  x : α
deriving Repr

def Inp.aff (i : Inp α) : Aff α := ⟨i.a, i.u⟩

/-- `y_t = c_t * h_t + d * x_t` with `h = seqScan h0`. -/
def ys (d h0 : α) (ins : List (Inp α)) : List α :=
  List.zipWith (fun i h => i.c * h + d * i.x) ins (seqScan h0 (ins.map Inp.aff))

/-! ### Reverse scan and the adjoint -/

/-- Reverse scan: `g_t = a_t * g_{t+1} + u_t` from `g_n = g`; returns `[g_0, …, g_{n-1}]`. -/
def revScan (g : α) : List (Aff α) → List α
  | [] => []
  | p :: ps => p.apply ((revScan g ps).headD g) :: revScan g ps

/-- The state at the left end of a reverse scan (`g_0`, or `g` for `[]`). -/
def revFinal (g : α) (ps : List (Aff α)) : α := ps.foldr (fun p acc => p.apply acc) g

/-- Phase 1 of the chunked reverse scan: product of `a` and `g` at the chunk's first
step of a local reverse scan from `0`. -/
def revSummarize (c : List (Aff α)) : Aff α := ⟨prodA c, (revScan 0 c).headD 0⟩

/-- Phase 2, right to left over the chunk summaries.  `.2` lists, per chunk, the state
carried in from its right (`g` for the last chunk); `.1` is the state left of all chunks. -/
def revCarries (g : α) : List (Aff α) → α × List α
  | [] => (g, [])
  | s :: ss => ((s.apply (revCarries g ss).1), (revCarries g ss).1 :: (revCarries g ss).2)

/-- The chunked reverse scan (phase 3: each chunk reverse-scanned from its carried-in state). -/
def chunkedRevScan (L : Nat) (g : α) (ps : List (Aff α)) : List α :=
  let cs := chunksOf L ps
  let gin := (revCarries g (cs.map revSummarize)).2
  (List.zipWith (fun c g => revScan g c) cs gin).flatten

/-- One time step for the adjoint: forward `a, u`, readout `c`, upstream gradient `gy = dL/dy`. -/
structure Step (α : Type) where
  a : α
  u : α
  c : α
  gy : α
deriving Repr

def Step.aff (x : Step α) : Aff α := ⟨x.a, x.u⟩

/-- `a_{t+1}` seen from step `t`: the `a` of the next step, `0` past the end (`g_T = 0`). -/
def nextA : List (Step α) → α
  | [] => 0
  | x :: _ => x.a

/-- The adjoint recurrence as affine pairs: `g_t = a_{t+1} * g_{t+1} + c_t * gy_t`. -/
def adjPairs : List (Step α) → List (Aff α)
  | [] => []
  | x :: rest => ⟨nextA rest, x.c * x.gy⟩ :: adjPairs rest

/-- `[g_0, …, g_{n-1}]`, `g_t = c_t * gy_t + a_{t+1} * g_{t+1}`, `g_n = 0`. -/
def adjoint (xs : List (Step α)) : List α := revScan 0 (adjPairs xs)

/-- The forward states `h_t` of the steps. -/
def stepStates (h0 : α) (xs : List (Step α)) : List α := seqScan h0 (xs.map Step.aff)

/-- Sum of a list. -/
def sumL : List α → α
  | [] => 0
  | x :: xs => x + sumL xs

end
end ClmVerify
