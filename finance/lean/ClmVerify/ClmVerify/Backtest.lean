import ClmVerify.Scan
import ClmVerify.Windows
/-!
# C5: policy and backtest accounting (definitions)

Per security, decision days `t, t+1, …` of one segment, actions `BUY / SELL / HOLD`,
position before the first decision `0`.  Positions are `0` or `1` (`Nat`).

`ret` of decision `t` is `O[t+2] / O[t+1] - 1` when both opens exist (`none` when one is
missing); it is data here, the model only says which day indices it comes from.
Money is in an arbitrary commutative ring (`Lean.Grind.CommRing`).
-/
namespace ClmVerify

-- The core ring classes carry `NatCast` as a field; use it for `trade : Nat` in money.
attribute [local instance] Lean.Grind.Semiring.natCast

/-- `p(t) = 1` on BUY, `0` on SELL, `p(t-1)` on HOLD. -/
def nextPos (prev : Nat) : Action → Nat
  | .buy => 1
  | .sell => 0
  | .hold => prev

/-- `trade(t) = |p(t) - p(t-1)|`. -/
def tradeSize (prev p : Nat) : Nat := ((p : Int) - (prev : Int)).natAbs

/-- Positions and trades of a run of actions from position `prev` (no calendar boundary;
this is what `clmcheck backtest` prints). -/
def posTrades (prev : Nat) : List Action → List (Nat × Nat)
  | [] => []
  | a :: as => (nextPos prev a, tradeSize prev (nextPos prev a)) :: posTrades (nextPos prev a) as

/-- The decision of day `t` is executed at the open of day `t + 1`. -/
def execDay (t : Nat) : Nat := t + 1

/-- The prices the P&L of decision `t` reads: `O[t+1]` and `O[t+2]` (by day index). -/
def pnlReads (t : Nat) : List Nat := [t + 1, t + 2]

/-- One day of the per-security ledger. -/
structure Row (α : Type) where
  pos : Nat
  trade : Nat
  gross : α
  cost : α
  net : α

section
variable {α : Type} [Lean.Grind.CommRing α]

/-- Decision `t` (`N` days in the calendar, one-way cost `κ`, previous position `prev`).
If `t + 2 > N - 1` it is not traded: `gross = cost = 0`, position unchanged.  A missing
open (`ret = none`) gives `gross = 0`; the position is still recorded and the cost charged. -/
def runDay (N : Nat) (κ : α) (t prev : Nat) (a : Action) (ret : Option α) : Row α :=
  if t + 2 > N - 1 then ⟨prev, 0, 0, 0, 0⟩
  else
    let p := nextPos prev a
    let tr := tradeSize prev p
    let g : α := match ret with
      | some r => (p : α) * r
      | none => 0
    ⟨p, tr, g, κ * (tr : α), g - κ * (tr : α)⟩

/-- The ledger of consecutive decision days `t, t+1, …` starting from position `prev`. -/
def runBacktest (N : Nat) (κ : α) : Nat → Nat → List (Action × Option α) → List (Row α)
  | _, _, [] => []
  | t, prev, (a, r) :: ds =>
    runDay N κ t prev a r :: runBacktest N κ (t + 1) (runDay N κ t prev a r).pos ds

/-- A test segment's ledger: decision days `ts, ts+1, …`, position `0` before the first. -/
def runSegment (N : Nat) (κ : α) (ts : Nat) (ds : List (Action × Option α)) : List (Row α) :=
  runBacktest N κ ts 0 ds

end
end ClmVerify
