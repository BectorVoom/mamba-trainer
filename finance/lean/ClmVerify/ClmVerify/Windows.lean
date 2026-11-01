/-!
# C2, C3: feature windows, label windows, label classes (definitions)

Indices are trading-day indices (`Nat`).  For a decision on day `t`:

* the model input is the window of feature rows `t - W + 1 .. t`;
* feature row `r` reads days `r - K .. r` (C2, lookback `K`);
* the label reads `O[t+1]`, `O[t+1+h]` and `L[t+1 .. t+h]` and exists iff every bar of
  `t+1 .. t+1+h` exists (C3).
-/
namespace ClmVerify

/-- Rows of the input window of decision day `t`: `t + 1 - W, …, t` (needs `W ≤ t + 1`). -/
def windowRows (W t : Nat) : List Nat := (List.range W).map (fun i => t + 1 - W + i)

/-- Days read by feature row `r`: `r - K, …, r` (needs `K ≤ r`). -/
def rowReads (K r : Nat) : List Nat := (List.range (K + 1)).map (fun j => r - K + j)

/-- Every day the input window of decision day `t` reads. -/
def featureReads (K W t : Nat) : List Nat := (windowRows W t).flatMap (rowReads K)

def labelStart (t : Nat) : Nat := t + 1
def labelEnd (h t : Nat) : Nat := t + 1 + h

/-- The days whose bars must exist for the label of `t` (C3: `t+1 .. t+1+h`). -/
def labelDays (h t : Nat) : List Nat := (List.range (h + 1)).map (fun j => t + 1 + j)

/-- The prices the label of `t` reads: `O[t+1]`, `O[t+1+h]`, and `L[t+1..t+h]` (by day). -/
def labelPriceReads (h t : Nat) : List Nat :=
  [t + 1, t + 1 + h] ++ (List.range h).map (fun j => t + 1 + j)

/-- Actions / label classes (C3, C5): `BUY = 0`, `SELL = 1`, `HOLD = 2`. -/
inductive Action where
  | buy
  | sell
  | hold
deriving DecidableEq, Repr

def Action.code : Action → Nat
  | .buy => 0
  | .sell => 1
  | .hold => 2

def Action.ofCode? : Nat → Option Action
  | 0 => some .buy
  | 1 => some .sell
  | 2 => some .hold
  | _ => none

section
variable {α : Type} [Lean.Grind.CommRing α]

/-- `theta = 2*kappa + m*sigma_h`. -/
def theta (kappa m sigma : α) : α := 2 * kappa + m * sigma

/-- C3 class rule: `BUY` if `r > θ`, `SELL` if `r < -θ`, `HOLD` otherwise. -/
def classify [LT α] [DecidableLT α] (θ r : α) : Action :=
  if θ < r then .buy else if r < -θ then .sell else .hold

end
end ClmVerify
