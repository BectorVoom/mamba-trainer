/-!
# C4: walk-forward folds, purging and embargo (definitions)

`years : List Int` is the calendar year of each trading day `t = 0 .. N-1`
(`N = years.length`).  Segments are half-open day ranges `[start, stop)`.
The executable `clmcheck` runs these definitions.
-/
namespace ClmVerify

/-- A half-open range of days `[start, stop)`. -/
structure Seg where
  start : Nat
  stop : Nat
deriving Repr, DecidableEq

def Seg.mem (s : Seg) (t : Nat) : Prop := s.start ≤ t ∧ t < s.stop
def Seg.len (s : Seg) : Nat := s.stop - s.start
/-- The days of the segment in increasing order. -/
def Seg.days (s : Seg) : List Nat := List.range' s.start s.len

inductive Scheme where
  | expanding
  | rolling (trainDays : Nat)
deriving Repr, DecidableEq

structure FoldConfig where
  scheme : Scheme
  firstTest : Int
  lastTest : Int
  valDays : Nat
  calDays : Nat
  /-- label horizon `h` -/
  h : Nat
  /-- embargo `e` -/
  e : Nat
  minTrain : Nat
deriving Repr

structure Fold where
  year : Int
  train : Seg
  val : Seg
  cal : Seg
  test : Seg
deriving Repr, DecidableEq

/-- `year[t]` (`0` past the end; only used for `t < N`). -/
def yr (years : List Int) (t : Nat) : Int := years.getD t 0

/-- The calendar's years are non-decreasing. -/
def Sorted (years : List Int) : Prop :=
  ∀ i j, i ≤ j → j < years.length → yr years i ≤ yr years j

/-- First index whose year satisfies `p`. -/
def firstIdx (p : Int → Bool) : List Int → Option Nat
  | [] => none
  | y :: ys => if p y then some 0 else (firstIdx p ys).map (· + 1)

/-- `ts` = first `t` with `year[t] == Y` (none: year `Y` does not occur). -/
def testStart (years : List Int) (Y : Int) : Option Nat := firstIdx (fun y => decide (y = Y)) years

/-- `te` = first `t` with `year[t] > Y`, or `N`. -/
def testEnd (years : List Int) (Y : Int) : Nat :=
  (firstIdx (fun y => decide (Y < y)) years).getD years.length

/-- The fold of test year `Y`, or `none` when it is skipped (C4). -/
def mkFold (cfg : FoldConfig) (years : List Int) (Y : Int) : Option Fold :=
  match testStart years Y with
  | none => none
  | some ts =>
    let cal : Seg := ⟨ts - cfg.calDays, ts⟩
    let val : Seg := ⟨ts - cfg.calDays - cfg.valDays, ts - cfg.calDays⟩
    let train : Seg := match cfg.scheme with
      | .expanding => ⟨0, val.start⟩
      | .rolling d => ⟨val.start - d, val.start⟩   -- `Nat` subtraction = `max(0, ·)`
    if cfg.calDays + cfg.valDays ≤ ts ∧ cfg.minTrain ≤ train.len then
      some ⟨Y, train, val, cal, ⟨ts, testEnd years Y⟩⟩
    else none

/-- `first_test_year ..= last_test_year`, in order. -/
def testYears (cfg : FoldConfig) : List Int :=
  (List.range (cfg.lastTest - cfg.firstTest + 1).toNat).map (fun (i : Nat) => cfg.firstTest + (i : Int))

/-- The emitted folds, in test-year order. -/
def folds (cfg : FoldConfig) (years : List Int) : List Fold :=
  (testYears cfg).filterMap (mkFold cfg years)

/-! ## Purging and embargo -/

/-- Segment `i` of a fold: `0 = train, 1 = val, 2 = cal, 3 = test`. -/
def Fold.seg (f : Fold) : Nat → Seg
  | 0 => f.train
  | 1 => f.val
  | 2 => f.cal
  | _ => f.test

/-- Keep rule of a segment whose next segment starts at `sNext`: `t + 1 + h + e < sNext`. -/
def keepRule (h e sNext t : Nat) : Bool := decide (t + 1 + h + e < sNext)

/-- Keep rule of the test segment: `label_end(t) = t + 1 + h ≤ N - 1`. -/
def testKeepRule (N h t : Nat) : Bool := decide (t + 1 + h ≤ N - 1)

/-- The keep rule of segment `i` of `f`. -/
def Fold.keeps (N h e : Nat) (f : Fold) (i t : Nat) : Bool :=
  if i < 3 then keepRule h e (f.seg (i + 1)).start t else testKeepRule N h t

/-- The kept decision days of segment `i` (by the rule, day by day). -/
def Fold.kept (N h e : Nat) (f : Fold) (i : Nat) : List Nat :=
  (f.seg i).days.filter (f.keeps N h e i)

def clamp (x a b : Nat) : Nat := min (max x a) b

/-- The end of the kept prefix of segment `i` (closed form):
`clamp(s_next - 1 - h - e, a, b)`, or `clamp(N - 1 - h, a, b)` for the test segment. -/
def Fold.keepEnd (N h e : Nat) (f : Fold) (i : Nat) : Nat :=
  if i < 3 then clamp ((f.seg (i + 1)).start - 1 - h - e) (f.seg i).start (f.seg i).stop
  else clamp (N - 1 - h) (f.seg i).start (f.seg i).stop

end ClmVerify
