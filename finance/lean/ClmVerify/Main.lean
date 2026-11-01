import ClmVerify.Scan
import ClmVerify.Folds
import ClmVerify.Backtest
/-!
`clmcheck`: differential-test oracle for the Rust code.

Reads stdin, one command per line (whitespace-separated integers); for each command
prints its result lines and then a line `end`.  A malformed or unknown command prints
`error <msg>` then `end`.  Blank lines and lines starting with `#` are ignored (no output).

```
folds N scheme train_days first_test last_test val_days cal_days h e min_train_days nyears y1 c1 … yk ck
  → one line per emitted fold, in test-year order:
    fold Y train a b val a b cal a b test a b keep ta tb va vb ca cb xa xb
backtest n a1 … an                 (0 = BUY, 1 = SELL, 2 = HOLD)
  → pos p1 … pn / trade t1 … tn
scan n L h0 a1 u1 … an un          (L ≥ 1)
  → seq h1 … hn / chunk h1 … hn
adjoint n a1 c1 gy1 … an cn gyn
  → g g1 … gn
adjointc n L a1 c1 gy1 … an cn gyn (extension: the chunked reverse kernel, L ≥ 1)
  → g g1 … gn / chunk g1 … gn
```

This file only parses and prints.  It imports only the definition modules
(`Scan`, `Folds`, `Backtest`), so every result comes from the definitions the
theorems are about: `seqScan`, `chunkedScan`, `adjoint`, `chunkedRevScan`, `folds`,
`Fold.keepEnd`, `posTrades`.
-/
open ClmVerify

/-- Whitespace-separated words of a line. -/
def words (s : String) : List String :=
  let flush (cur : List Char) (acc : List String) : List String :=
    if cur.isEmpty then acc else String.ofList cur.reverse :: acc
  let rec go (cs cur : List Char) (acc : List String) : List String :=
    match cs with
    | [] => (flush cur acc).reverse
    | c :: cs => if c.isWhitespace then go cs [] (flush cur acc) else go cs (c :: cur) acc
  go s.toList [] []

def natOfDigits (cs : List Char) : Option Nat :=
  if cs.isEmpty then none
  else cs.foldlM (fun acc c => if c.isDigit then some (acc * 10 + (c.toNat - '0'.toNat)) else none) 0

def parseInt? (w : String) : Option Int :=
  match w.toList with
  | '-' :: ds => (natOfDigits ds).map (fun n => -(n : Int))
  | '+' :: ds => (natOfDigits ds).map (fun n => (n : Int))
  | ds => (natOfDigits ds).map (fun n => (n : Int))

def fmtList {β : Type} [ToString β] (tag : String) (xs : List β) : String :=
  " ".intercalate (tag :: xs.map toString)

/-- A non-negative value, or an error naming the field. -/
def nat (name : String) (x : Int) : Except String Nat :=
  if x < 0 then throw s!"{name} must be >= 0, got {x}" else pure x.toNat

def pairs : List Int → List (Int × Int)
  | a :: b :: rest => (a, b) :: pairs rest
  | _ => []

def triples : List Int → List (Int × Int × Int)
  | a :: b :: c :: rest => (a, b, c) :: triples rest
  | _ => []

/-- Check the count `n` against the remaining numbers (`per` numbers per item). -/
def checkCount (n : Nat) (per : Nat) (rest : List Int) : Except String Unit :=
  if rest.length = per * n then pure ()
  else throw s!"expected {per * n} numbers after the header, found {rest.length}"

def runFolds : List Int → Except String (List String)
  | nN :: sch :: td :: ft :: lt :: vd :: cd :: hh :: ee :: mt :: ny :: rest => do
    let N ← nat "N" nN
    let trainDays ← nat "train_days" td
    let valDays ← nat "val_days" vd
    let calDays ← nat "cal_days" cd
    let h ← nat "h" hh
    let e ← nat "e" ee
    let minTrain ← nat "min_train_days" mt
    let nyears ← nat "nyears" ny
    let scheme ← match sch with
      | 0 => pure Scheme.expanding
      | 1 => pure (Scheme.rolling trainDays)
      | _ => throw s!"scheme must be 0 (expanding) or 1 (rolling), got {sch}"
    checkCount nyears 2 rest
    let runs := pairs rest
    let mut years : List Int := []
    let mut prev : Option Int := none
    for (y, c) in runs do
      if c < 1 then throw s!"run of year {y} has count {c} < 1"
      if let some p := prev then
        if y ≤ p then throw s!"years must be strictly increasing across runs ({p} then {y})"
      prev := some y
      years := years ++ List.replicate c.toNat y
    if years.length ≠ N then throw s!"run counts sum to {years.length}, not N = {N}"
    let cfg : FoldConfig :=
      { scheme, firstTest := ft, lastTest := lt, valDays, calDays, h, e, minTrain }
    let segs (f : Fold) : List Nat :=
      [f.train.start, f.train.stop, f.val.start, f.val.stop, f.cal.start, f.cal.stop,
        f.test.start, f.test.stop]
    let keeps (f : Fold) : List Nat :=
      [0, 1, 2, 3].flatMap fun i => [(f.seg i).start, f.keepEnd N h e i]
    return (folds cfg years).map fun f =>
      let s := segs f
      let k := keeps f
      s!"fold {f.year} train {s[0]!} {s[1]!} val {s[2]!} {s[3]!} cal {s[4]!} {s[5]!} " ++
        s!"test {s[6]!} {s[7]!} keep " ++ " ".intercalate (k.map toString)
  | _ => throw "folds needs at least 11 numbers"

def runBacktestCmd : List Int → Except String (List String)
  | n :: rest => do
    let n ← nat "n" n
    checkCount n 1 rest
    let acts ← rest.mapM fun (c : Int) =>
      match Action.ofCode? (if c < 0 then 3 else c.toNat) with
      | some a => pure a
      | none => throw s!"action must be 0 (BUY), 1 (SELL) or 2 (HOLD), got {c}"
    let pt := posTrades 0 acts
    return [fmtList "pos" (pt.map Prod.fst), fmtList "trade" (pt.map Prod.snd)]
  | [] => throw "backtest needs n"

def runScan : List Int → Except String (List String)
  | n :: L :: h0 :: rest => do
    let n ← nat "n" n
    let L ← nat "L" L
    if L < 1 then throw "chunk size L must be >= 1"
    checkCount n 2 rest
    let ps : List (Aff Int) := (pairs rest).map fun (a, u) => ⟨a, u⟩
    return [fmtList "seq" (seqScan h0 ps), fmtList "chunk" (chunkedScan L h0 ps)]
  | _ => throw "scan needs n L h0"

def adjSteps (rest : List Int) : List (Step Int) :=
  (triples rest).map fun (a, c, gy) => ⟨a, 0, c, gy⟩

def runAdjoint : List Int → Except String (List String)
  | n :: rest => do
    let n ← nat "n" n
    checkCount n 3 rest
    return [fmtList "g" (adjoint (adjSteps rest))]
  | [] => throw "adjoint needs n"

def runAdjointChunked : List Int → Except String (List String)
  | n :: L :: rest => do
    let n ← nat "n" n
    let L ← nat "L" L
    if L < 1 then throw "chunk size L must be >= 1"
    checkCount n 3 rest
    let xs := adjSteps rest
    return [fmtList "g" (adjoint xs), fmtList "chunk" (chunkedRevScan L 0 (adjPairs xs))]
  | _ => throw "adjointc needs n L"

def runCommand (ws : List String) : Except String (List String) := do
  match ws with
  | [] => throw "empty command"
  | cmd :: args =>
    let some nums := args.mapM parseInt? | throw "arguments must be integers"
    match cmd with
    | "folds" => runFolds nums
    | "backtest" => runBacktestCmd nums
    | "scan" => runScan nums
    | "adjoint" => runAdjoint nums
    | "adjointc" => runAdjointChunked nums
    | _ => throw s!"unknown command {cmd}"

partial def loop (stdin stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  let ws := words line
  match ws with
  | [] => pure ()
  | w :: _ =>
    if w.startsWith "#" then pure ()
    else
      match runCommand ws with
      | .ok out => for l in out do stdout.putStrLn l
      | .error msg => stdout.putStrLn s!"error {msg}"
      stdout.putStrLn "end"
      stdout.flush
  loop stdin stdout

def main : IO Unit := do
  loop (← IO.getStdin) (← IO.getStdout)
