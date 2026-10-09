import MotifDecoder.Basic
/-!
`motifcheck <vocab-file> <sequences-file> [<budgets-file>]`

Runs `MotifDecoder.check` (that is, `allowed`/`apply`, the definitions the
theorems are about) on every sequence.  This file only reads and prints; it
imports nothing but the definitions (`MotifDecoder/Basic.lean`).
-/
open MotifDecoder

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

def parseNat (w : String) : Option Nat := natOfDigits w.toList

def parseToken (w : String) : Option Token :=
  match w.toList with
  | ['E'] => some .end_
  | 'M' :: ds => (natOfDigits ds).map .motif
  | 'A' :: ds => (natOfDigits ds).map .atom
  | 'B' :: ds => (natOfDigits ds).map .bond
  | _ => none

def triples : List Nat → List Bond
  | i :: j :: o :: rest => (i, j, o) :: triples rest
  | _ => []

/-- `n e_0 … e_{n-1} f_0 … f_{n-1} k i_1 j_1 o_1 … i_k j_k o_k` -/
def parseMotif (line : String) : Except String Motif := do
  let some nums := (words line).mapM parseNat | throw "not a list of natural numbers"
  match nums with
  | [] => throw "empty line"
  | n :: rest =>
    if rest.length < 2 * n + 1 then throw "too few numbers" else
    let k := rest.getD (2 * n) 0
    let bs := rest.drop (2 * n + 1)
    if bs.length ≠ 3 * k then throw s!"expected {3 * k} bond numbers, found {bs.length}" else
    return { elements := rest.take n, free := (rest.drop n).take n, bonds := triples bs }

def parseVocab (lines : List String) : Except String Vocab := do
  match lines.filter (fun l => !(words l).isEmpty) with
  | [] => throw "empty vocabulary file"
  | first :: rest =>
    let some n := (match words first with | [w] => parseNat w | _ => none)
      | throw "first line must be the number of motifs"
    if rest.length ≠ n then throw s!"expected {n} motif lines, found {rest.length}" else
    let mut out : Array Motif := #[]
    for line in rest do
      match parseMotif line with
      | .ok m => out := out.push m
      | .error e => throw s!"motif {out.size}: {e}"
    return out

/-- `H e_1 c_1 e_2 c_2 …`; an empty line or `-` means no budget. -/
def parseBudget (line : String) : Except String (Option Budget) := do
  let ws := words line
  if ws.isEmpty || ws == ["-"] then return none
  let some nums := ws.mapM parseNat | throw "not a list of natural numbers"
  match nums with
  | [] => return none
  | h :: rest =>
    if rest.length % 2 ≠ 0 then throw "element/count pairs expected" else
    let rec pairs : List Nat → List (Nat × Nat)
      | e :: c :: more => (e, c) :: pairs more
      | _ => []
    return some { target := pairs rest, hydrogens := h }

def fmtNats (l : List Nat) : String := " ".intercalate (l.map toString)

def fmtState (s : State) : String :=
  s!"OK {s.elements.length} {s.bonds.length} | {fmtNats s.elements} | {fmtNats s.free} | " ++
    fmtNats (s.bonds.flatMap (fun b => [b.1, b.2.1, b.2.2]))

/-- Longest prefix of well-formed tokens, and whether the whole line was well-formed. -/
def parseTokens : List String → List Token × Bool
  | [] => ([], true)
  | w :: ws =>
    match parseToken w with
    | none => ([], false)
    | some t => let (ts, ok) := parseTokens ws; (t :: ts, ok)

/-- One output line.  A word that is not a token counts as a rejected token. -/
def checkLine (V : Vocab) (B : Option Budget) (line : String) : String :=
  let (ts, wellFormed) := parseTokens (words line)
  match check V B ts with
  | .ok s => if wellFormed then fmtState s else s!"ERR {ts.length}"
  | .error i => s!"ERR {i}"

def main (args : List String) : IO UInt32 := do
  let err ← IO.getStderr
  let (vocabPath, seqPath, budgetPath?) ←
    match args with
    | [v, s] => pure (v, s, none)
    | [v, s, b] => pure (v, s, some b)
    | _ =>
      err.putStrLn "usage: motifcheck <vocab-file> <sequences-file> [<budgets-file>]"
      return 2
  let V ←
    match parseVocab (← IO.FS.lines vocabPath).toList with
    | .ok V => pure V
    | .error e => err.putStrLn s!"{vocabPath}: {e}"; return 1
  if !decide (VocabWF V) then
    err.putStrLn s!"warning: {vocabPath} is not a well-formed vocabulary (the theorems assume VocabWF)"
  let seqs ← IO.FS.lines seqPath
  let mut budgets : Array (Option Budget) := #[]
  if let some bp := budgetPath? then
    for line in (← IO.FS.lines bp) do
      match parseBudget line with
      | .ok b => budgets := budgets.push b
      | .error e => err.putStrLn s!"{bp}: line {budgets.size + 1}: {e}"; return 1
    if budgets.size ≠ seqs.size then
      err.putStrLn s!"{bp}: {budgets.size} budgets for {seqs.size} sequences"; return 1
  let out ← IO.getStdout
  for h : i in [0:seqs.size] do
    out.putStrLn (checkLine V (budgets.getD i none) seqs[i])
  return 0
