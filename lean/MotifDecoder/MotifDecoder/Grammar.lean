import MotifDecoder.Basic
/-!
Item 7: the grammar is unambiguous.

    molecule := motif m  body  end_
    body     := ( atom a  bond o  motif m'  atom b  body  end_ )*
-/
namespace MotifDecoder

/-- A motif tree: a motif id and its children, each `(a, o, b, subtree)` with
attach atom `a`, bond order `o` and entry atom `b`. -/
inductive MotifTree where
  | node (m : Nat) (children : List (Nat × Nat × Nat × MotifTree))

abbrev Child := Nat × Nat × Nat × MotifTree

/-- `body` of the grammar. -/
def serBody : List Child → List Token
  | [] => []
  | (a, o, b, .node m cs) :: rest =>
    .atom a :: .bond o :: .motif m :: .atom b :: (serBody cs ++ .end_ :: serBody rest)

/-- `molecule` of the grammar. -/
def serialize : MotifTree → List Token
  | .node m cs => .motif m :: (serBody cs ++ [.end_])

/-- Parse a `body` with a recursion budget; returns the children and the
unread tokens.  A body ends at the first token that does not start a child. -/
def parseBody : Nat → List Token → Option (List Child × List Token)
  | 0, _ => none
  | fuel + 1, .atom a :: .bond o :: .motif m :: .atom b :: ts =>
    match parseBody fuel ts with
    | some (cs, .end_ :: ts') =>
      match parseBody fuel ts' with
      | some (rest, ts'') => some ((a, o, b, .node m cs) :: rest, ts'')
      | none => none
    | _ => none
  | _ + 1, ts => some ([], ts)

/-- Parse a whole sequence (fuel: its length, which is always enough). -/
def parse : List Token → Option MotifTree
  | .motif m :: ts =>
    match parseBody (ts.length + 1) ts with
    | some (cs, [.end_]) => some (.node m cs)
    | _ => none
  | _ => none

theorem parseBody_serBody : ∀ (fuel : Nat) (cs : List Child) (rest : List Token),
    (serBody cs).length < fuel →
      parseBody fuel (serBody cs ++ .end_ :: rest) = some (cs, .end_ :: rest)
  | 0, _, _, h => absurd h (Nat.not_lt_zero _)
  | fuel + 1, [], rest, _ => by simp [serBody, parseBody]
  | fuel + 1, (a, o, b, .node m cs) :: more, rest, h => by
    have h' : (serBody cs).length + (serBody more).length + 5 < fuel + 1 := by
      simp only [serBody, List.length_cons, List.length_append] at h; omega
    simp only [serBody, List.cons_append, List.append_assoc, parseBody]
    rw [parseBody_serBody fuel cs _ (by omega)]
    simp only
    rw [parseBody_serBody fuel more rest (by omega)]

/-- **Item 7.** Round trip: parsing a serialized tree gives the tree back. -/
theorem parse_serialize (t : MotifTree) : parse (serialize t) = some t := by
  cases t with
  | node m cs =>
    simp only [serialize, parse]
    rw [parseBody_serBody _ cs [] (by simp only [List.length_append, List.length_cons, List.length_nil]; omega)]

/-- Hence `serialize` is injective: a token sequence has at most one tree. -/
theorem serialize_injective {t₁ t₂ : MotifTree} (h : serialize t₁ = serialize t₂) : t₁ = t₂ := by
  have h1 := parse_serialize t₁
  rw [h, parse_serialize t₂] at h1
  exact (Option.some.inj h1).symm

/-- The parser only accepts serializations: whatever `parseBody` reads is the
serialization of what it returns. -/
theorem serBody_of_parseBody : ∀ (fuel : Nat) (ts : List Token) (cs : List Child)
    (rest : List Token), parseBody fuel ts = some (cs, rest) → ts = serBody cs ++ rest := by
  intro fuel
  induction fuel with
  | zero => intro ts cs rest h; simp [parseBody] at h
  | succ fuel ih =>
    intro ts cs rest h
    unfold parseBody at h
    split at h
    · exact absurd h (by simp)
    · next fuel' a o m b ts0 heq =>
      cases heq
      split at h
      · next cs0 ts' h0 =>
        split at h
        · next more ts'' h1 =>
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          have e0 := ih _ _ _ h0
          have e1 := ih _ _ _ h1
          simp only [serBody, List.cons_append, List.append_assoc]
          rw [e0, e1]
        · exact absurd h (by simp)
      · exact absurd h (by simp)
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp [serBody]

/-- Converse round trip: a sequence that parses is the serialization of its tree. -/
theorem serialize_of_parse {ts : List Token} {t : MotifTree} (h : parse ts = some t) :
    serialize t = ts := by
  unfold parse at h
  split at h
  · next m ts0 =>
    split at h
    · next cs h0 =>
      simp only [Option.some.injEq] at h
      subst h
      have := serBody_of_parseBody _ _ _ _ h0
      simp only [serialize]; rw [this]
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- `parse` and `serialize` are inverse bijections between motif trees and the
token sequences of the grammar. -/
theorem parse_eq_some_iff (ts : List Token) (t : MotifTree) :
    parse ts = some t ↔ serialize t = ts :=
  ⟨serialize_of_parse, fun h => h ▸ parse_serialize t⟩

end MotifDecoder
