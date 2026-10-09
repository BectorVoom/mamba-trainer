import MotifDecoder.Invariant
import MotifDecoder.Grammar
/-!
Beyond the requested list: the machine only accepts sentences of the grammar.
Every accepted sequence is `serialize t` for a tree `t`, which is unique by
`serialize_injective`.  (The converse, that the serialization of every *valid*
tree is accepted, is not formalised.)
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

theorem serBody_append (xs ys : List Child) : serBody (xs ++ ys) = serBody xs ++ serBody ys := by
  induction xs with
  | nil => simp [serBody]
  | cons x xs ih =>
    obtain ⟨a, o, b, ⟨m, cs⟩⟩ := x
    simp [serBody, ih]

/-- An open motif below the top of the stack: the children its parent had
completed when it was opened, and the tokens `atom a, bond o, motif m, atom b`
that opened it. -/
abbrev Pending := List Child × Nat × Nat × Nat × Nat

def flatPending : List Pending → List Token
  | [] => []
  | (pd, a, o, m, b) :: rest =>
    flatPending rest ++ serBody pd ++ [.atom a, .bond o, .motif m, .atom b]

def Phase.pendingTokens : Phase → List Token
  | .afterAtom a => [.atom a]
  | .afterBond a o => [.atom a, .bond o]
  | .afterMotif a o m => [.atom a, .bond o, .motif m]
  | _ => []

/-- The tokens read so far are a prefix of a sentence: the root motif, the
pending open motifs (bottom first), the children completed in the top motif,
and the tokens of a half-read attachment. -/
def OpenPrefix (ts : List Token) (s : State) : Prop :=
  ∃ (m0 : Nat) (pend : List Pending) (cur : List Child),
    s.stack.length = pend.length + 1 ∧
      ts = .motif m0 :: (flatPending pend ++ serBody cur) ++ s.phase.pendingTokens

structure GrammarInv (ts : List Token) (s : State) : Prop where
  start : s.phase = .start → ts = []
  done : s.phase = .done → ∃ t, ts = serialize t
  inner : s.phase ≠ .start → s.phase ≠ .done → OpenPrefix ts s

theorem ReachBy.grammarInv {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    GrammarInv ts s := by
  induction h with
  | init => exact ⟨fun _ => rfl, (fun h => by cases h), fun h => absurd rfl h⟩
  | @snoc ts s s' t hr hs ih =>
    have hi := hr.inv
    cases hs with
    | @start m hp hm hb =>
      have h0 := ih.start hp
      have hs0 := hi.start_eq hp
      subst h0; subst hs0
      refine ⟨(fun h => by cases h), (fun h => by cases h), fun _ _ => ⟨m, [], [], rfl, ?_⟩⟩
      simp [flatPending, serBody, startMotif, Phase.pendingTokens]
    | pop hp hb =>
      obtain ⟨m0, pend, cur, hlen, hts⟩ := ih.inner (by simp [hp]) (by simp [hp])
      simp only [hp, Phase.pendingTokens, List.append_nil] at hts
      cases hstk : s.stack with
      | nil => simp [hstk] at hlen
      | cons f rest =>
        simp only [hstk, List.length_cons, Nat.add_right_cancel_iff] at hlen
        cases pend with
        | nil =>
          have hrest : rest = [] := List.eq_nil_of_length_eq_zero hlen
          have hph : (popFrame s).phase = .done := by simp [popFrame, hstk, hrest]
          refine ⟨(fun h => by rw [hph] at h; cases h), fun _ => ⟨.node m0 cur, ?_⟩,
            fun _ h => absurd hph h⟩
          simp [hts, serialize, flatPending]
        | cons p pend' =>
          obtain ⟨pd, a, o, m, b⟩ := p
          have hrest : rest ≠ [] := by
            intro h0; simp [h0] at hlen
          have hph : (popFrame s).phase = .body := by simp [popFrame, hstk, hrest]
          refine ⟨(fun h => by rw [hph] at h; cases h), (fun h => by rw [hph] at h; cases h),
            fun _ _ => ⟨m0, pend', pd ++ [(a, o, b, .node m cur)], ?_, ?_⟩⟩
          · simp only [popFrame, hstk, List.tail_cons]
            simpa using hlen
          · rw [hph]
            simp [hts, flatPending, serBody_append, serBody, Phase.pendingTokens]
    | @atom a hp ha hf =>
      obtain ⟨m0, pend, cur, hlen, hts⟩ := ih.inner (by simp [hp]) (by simp [hp])
      simp only [hp, Phase.pendingTokens, List.append_nil] at hts
      refine ⟨(fun h => by cases h), (fun h => by cases h), fun _ _ => ⟨m0, pend, cur, hlen, ?_⟩⟩
      simp [hts, State.setPhase, Phase.pendingTokens]
    | @bond a o hp h1 h3 hf =>
      obtain ⟨m0, pend, cur, hlen, hts⟩ := ih.inner (by simp [hp]) (by simp [hp])
      simp only [hp, Phase.pendingTokens] at hts
      refine ⟨(fun h => by cases h), (fun h => by cases h), fun _ _ => ⟨m0, pend, cur, hlen, ?_⟩⟩
      simp [hts, State.setPhase, Phase.pendingTokens]
    | @motif a o m hp hm he hb =>
      obtain ⟨m0, pend, cur, hlen, hts⟩ := ih.inner (by simp [hp]) (by simp [hp])
      simp only [hp, Phase.pendingTokens] at hts
      refine ⟨(fun h => by cases h), (fun h => by cases h), fun _ _ => ⟨m0, pend, cur, hlen, ?_⟩⟩
      simp [hts, State.setPhase, Phase.pendingTokens]
    | @attach a o m b hp hb hf =>
      obtain ⟨m0, pend, cur, hlen, hts⟩ := ih.inner (by simp [hp]) (by simp [hp])
      simp only [hp, Phase.pendingTokens] at hts
      refine ⟨(fun h => by cases h), (fun h => by cases h),
        fun _ _ => ⟨m0, (cur, a, o, m, b) :: pend, [], ?_, ?_⟩⟩
      · simp [attach, hlen]
      · simp [hts, attach, flatPending, serBody, Phase.pendingTokens]

/-- Every accepted sequence is the serialization of a motif tree. -/
theorem Accepted.exists_tree {ts : List Token} (h : Accepted V B ts) :
    ∃ t, serialize t = ts := by
  obtain ⟨s, hs⟩ := h
  obtain ⟨t, ht⟩ := hs.reachBy.grammarInv.done hs.2
  exact ⟨t, ht.symm⟩

/-- Every accepted sequence parses, and its tree serializes back to it. -/
theorem Accepted.parses {ts : List Token} (h : Accepted V B ts) :
    ∃ t, parse ts = some t ∧ serialize t = ts := by
  obtain ⟨t, ht⟩ := h.exists_tree
  exact ⟨t, ht ▸ parse_serialize t, ht⟩

end MotifDecoder
