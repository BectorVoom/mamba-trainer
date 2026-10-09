import MotifDecoder.Accounting
/-!
Item 5: mask facts, the token count of an accepted sequence, and (beyond the
list) that the ghost counters agree with the token sequence and that the
unbudgeted machine has no dead end.
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

/-- **Item 5a.** Without a budget, `end_` is always allowed in `body`. -/
theorem end_allowed_in_body {s : State} (hp : s.phase = .body) :
    allowed V none s .end_ = true := by
  simp [allowed, hp]

/-- With a budget, `end_` is still always allowed in `body` unless it would
empty the stack. -/
theorem end_allowed_in_body_inner {s : State} (hp : s.phase = .body) (h : s.stack.tail ≠ []) :
    allowed V B s .end_ = true := by
  simp [allowed, hp, h]

/-- **Item 5b.** If `motif m` is allowed after a bond, the following
`afterMotif` phase has at least one allowed `atom b`. -/
theorem entry_exists {s : State} {a o m : Nat} (hp : s.phase = .afterBond a o)
    (h : allowed V B s (.motif m) = true) :
    ∃ b, allowed V B (apply V s (.motif m)) (.atom b) = true := by
  simp only [allowed, hp, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨b, hb, hf⟩ := (hasEntry_iff _ _).1 h.1.2
  refine ⟨b, ?_⟩
  simp [allowed, apply, hp, State.setPhase, hb, hf]

/-- **Item 5c.** In `afterAtom` reached from `body`, `bond 1` is allowed. -/
theorem bond_one_allowed {s : State} {a : Nat} (hp : s.phase = .body)
    (h : allowed V B s (.atom a) = true) :
    allowed V B (apply V s (.atom a)) (.bond 1) = true := by
  simp only [allowed, hp, Bool.and_eq_true, decide_eq_true_eq] at h
  have e : apply V s (.atom a) = s.setPhase (.afterAtom a) := by simp [apply, hp]
  rw [e]
  exact (Step.bond (s := s.setPhase (.afterAtom a)) rfl (Nat.le_refl 1) (by omega) h.2).is_allowed

/-- The same for every reachable `afterAtom` state (they are all reached from `body`). -/
theorem ReachBy.bond_one_allowed {ts : List Token} {s : State} {a : Nat}
    (h : ReachBy V B ts s) (hp : s.phase = .afterAtom a) :
    allowed V B s (.bond 1) = true := by
  obtain ⟨_, _, hf⟩ := h.inv.afterAtom hp
  simp [allowed, hp, hf]

/-! ### Token count -/

def Phase.offset : Phase → Nat
  | .start => 3
  | .body => 0
  | .afterAtom _ => 1
  | .afterBond _ _ => 2
  | .afterMotif _ _ _ => 3
  | .done => 0

/-- Length of the token sequence in terms of motifs added, open frames and phase. -/
theorem ReachBy.length_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    ts.length + s.stack.length + 3 = 5 * s.added.length + s.phase.offset := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    have hi := hr.inv
    cases hs with
    | start hp hm hb =>
      simp only [hp, Phase.offset] at ih
      simp [startMotif, Phase.offset]; omega
    | pop hp hb =>
      simp only [hp, Phase.offset] at ih
      have hne := hi.body_ne hp
      have hl : (popFrame s).stack.length + 1 = s.stack.length := by
        cases hs : s.stack with
        | nil => exact absurd hs hne
        | cons f rest => simp [popFrame, hs]
      have ha : (popFrame s).added = s.added := rfl
      have ho : (popFrame s).phase.offset = 0 := by
        rcases popFrame_phase s with h' | h' <;> rw [h'] <;> rfl
      rw [ha, ho]; simp; omega
    | atom hp ha hf =>
      simp only [hp, Phase.offset] at ih
      simp [State.setPhase, Phase.offset]; omega
    | bond hp h1 h3 hf =>
      simp only [hp, Phase.offset] at ih
      simp [State.setPhase, Phase.offset]; omega
    | motif hp hm he hb =>
      simp only [hp, Phase.offset] at ih
      simp [State.setPhase, Phase.offset]; omega
    | attach hp hb hf =>
      simp only [hp, Phase.offset] at ih
      simp [attach, Phase.offset]; omega

/-- **Item 5d.** An accepted sequence that added `k` motifs has exactly
`5 * k − 3` tokens. -/
theorem AcceptedAs.length_eq {ts : List Token} {s : State} (h : AcceptedAs V B ts s) :
    ts.length = 5 * s.added.length - 3 := by
  have h1 := h.reachBy.length_eq
  have h2 := (h.reachBy.inv.done h.2).1
  simp only [h.2, h2, Phase.offset, List.length_nil] at h1
  omega

/-- The same without truncating subtraction. -/
theorem AcceptedAs.length_add {ts : List Token} {s : State} (h : AcceptedAs V B ts s) :
    ts.length + 3 = 5 * s.added.length := by
  have h1 := h.reachBy.length_eq
  have h2 := (h.reachBy.inv.done h.2).1
  simp only [h.2, h2, Phase.offset, List.length_nil] at h1
  omega

/-! ### The ghost counters agree with the token sequence -/

/-- The motif ids named by the `motif` tokens, in order. -/
def tokenMotifs (ts : List Token) : List Nat :=
  ts.filterMap (fun t => match t with | .motif m => some m | _ => none)

/-- Sum of the orders named by the `bond` tokens. -/
def tokenOrders (ts : List Token) : Nat :=
  (ts.filterMap (fun t => match t with | .bond o => some o | _ => none)).sum

def Phase.pendingMotif : Phase → List Nat
  | .afterMotif _ _ m => [m]
  | _ => []

def Phase.pendingOrder : Phase → Nat
  | .afterBond _ o => o
  | .afterMotif _ o _ => o
  | _ => 0

theorem ReachBy.tokenMotifs_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    tokenMotifs ts = s.added ++ s.phase.pendingMotif := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb =>
      simp only [hp, Phase.pendingMotif, List.append_nil] at ih
      simp [tokenMotifs, List.filterMap_append, startMotif, Phase.pendingMotif] at ih ⊢
      exact ih
    | pop hp hb =>
      simp only [hp, Phase.pendingMotif, List.append_nil] at ih
      have ho : (popFrame s).phase.pendingMotif = [] := by
        rcases popFrame_phase s with h' | h' <;> rw [h'] <;> rfl
      have ha : (popFrame s).added = s.added := rfl
      rw [ho, ha]
      simp [tokenMotifs, List.filterMap_append] at ih ⊢
      exact ih
    | atom hp ha hf =>
      simp only [hp, Phase.pendingMotif, List.append_nil] at ih
      simp [tokenMotifs, List.filterMap_append, State.setPhase, Phase.pendingMotif] at ih ⊢
      exact ih
    | bond hp h1 h3 hf =>
      simp only [hp, Phase.pendingMotif, List.append_nil] at ih
      simp [tokenMotifs, List.filterMap_append, State.setPhase, Phase.pendingMotif] at ih ⊢
      exact ih
    | motif hp hm he hb =>
      simp only [hp, Phase.pendingMotif, List.append_nil] at ih
      simp [tokenMotifs, List.filterMap_append, State.setPhase, Phase.pendingMotif] at ih ⊢
      exact ih
    | attach hp hb hf =>
      simp only [hp, Phase.pendingMotif] at ih
      simp [tokenMotifs, List.filterMap_append, attach, Phase.pendingMotif] at ih ⊢
      exact ih

theorem ReachBy.tokenOrders_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    tokenOrders ts = s.attachOrder + s.phase.pendingOrder := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb =>
      simp only [hp, Phase.pendingOrder] at ih
      simp [tokenOrders, List.filterMap_append, startMotif, Phase.pendingOrder] at ih ⊢
      exact ih
    | pop hp hb =>
      simp only [hp, Phase.pendingOrder] at ih
      have ho : (popFrame s).phase.pendingOrder = 0 := by
        rcases popFrame_phase s with h' | h' <;> rw [h'] <;> rfl
      have ha : (popFrame s).attachOrder = s.attachOrder := rfl
      rw [ho, ha]
      simp [tokenOrders, List.filterMap_append] at ih ⊢
      exact ih
    | atom hp ha hf =>
      simp only [hp, Phase.pendingOrder] at ih
      simp [tokenOrders, List.filterMap_append, State.setPhase, Phase.pendingOrder] at ih ⊢
      exact ih
    | bond hp h1 h3 hf =>
      simp only [hp, Phase.pendingOrder] at ih
      simp [tokenOrders, List.filterMap_append, State.setPhase, Phase.pendingOrder] at ih ⊢
      exact ih
    | motif hp hm he hb =>
      simp only [hp, Phase.pendingOrder] at ih
      simp [tokenOrders, List.filterMap_append, State.setPhase, Phase.pendingOrder] at ih ⊢
      exact ih
    | attach hp hb hf =>
      simp only [hp, Phase.pendingOrder] at ih
      simp [tokenOrders, List.filterMap_append, attach, Phase.pendingOrder] at ih ⊢
      exact ih

/-- For an accepted sequence the ghost list `added` is the list of `motif` tokens. -/
theorem AcceptedAs.added_eq {ts : List Token} {s : State} (h : AcceptedAs V B ts s) :
    s.added = tokenMotifs ts := by
  have := h.reachBy.tokenMotifs_eq
  simp only [h.2, Phase.pendingMotif, List.append_nil] at this
  exact this.symm

/-- For an accepted sequence the ghost counter `attachOrder` is the sum of the `bond` tokens. -/
theorem AcceptedAs.attachOrder_eq {ts : List Token} {s : State} (h : AcceptedAs V B ts s) :
    s.attachOrder = tokenOrders ts := by
  have := h.reachBy.tokenOrders_eq
  simp only [h.2, Phase.pendingOrder] at this
  omega

/-- **Item 5d, in terms of the tokens only.**  An accepted sequence with `k`
`motif` tokens has exactly `5 * k − 3` tokens. -/
theorem Accepted.length_eq {ts : List Token} (h : Accepted V B ts) :
    ts.length = 5 * (tokenMotifs ts).length - 3 := by
  obtain ⟨s, hs⟩ := h
  rw [← hs.added_eq]; exact hs.length_eq

/-! ### No dead end without a budget (beyond the requested list) -/

theorem ReachBy.added_lt {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    ∀ m ∈ s.added, m < V.size := by
  induction h with
  | init => intro m hm; simp [State.init] at hm
  | @snoc ts s s' t hr hs ih =>
    have hi := hr.inv
    cases hs with
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | start hp hm hb =>
      intro x hx
      simp only [startMotif, List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · exact ih x hx
      · exact hm
    | attach hp hb hf =>
      obtain ⟨_, _, _, _, _, hm, _, _⟩ := hi.afterMotif hp
      intro x hx
      simp only [attach, List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · exact ih x hx
      · exact hm

theorem nth_flatMap_pos (g : Nat → List Nat) (l : List Nat) (i : Nat)
    (h : 0 < nth (l.flatMap g) i) : ∃ m ∈ l, ∃ b, nth (g m) b = nth (l.flatMap g) i := by
  induction l generalizing i with
  | nil => simp at h
  | cons x xs ih =>
    simp only [List.flatMap_cons] at h ⊢
    by_cases hi : i < (g x).length
    · exact ⟨x, List.mem_cons_self .., i, (nth_append_left hi).symm⟩
    · obtain ⟨k, rfl⟩ : ∃ k, i = (g x).length + k := ⟨i - (g x).length, by omega⟩
      rw [nth_append_right] at h ⊢
      obtain ⟨m, hm, b, hb⟩ := ih k h
      exact ⟨m, List.mem_cons_of_mem _ hm, b, hb⟩

/-- After any allowed `bond o` some motif of the vocabulary is allowed (without
a budget): the source atom's own motif has an atom with `free ≥ o`. -/
theorem ReachBy.motif_exists (hV : VocabWF V) {ts : List Token} {s : State} {a o : Nat}
    (h : ReachBy V none ts s) (hp : s.phase = .afterBond a o) :
    ∃ m, allowed V none s (.motif m) = true := by
  obtain ⟨_, _, h1, _, hf⟩ := h.inv.afterBond hp
  have hle : nth s.free (s.top.1 + a) ≤
      nth (s.added.flatMap (fun m => (motifAt V m).free)) (s.top.1 + a) :=
    h.free_le_orig.2 (s.top.1 + a)
  have hpos : 0 < nth (s.added.flatMap (fun m => (motifAt V m).free)) (s.top.1 + a) := by omega
  obtain ⟨m, hm, b, hb⟩ := nth_flatMap_pos _ _ _ hpos
  have hmlt := h.added_lt m hm
  have hbpos : 0 < nth (motifAt V m).free b := by rw [hb]; exact hpos
  have hblt : b < (motifAt V m).size := by
    rw [← (hV m hmlt).free_len]; exact lt_length_of_nth_pos hbpos
  refine ⟨m, ?_⟩
  have he : hasEntry (motifAt V m) o = true :=
    (hasEntry_iff _ _).2 ⟨b, hblt, by rw [hb]; omega⟩
  simp [allowed, hp, hmlt, he]

/-- **Progress.**  With a well-formed, non-empty vocabulary and no budget, every
reachable state that is not `done` has an allowed token. -/
theorem ReachBy.progress (hV : VocabWF V) (hne : 0 < V.size) {ts : List Token} {s : State}
    (h : ReachBy V none ts s) (hnd : s.phase ≠ .done) : ∃ t, allowed V none s t = true := by
  cases hp : s.phase with
  | start => exact ⟨.motif 0, by simp [allowed, hp, hne]⟩
  | body => exact ⟨.end_, end_allowed_in_body hp⟩
  | afterAtom a => exact ⟨.bond 1, h.bond_one_allowed hp⟩
  | afterBond a o =>
    obtain ⟨m, hm⟩ := h.motif_exists hV hp
    exact ⟨.motif m, hm⟩
  | afterMotif a o m =>
    obtain ⟨_, _, _, _, _, _, _, b, hb, hf⟩ := h.inv.afterMotif hp
    exact ⟨.atom b, by simp [allowed, hp, hb, hf]⟩
  | done => exact absurd hp hnd

/-- In `done` nothing is allowed. -/
theorem done_nothing_allowed {s : State} (hp : s.phase = .done) (t : Token) :
    allowed V B s t = false := by
  cases t <;> simp [allowed, hp]

end MotifDecoder
