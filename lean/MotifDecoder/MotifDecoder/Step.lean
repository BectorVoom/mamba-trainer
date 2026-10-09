import MotifDecoder.ListLemmas
/-!
Relational form of the machine (`Step`), reachability (`ReachBy`), and their
equivalence with the executable `allowed`/`apply`/`run`.
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

/-- `Step V B s t s'`: token `t` is allowed in `s` and leads to `s'`.  One
constructor per transition of the specification; `step_iff` shows this is
exactly `allowed`/`apply`. -/
inductive Step (V : Vocab) (B : Option Budget) : State → Token → State → Prop
  | start {s : State} {m : Nat} (hp : s.phase = .start) (hm : m < V.size)
      (hb : fits B s.elements (motifAt V m) = true) :
      Step V B s (.motif m) (startMotif V s m)
  | pop {s : State} (hp : s.phase = .body) (hb : s.stack.tail = [] → finished B s = true) :
      Step V B s .end_ (popFrame s)
  | atom {s : State} {a : Nat} (hp : s.phase = .body) (ha : a < s.top.2)
      (hf : 1 ≤ nth s.free (s.top.1 + a)) :
      Step V B s (.atom a) (s.setPhase (.afterAtom a))
  | bond {s : State} {a o : Nat} (hp : s.phase = .afterAtom a) (h1 : 1 ≤ o) (h3 : o ≤ 3)
      (hf : o ≤ nth s.free (s.top.1 + a)) :
      Step V B s (.bond o) (s.setPhase (.afterBond a o))
  | motif {s : State} {a o m : Nat} (hp : s.phase = .afterBond a o) (hm : m < V.size)
      (he : ∃ b, b < (motifAt V m).size ∧ o ≤ nth (motifAt V m).free b)
      (hb : fits B s.elements (motifAt V m) = true) :
      Step V B s (.motif m) (s.setPhase (.afterMotif a o m))
  | attach {s : State} {a o m b : Nat} (hp : s.phase = .afterMotif a o m)
      (hb : b < (motifAt V m).size) (hf : o ≤ nth (motifAt V m).free b) :
      Step V B s (.atom b) (attach V s a o m b)

theorem Step.is_allowed {s s' : State} {t : Token} (h : Step V B s t s') :
    allowed V B s t = true := by
  cases h with
  | start hp hm hb => simp [MotifDecoder.allowed, hp, hm, hb]
  | pop hp hb =>
    simp only [MotifDecoder.allowed, hp]
    cases hs : s.stack.tail with
    | nil => simp [hb hs]
    | cons => simp
  | atom hp ha hf => simp [MotifDecoder.allowed, hp, ha, hf]
  | bond hp h1 h3 hf => simp [MotifDecoder.allowed, hp, h1, h3, hf]
  | motif hp hm he hb => simp [MotifDecoder.allowed, hp, hm, hb, (hasEntry_iff _ _).2 he]
  | attach hp hb hf => simp [MotifDecoder.allowed, hp, hb, hf]

theorem Step.apply_eq {s s' : State} {t : Token} (h : Step V B s t s') : s' = apply V s t := by
  cases h <;> simp [apply, *]

theorem Step.of_allowed {s : State} {t : Token} (h : allowed V B s t = true) :
    Step V B s t (apply V s t) := by
  unfold MotifDecoder.allowed at h
  split at h
  next m hp =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [apply, hp]; exact .start hp h.1 h.2
  next hp =>
    simp only [apply, hp]
    refine .pop hp (fun hs => ?_)
    simpa [hs] using h
  next a hp =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [apply, hp]; exact .atom hp h.1 h.2
  next a o hp =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [apply, hp]; exact .bond hp h.1.1 h.1.2 h.2
  next a o m hp =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [apply, hp]; exact .motif hp h.1.1 ((hasEntry_iff _ _).1 h.1.2) h.2
  next a o m b hp =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [apply, hp]; exact .attach hp h.1 h.2
  next => exact absurd h (by simp)

/-- The relational and the executable machine are the same. -/
theorem step_iff {s s' : State} {t : Token} :
    Step V B s t s' ↔ allowed V B s t = true ∧ s' = apply V s t :=
  ⟨fun h => ⟨h.is_allowed, h.apply_eq⟩, fun ⟨h, e⟩ => e ▸ Step.of_allowed h⟩

theorem step_eq_some_iff {s s' : State} {t : Token} :
    step V B s t = some s' ↔ Step V B s t s' := by
  rw [step_iff]; unfold step
  by_cases h : allowed V B s t = true
  · simp [h, eq_comm]
  · simp [h]

/-- `ReachBy V B ts s`: reading the tokens `ts` from the initial state, every
one allowed in turn, leaves the machine in `s`. -/
inductive ReachBy (V : Vocab) (B : Option Budget) : List Token → State → Prop
  | init : ReachBy V B [] State.init
  | snoc {ts : List Token} {s s' : State} {t : Token} :
      ReachBy V B ts s → Step V B s t s' → ReachBy V B (ts ++ [t]) s'

/-- A state reachable from the initial state by allowed tokens. -/
def Reachable (V : Vocab) (B : Option Budget) (s : State) : Prop := ∃ ts, ReachBy V B ts s

theorem run_append_singleton (s : State) (ts : List Token) (t : Token) :
    run V B s (ts ++ [t]) = (run V B s ts).bind (fun s' => step V B s' t) := by
  induction ts generalizing s with
  | nil => cases h : step V B s t <;> simp [run, h]
  | cons x xs ih =>
    simp only [List.cons_append, run]
    cases step V B s x with
    | none => rfl
    | some s1 => exact ih s1

theorem ReachBy.run_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    run V B State.init ts = some s := by
  induction h with
  | init => rfl
  | snoc _ hs ih => rw [run_append_singleton, ih]; exact step_eq_some_iff.2 hs

theorem ReachBy.of_run {pre ts : List Token} {s s' : State} (hpre : ReachBy V B pre s)
    (h : run V B s ts = some s') : ReachBy V B (pre ++ ts) s' := by
  induction ts generalizing pre s with
  | nil => simp only [run, Option.some.injEq] at h; subst h; simpa using hpre
  | cons t ts ih =>
    simp only [run] at h
    cases hst : step V B s t with
    | none => rw [hst] at h; exact absurd h (by simp)
    | some s1 =>
      rw [hst] at h
      have := ih (hpre.snoc (step_eq_some_iff.1 hst)) h
      simpa using this

/-- The fold over a token list and the inductive reachability agree. -/
theorem reachBy_iff_run {ts : List Token} {s : State} :
    ReachBy V B ts s ↔ run V B State.init ts = some s :=
  ⟨ReachBy.run_eq, fun h => by simpa using ReachBy.of_run ReachBy.init h⟩

theorem reachable_iff {s : State} :
    Reachable V B s ↔ ∃ ts, run V B State.init ts = some s :=
  ⟨fun ⟨ts, h⟩ => ⟨ts, h.run_eq⟩, fun ⟨ts, h⟩ => ⟨ts, reachBy_iff_run.2 h⟩⟩

theorem AcceptedAs.reachBy {ts : List Token} {s : State} (h : AcceptedAs V B ts s) :
    ReachBy V B ts s := reachBy_iff_run.2 h.1

/-! ### The checker used by `motifcheck` -/

theorem checkFrom_ok_iff (s : State) (i : Nat) (ts : List Token) (s' : State) :
    checkFrom V B s i ts = .ok s' ↔ run V B s ts = some s' ∧ s'.phase = .done := by
  induction ts generalizing s i with
  | nil =>
    simp only [checkFrom, run, Option.some.injEq]
    by_cases hp : s.phase = .done
    · simp only [hp, ite_true, Except.ok.injEq]
      constructor
      · rintro rfl; exact ⟨rfl, hp⟩
      · rintro ⟨rfl, _⟩; rfl
    · simp only [hp, ite_false]
      constructor
      · intro h; exact absurd h (by simp)
      · rintro ⟨rfl, h⟩; exact absurd h hp
  | cons t ts ih =>
    simp only [checkFrom, run, step]
    by_cases ha : allowed V B s t = true
    · simp only [ha, ite_true]; exact ih _ _
    · simp [ha]

/-- `check` returns `ok s` exactly for the accepted sequences. -/
theorem check_ok_iff (ts : List Token) (s : State) :
    check V B ts = .ok s ↔ AcceptedAs V B ts s := checkFrom_ok_iff _ _ _ _

/-- What an `ERR j` means: the first `j` tokens are allowed in turn, and either
token `j` is rejected, or `j` is the sequence length and the phase is not `done`. -/
theorem checkFrom_error (s : State) (i : Nat) (ts : List Token) (j : Nat)
    (h : checkFrom V B s i ts = .error j) :
    ∃ k s', j = i + k ∧ k ≤ ts.length ∧ run V B s (ts.take k) = some s' ∧
      ((∃ t, ts[k]? = some t ∧ allowed V B s' t = false) ∨ (k = ts.length ∧ s'.phase ≠ .done)) := by
  induction ts generalizing s i with
  | nil =>
    simp only [checkFrom] at h
    by_cases hp : s.phase = .done
    · simp [hp] at h
    · simp only [hp, ite_false, Except.error.injEq] at h
      exact ⟨0, s, by omega, Nat.le_refl _, rfl, .inr ⟨rfl, hp⟩⟩
  | cons t ts ih =>
    simp only [checkFrom] at h
    by_cases ha : allowed V B s t = true
    · simp only [ha, ite_true] at h
      obtain ⟨k, s', hj, hk, hrun, hrest⟩ := ih _ _ h
      refine ⟨k + 1, s', by omega, by simpa using hk, ?_, ?_⟩
      · simp only [List.take_succ_cons, run, step, ha, ite_true]; exact hrun
      · rcases hrest with ⟨t', ht', hna⟩ | ⟨hk', hp⟩
        · exact .inl ⟨t', by simpa using ht', hna⟩
        · exact .inr ⟨by simp [hk'], hp⟩
    · simp only [ha] at h
      simp only [Bool.false_eq_true, ite_false, Except.error.injEq] at h
      exact ⟨0, s, by omega, Nat.zero_le _, rfl, .inl ⟨t, rfl, by simpa using ha⟩⟩

theorem check_error (ts : List Token) (j : Nat) (h : check V B ts = .error j) :
    j ≤ ts.length ∧ ∃ s', run V B State.init (ts.take j) = some s' ∧
      ((∃ t, ts[j]? = some t ∧ allowed V B s' t = false) ∨ (j = ts.length ∧ s'.phase ≠ .done)) := by
  obtain ⟨k, s', hj, hk, hrun, hrest⟩ := checkFrom_error _ _ _ _ h
  have : j = k := by omega
  subst this
  exact ⟨hk, s', hrun, hrest⟩

end MotifDecoder
