import MotifDecoder.Basic
/-! Facts about `nth`, `subAt`, `shiftBond` and the budget tests. -/
namespace MotifDecoder

theorem nth_eq_getD (l : List Nat) (i : Nat) : nth l i = l.getD i 0 := by
  induction l generalizing i with
  | nil => simp [nth]
  | cons x xs ih => cases i <;> simp [nth, ih]

@[simp] theorem nth_nil (i : Nat) : nth [] i = 0 := rfl

theorem nth_of_length_le {l : List Nat} {i : Nat} (h : l.length ≤ i) : nth l i = 0 := by
  induction l generalizing i with
  | nil => rfl
  | cons x xs ih =>
    cases i with
    | zero => simp at h
    | succ i => simp only [nth]; exact ih (by simpa using h)

theorem lt_length_of_nth_pos {l : List Nat} {i : Nat} (h : 0 < nth l i) : i < l.length := by
  apply Nat.lt_of_not_le
  intro hle
  rw [nth_of_length_le hle] at h
  exact Nat.lt_irrefl _ h

theorem nth_append_left {l r : List Nat} {i : Nat} (h : i < l.length) :
    nth (l ++ r) i = nth l i := by
  induction l generalizing i with
  | nil => simp at h
  | cons x xs ih =>
    cases i with
    | zero => rfl
    | succ i => simp only [List.cons_append, nth]; exact ih (by simpa using h)

theorem nth_append_right (l r : List Nat) (i : Nat) : nth (l ++ r) (l.length + i) = nth r i := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    have : (x :: xs).length + i = (xs.length + i) + 1 := by simp; omega
    rw [this]; simp only [List.cons_append, nth]; exact ih

@[simp] theorem length_subAt (l : List Nat) (i o : Nat) : (subAt l i o).length = l.length := by
  induction l generalizing i with
  | nil => rfl
  | cons x xs ih => cases i <;> simp [subAt, ih]

theorem nth_subAt_same (l : List Nat) (i o : Nat) : nth (subAt l i o) i = nth l i - o := by
  induction l generalizing i with
  | nil => simp [subAt]
  | cons x xs ih => cases i <;> simp [subAt, nth, ih]

theorem nth_subAt_ne (l : List Nat) {i j : Nat} (o : Nat) (h : i ≠ j) :
    nth (subAt l i o) j = nth l j := by
  induction l generalizing i j with
  | nil => simp [subAt]
  | cons x xs ih =>
    cases i with
    | zero =>
      cases j with
      | zero => exact absurd rfl h
      | succ j => simp [subAt, nth]
    | succ i =>
      cases j with
      | zero => simp [subAt, nth]
      | succ j => simp only [subAt, nth]; exact ih (fun e => h (by rw [e]))

theorem nth_subAt_le (l : List Nat) (i o j : Nat) : nth (subAt l i o) j ≤ nth l j := by
  by_cases h : i = j
  · subst h; rw [nth_subAt_same]; exact Nat.sub_le _ _
  · rw [nth_subAt_ne l o h]; exact Nat.le_refl _

/-- With the guard `o ≤ l[i]` the subtraction is exact, so the sum drops by exactly `o`. -/
theorem sum_subAt {l : List Nat} {i o : Nat} (h : o ≤ nth l i) : (subAt l i o).sum + o = l.sum := by
  induction l generalizing i with
  | nil => simp [nth] at h; simp [subAt, h]
  | cons x xs ih =>
    cases i with
    | zero => simp only [nth] at h; simp only [subAt, List.sum_cons]; omega
    | succ i =>
      simp only [nth] at h
      simp only [subAt, List.sum_cons]
      have := ih h
      omega

theorem nth_append_le {l₁ l₂ : List Nat} (r : List Nat) (hlen : l₁.length = l₂.length)
    (h : ∀ i, nth l₁ i ≤ nth l₂ i) (i : Nat) : nth (l₁ ++ r) i ≤ nth (l₂ ++ r) i := by
  by_cases hi : i < l₁.length
  · rw [nth_append_left hi, nth_append_left (hlen ▸ hi)]; exact h i
  · obtain ⟨k, rfl⟩ : ∃ k, i = l₁.length + k := ⟨i - l₁.length, by omega⟩
    rw [nth_append_right]; rw [hlen, nth_append_right]; exact Nat.le_refl _

/-! ### Bonds -/

@[simp] theorem shiftBond_order (n : Nat) (b : Bond) : (shiftBond n b).2.2 = b.2.2 := rfl

theorem BondOK.mono {n n' : Nat} {b : Bond} (h : BondOK n b) (hn : n ≤ n') : BondOK n' b := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  exact ⟨Nat.lt_of_lt_of_le h1 hn, Nat.lt_of_lt_of_le h2 hn, h3, h4, h5⟩

theorem BondOK.shift {k : Nat} {b : Bond} (h : BondOK k b) (n : Nat) :
    BondOK (n + k) (shiftBond n b) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  refine ⟨?_, ?_, ?_, h4, h5⟩ <;> simp only [shiftBond] <;> omega

/-! ### Budget tests -/

theorem targetCount_eq_zero {t : List (Nat × Nat)} {e : Nat} (h : e ∉ t.map Prod.fst) :
    targetCount t e = 0 := by
  induction t with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, c⟩ := p
    simp only [List.map_cons, List.mem_cons, not_or] at h
    have hk : ¬ k = e := fun hk => h.1 hk.symm
    simp only [targetCount, hk, ite_false]
    exact ih h.2

/-- On a state within the target, the mask's test (over the motif's own
elements only) is the specification's "for every element". -/
theorem fits_some_iff (b : Budget) (els : List Nat) (mt : Motif)
    (h : ∀ e, els.count e ≤ b.count e) :
    fits (some b) els mt = true ↔ ∀ e, mt.elements.count e + els.count e ≤ b.count e := by
  simp only [fits, List.all_eq_true, decide_eq_true_eq]
  constructor
  · intro hh e
    by_cases he : e ∈ mt.elements
    · exact hh e he
    · rw [List.count_eq_zero_of_not_mem he]; simpa using h e
  · intro hh e _; exact hh e

@[simp] theorem fits_none (els : List Nat) (mt : Motif) : fits none els mt = true := rfl

/-- The mask's final test is exactly: element counts equal the target (for
every element) and the free valences sum to `H`. -/
theorem finished_some_iff (b : Budget) (s : State) :
    finished (some b) s = true ↔
      (∀ e, s.elements.count e = b.count e) ∧ s.free.sum = b.hydrogens := by
  simp only [finished, Bool.and_eq_true, List.all_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun e => ?_, h2⟩
    by_cases he : e ∈ s.elements ++ b.target.map Prod.fst
    · exact h1 e he
    · rw [List.mem_append, not_or] at he
      rw [List.count_eq_zero_of_not_mem he.1]
      exact (targetCount_eq_zero he.2).symm
  · rintro ⟨h1, h2⟩; exact ⟨fun e _ => h1 e, h2⟩

@[simp] theorem finished_none (s : State) : finished none s = true := rfl

theorem hasEntry_iff (mt : Motif) (o : Nat) :
    hasEntry mt o = true ↔ ∃ b, b < mt.size ∧ o ≤ nth mt.free b := by
  simp [hasEntry, List.any_eq_true]

end MotifDecoder
