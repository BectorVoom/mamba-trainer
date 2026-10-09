import MotifDecoder.Invariant
/-!
Items 2, 3 and 4: valence accounting, counting, composition (and the budget).
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

theorem popFrame_phase (s : State) :
    (popFrame s).phase = .done ∨ (popFrame s).phase = .body := by
  simp only [popFrame]; split
  · exact .inl rfl
  · exact .inr rfl

/-! ## Item 2: valence accounting -/

section Exact
variable {ts : List Token} {s : State} {a o m b : Nat}

/-- What an allowed entry atom means. -/
theorem allowed_entry (hp : s.phase = .afterMotif a o m)
    (hal : allowed V B s (.atom b) = true) :
    b < (motifAt V m).size ∧ o ≤ nth (motifAt V m).free b := by
  simpa [allowed, hp] using hal

theorem apply_entry (hp : s.phase = .afterMotif a o m) :
    apply V s (.atom b) = attach V s a o m b := by
  simp [apply, hp]

private theorem attach_facts (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (b : Nat) :
    s.top.1 + a < s.free.length ∧ s.natoms = s.free.length ∧ s.top.1 + a ≠ s.natoms + b ∧
      o ≤ nth s.free (s.top.1 + a) := by
  have hi := h.inv
  have hwf := h.wf hV
  have hsrc := hi.src_lt hp
  have hlen : s.natoms = s.free.length := hwf.len
  obtain ⟨_, _, _, _, hf, _⟩ := hi.afterMotif hp
  exact ⟨by omega, hlen, by omega, hf⟩

/-- **Item 2, exactness at the source atom.**  When the entry atom `b` is
allowed, the source atom's `free` drops by exactly `o` (the truncating
subtraction does not truncate). -/
theorem nth_attach_src (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (hal : allowed V B s (.atom b) = true) :
    nth (apply V s (.atom b)).free (s.top.1 + a) + o = nth s.free (s.top.1 + a) := by
  obtain ⟨hb, _⟩ := allowed_entry hp hal
  obtain ⟨hsrc, _, hne, hf⟩ := attach_facts hV h hp b
  rw [apply_entry hp]
  simp only [attach]
  rw [nth_subAt_ne _ _ hne.symm, nth_subAt_same, nth_append_left hsrc]
  omega

/-- **Item 2, exactness at the entry atom.**  The new motif's atom `b` starts
at the motif's `free[b]` and drops by exactly `o`. -/
theorem nth_attach_dst (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (hal : allowed V B s (.atom b) = true) :
    nth (apply V s (.atom b)).free (s.natoms + b) + o = nth (motifAt V m).free b := by
  obtain ⟨hb, hfb⟩ := allowed_entry hp hal
  obtain ⟨_, hlen, hne, _⟩ := attach_facts hV h hp b
  rw [apply_entry hp]
  simp only [attach]
  rw [nth_subAt_same, nth_subAt_ne _ _ hne, hlen, nth_append_right]
  omega

/-- Every other atom keeps its value (old atoms their `free`, the new motif's
atoms the motif's `free`). -/
theorem nth_attach_other (hp : s.phase = .afterMotif a o m) {j : Nat}
    (h1 : j ≠ s.top.1 + a) (h2 : j ≠ s.natoms + b) :
    nth (apply V s (.atom b)).free j = nth (s.free ++ (motifAt V m).free) j := by
  rw [apply_entry hp]
  simp only [attach]
  rw [nth_subAt_ne _ _ (Ne.symm h2), nth_subAt_ne _ _ (Ne.symm h1)]

private theorem attach_free_sum (hV : VocabWF V) (h : ReachBy V B ts s)
    (hp : s.phase = .afterMotif a o m) (hfb : o ≤ nth (motifAt V m).free b) :
    (attach V s a o m b).free.sum + 2 * o = s.free.sum + (motifAt V m).free.sum := by
  obtain ⟨hsrc, hlen, hne, hf⟩ := attach_facts hV h hp b
  simp only [attach]
  have e1 : o ≤ nth (s.free ++ (motifAt V m).free) (s.top.1 + a) := by
    rw [nth_append_left hsrc]; exact hf
  have e2 : o ≤ nth (subAt (s.free ++ (motifAt V m).free) (s.top.1 + a) o) (s.natoms + b) := by
    rw [nth_subAt_ne _ _ hne, hlen, nth_append_right]; exact hfb
  have s1 := sum_subAt e1
  have s2 := sum_subAt e2
  rw [List.sum_append] at s1
  omega

end Exact

/-- **Item 2, valence accounting.**  In every reachable state
`Σ free + 2 · (Σ orders of the attachment bonds) = Σ_{added motifs} Σ motif.free`. -/
theorem ReachBy.valence (hV : VocabWF V) {ts : List Token} {s : State}
    (h : ReachBy V B ts s) :
    s.free.sum + 2 * s.attachOrder = (s.added.map (fun m => (motifAt V m).free.sum)).sum := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb =>
      simp only [startMotif, List.sum_append, List.map_append, List.map_cons, List.map_nil,
        List.sum_cons, List.sum_nil]
      omega
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | @attach a o m b hp hb hf =>
      have := attach_free_sum hV hr hp hf
      have e : (attach V s a o m b).attachOrder = s.attachOrder + o := rfl
      have e' : (attach V s a o m b).added = s.added ++ [m] := rfl
      rw [e, e']
      simp only [List.sum_append, List.map_append, List.map_cons, List.map_nil,
        List.sum_cons, List.sum_nil]
      omega

/-- Sum of the orders of a bond list. -/
def orderSum (bs : List Bond) : Nat := (bs.map (fun b => b.2.2)).sum

theorem orderSum_append (xs ys : List Bond) : orderSum (xs ++ ys) = orderSum xs + orderSum ys := by
  simp [orderSum, List.map_append, List.sum_append]

theorem orderSum_shift (n : Nat) (bs : List Bond) :
    orderSum (bs.map (shiftBond n)) = orderSum bs := by
  induction bs with
  | nil => rfl
  | cons b rest ih =>
    simp only [orderSum, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]; rfl

/-- The ghost counter `attachOrder` is what it claims to be: the total order of
the state's bonds minus the orders of the motifs' own bonds. -/
theorem ReachBy.orderSum_bonds {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    orderSum s.bonds =
      s.attachOrder + (s.added.map (fun m => orderSum (motifAt V m).bonds)).sum := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb =>
      simp only [startMotif, orderSum_append, orderSum_shift, List.sum_append, List.map_append,
        List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
      omega
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | @attach a o m b hp hb hf =>
      simp only [attach, orderSum_append, orderSum_shift, List.sum_append, List.map_append,
        List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
      have : orderSum [(s.top.1 + a, s.natoms + b, o)] = o := by simp [orderSum]
      omega

/-- Item 2 without the ghost counter: total valence is conserved,
`Σ free + 2 · Σ (all bond orders) = Σ_{motifs} (Σ motif.free + 2 · Σ motif bond orders)`. -/
theorem ReachBy.valence_total (hV : VocabWF V) {ts : List Token} {s : State}
    (h : ReachBy V B ts s) :
    s.free.sum + 2 * orderSum s.bonds =
      (s.added.map (fun m => (motifAt V m).free.sum + 2 * orderSum (motifAt V m).bonds)).sum := by
  have h1 := h.valence hV
  have h2 := h.orderSum_bonds
  have key : ∀ l : List Nat,
      (l.map (fun m => (motifAt V m).free.sum + 2 * orderSum (motifAt V m).bonds)).sum =
        (l.map (fun m => (motifAt V m).free.sum)).sum +
          2 * (l.map (fun m => orderSum (motifAt V m).bonds)).sum := by
    intro l
    induction l with
    | nil => rfl
    | cons x xs ih => simp only [List.map_cons, List.sum_cons, ih]; omega
  rw [key]; omega

/-- The original `free` of every atom, in global numbering. -/
def origFree (V : Vocab) (s : State) : List Nat := s.added.flatMap (fun m => (motifAt V m).free)

/-- **Item 2, per-atom form.**  `free` is aligned with the motifs' original
values and never exceeds them. -/
theorem ReachBy.free_le_orig {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    s.free.length = (origFree V s).length ∧ ∀ i, nth s.free i ≤ nth (origFree V s) i := by
  induction h with
  | init => exact ⟨rfl, fun _ => Nat.le_refl _⟩
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | @start m hp hm hb =>
      have e : origFree V (startMotif V s m) = origFree V s ++ (motifAt V m).free := by
        simp [origFree, startMotif, List.flatMap_append]
      rw [e]
      refine ⟨by simp [startMotif, ih.1], ?_⟩
      exact nth_append_le _ ih.1 ih.2
    | @attach a o m b hp hb hf =>
      have e : origFree V (attach V s a o m b) = origFree V s ++ (motifAt V m).free := by
        simp [origFree, attach, List.flatMap_append]
      rw [e]
      refine ⟨by simp [attach, ih.1], fun i => ?_⟩
      simp only [attach]
      exact Nat.le_trans (nth_subAt_le _ _ _ _)
        (Nat.le_trans (nth_subAt_le _ _ _ _) (nth_append_le _ ih.1 ih.2 i))

/-! ## Item 3: counting -/

/-- **Item 3.** Number of atoms = sum of the sizes of the added motifs. -/
theorem ReachBy.natoms_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    s.elements.length = (s.added.map (fun m => (motifAt V m).size)).sum := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | start hp hm hb =>
      simp only [startMotif, List.length_append, List.sum_append, List.map_append, List.map_cons,
        List.map_nil, List.sum_cons, List.sum_nil, Motif.size]
      simp only [Motif.size] at ih; omega
    | attach hp hb hf =>
      simp only [attach, List.length_append, List.sum_append, List.map_append, List.map_cons,
        List.map_nil, List.sum_cons, List.sum_nil, Motif.size]
      simp only [Motif.size] at ih; omega

theorem ReachBy.nbonds_aux {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    s.bonds.length + 1 = (s.added.map (fun m => (motifAt V m).bonds.length)).sum +
      s.added.length + (if s.phase = .start then 1 else 0) := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb =>
      simp only [hp, ite_true] at ih
      simp only [startMotif, List.length_append, List.length_map, List.sum_append,
        List.map_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
        List.length_cons, List.length_nil]
      simp; omega
    | pop hp hb =>
      simp only [hp] at ih
      rcases popFrame_phase s with h' | h' <;> simp only [h'] <;> simpa [popFrame] using ih
    | atom hp ha hf => simp only [hp] at ih; simpa [State.setPhase] using ih
    | bond hp h1 h3 hf => simp only [hp] at ih; simpa [State.setPhase] using ih
    | motif hp hm he hb => simp only [hp] at ih; simpa [State.setPhase] using ih
    | attach hp hb hf =>
      simp only [hp] at ih
      simp only [attach, List.length_append, List.length_map, List.sum_append,
        List.map_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
        List.length_cons, List.length_nil]
      simp at ih ⊢; omega

/-- **Item 3.** Past `start`, number of bonds = sum of the motifs' bond counts
+ (number of motifs − 1), written without subtraction. -/
theorem ReachBy.nbonds_eq {ts : List Token} {s : State} (h : ReachBy V B ts s)
    (hne : s.phase ≠ .start) :
    s.bonds.length + 1 =
      (s.added.map (fun m => (motifAt V m).bonds.length)).sum + s.added.length := by
  have := h.nbonds_aux; simpa [hne] using this

/-- Past `start` at least one motif has been added. -/
theorem ReachBy.added_pos {ts : List Token} {s : State} (h : ReachBy V B ts s)
    (hne : s.phase ≠ .start) : 0 < s.added.length := by
  induction h with
  | init => exact absurd rfl hne
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb => simp [startMotif]
    | pop hp hb => exact ih (by simp [hp])
    | atom hp ha hf => exact ih (by simp [hp])
    | bond hp h1 h3 hf => exact ih (by simp [hp])
    | motif hp hm he hb => exact ih (by simp [hp])
    | attach hp hb hf => simp [attach]

/-- Item 3 in the literal form `bonds = Σ motif bonds + (k − 1)`. -/
theorem ReachBy.nbonds_eq' {ts : List Token} {s : State} (h : ReachBy V B ts s)
    (hne : s.phase ≠ .start) :
    s.bonds.length =
      (s.added.map (fun m => (motifAt V m).bonds.length)).sum + (s.added.length - 1) := by
  have := h.nbonds_eq hne; have := h.added_pos hne; omega

theorem sum_excess_int (f g : Nat → Nat) (l : List Nat) :
    (l.map (fun m => (f m : Int) + 1 - (g m : Int))).sum =
      ((l.map f).sum : Int) + (l.length : Int) - ((l.map g).sum : Int) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, List.sum_cons, List.length_cons, ih]; omega

theorem sum_excess_nat (f g : Nat → Nat) (l : List Nat) (h : ∀ m ∈ l, g m ≤ f m + 1) :
    (l.map (fun m => f m + 1 - g m)).sum + (l.map g).sum = (l.map f).sum + l.length := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    have hx := h x (List.mem_cons_self ..)
    have := ih (fun m hm => h m (List.mem_cons_of_mem _ hm))
    simp only [List.map_cons, List.sum_cons, List.length_cons]; omega

/-- **Item 3, corollary over the integers (no side condition).**
`bonds + 1 − atoms = Σ_{motifs} (motif.bonds + 1 − motif.size)`: the assembly
adds no cycle of its own. -/
theorem ReachBy.cyclomatic_int {ts : List Token} {s : State} (h : ReachBy V B ts s)
    (hne : s.phase ≠ .start) :
    (s.bonds.length : Int) + 1 - (s.elements.length : Int) =
      (s.added.map (fun m =>
        ((motifAt V m).bonds.length : Int) + 1 - ((motifAt V m).size : Int))).sum := by
  have h1 := h.nbonds_eq hne
  have h2 := h.natoms_eq
  rw [sum_excess_int (fun m => (motifAt V m).bonds.length) (fun m => (motifAt V m).size)]
  omega

/-- **Item 3, corollary over the naturals**, under the side condition that
makes each truncating subtraction exact: every added motif has
`size ≤ bonds + 1` (true of any connected motif). -/
theorem ReachBy.cyclomatic {ts : List Token} {s : State} (h : ReachBy V B ts s)
    (hne : s.phase ≠ .start)
    (hside : ∀ m ∈ s.added, (motifAt V m).size ≤ (motifAt V m).bonds.length + 1) :
    s.bonds.length + 1 - s.elements.length =
      (s.added.map (fun m => (motifAt V m).bonds.length + 1 - (motifAt V m).size)).sum := by
  have h1 := h.nbonds_eq hne
  have h2 := h.natoms_eq
  have h3 := sum_excess_nat (fun m => (motifAt V m).bonds.length) (fun m => (motifAt V m).size)
    s.added hside
  omega

/-! ## Item 4: composition -/

/-- **Item 4.** The element counts of the state are the sums of the element
counts of the added motifs. -/
theorem ReachBy.count_eq {ts : List Token} {s : State} (h : ReachBy V B ts s) (e : Nat) :
    s.elements.count e = (s.added.map (fun m => (motifAt V m).elements.count e)).sum := by
  induction h with
  | init => rfl
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | start hp hm hb =>
      simp only [startMotif, List.count_append, List.sum_append, List.map_append, List.map_cons,
        List.map_nil, List.sum_cons, List.sum_nil]
      omega
    | attach hp hb hf =>
      simp only [attach, List.count_append, List.sum_append, List.map_append, List.map_cons,
        List.map_nil, List.sum_cons, List.sum_nil]
      omega

/-- With a budget no element ever exceeds its target (so the mask's test over
the motif's own elements is the "for every element" test, `fits_some_iff`). -/
theorem ReachBy.within_budget {b : Budget} {ts : List Token} {s : State}
    (h : ReachBy V (some b) ts s) (e : Nat) : s.elements.count e ≤ b.count e :=
  h.inv.budget b rfl e

/-- On reachable states the budget test of `motif m` is exactly the
specification's: for every element, motif count + state count ≤ target. -/
theorem ReachBy.fits_iff {b : Budget} {ts : List Token} {s : State}
    (h : ReachBy V (some b) ts s) (mt : Motif) :
    fits (some b) s.elements mt = true ↔
      ∀ e, mt.elements.count e + s.elements.count e ≤ b.count e :=
  fits_some_iff b s.elements mt h.within_budget

/-- **Item 4, budget.**  An accepted sequence ends in a state whose element
counts equal the target and whose `Σ free` equals `H`. -/
theorem AcceptedAs.budget {b : Budget} {ts : List Token} {s : State}
    (h : AcceptedAs V (some b) ts s) :
    (∀ e, s.elements.count e = b.count e) ∧ s.free.sum = b.hydrogens :=
  (finished_some_iff b s).1 (h.reachBy.inv.done h.2).2

/-- Item 4 for the motifs themselves: the added motifs' element counts sum to
the target, and their free valences minus twice the attachment orders is `H`. -/
theorem AcceptedAs.budget_motifs (hV : VocabWF V) {b : Budget} {ts : List Token} {s : State}
    (h : AcceptedAs V (some b) ts s) :
    (∀ e, (s.added.map (fun m => (motifAt V m).elements.count e)).sum = b.count e) ∧
      b.hydrogens + 2 * s.attachOrder = (s.added.map (fun m => (motifAt V m).free.sum)).sum := by
  obtain ⟨h1, h2⟩ := h.budget
  refine ⟨fun e => ?_, ?_⟩
  · rw [← h.reachBy.count_eq e]; exact h1 e
  · rw [← h2]; exact h.reachBy.valence hV

end MotifDecoder
