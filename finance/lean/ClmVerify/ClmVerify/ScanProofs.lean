import ClmVerify.Scan
/-!
# C7 proofs: scan algebra, chunked = sequential, causality, adjoint

All statements are over an arbitrary commutative semiring `α`
(`Lean.Grind.CommSemiring`), so they hold for `Int`, `Rat`, `Nat` and any other
instance.  Floating point is not a semiring (addition is not associative); see
REPORT.md.
-/
namespace ClmVerify
open Lean.Grind

theorem Aff.ext_iff' {α : Type} {p q : Aff α} : p = q ↔ p.a = q.a ∧ p.u = q.u := by
  cases p; cases q; simp

section
variable {α : Type} [CommSemiring α]

/-! ## (a) the combine is a monoid -/

theorem Aff.comb_assoc (p q r : Aff α) : (p.comb q).comb r = p.comb (q.comb r) := by
  simp only [Aff.ext_iff', Aff.comb]; constructor <;> grind

theorem Aff.one_comb (p : Aff α) : Aff.one.comb p = p := by
  simp only [Aff.ext_iff', Aff.comb, Aff.one]; constructor <;> grind

theorem Aff.comb_one (p : Aff α) : p.comb Aff.one = p := by
  simp only [Aff.ext_iff', Aff.comb, Aff.one]; constructor <;> grind

/-- `comb` is composition of the affine maps (apply `p` first). -/
theorem Aff.apply_comb (p q : Aff α) (h : α) : (p.comb q).apply h = q.apply (p.apply h) := by
  simp only [Aff.apply, Aff.comb]; grind

theorem Aff.one_apply (h : α) : (Aff.one : Aff α).apply h = h := by
  simp only [Aff.apply, Aff.one]; grind

/-! ## (b) folding the combine over a prefix gives `h_t` -/

theorem final_nil (h : α) : final h [] = h := rfl

theorem final_cons (h : α) (p : Aff α) (ps : List (Aff α)) :
    final h (p :: ps) = final (p.apply h) ps := rfl

theorem final_append (h : α) (xs ys : List (Aff α)) :
    final h (xs ++ ys) = final (final h xs) ys := by
  simp [final, List.foldl_append]

theorem final_eq_comb_fold_aux (r : Aff α) (h : α) (ps : List (Aff α)) :
    final (r.apply h) ps = (ps.foldl Aff.comb r).apply h := by
  induction ps generalizing r with
  | nil => rfl
  | cons p ps ih =>
    rw [final_cons, List.foldl_cons, ← ih, Aff.apply_comb]

/-- The final state is the folded combine applied to the initial state. -/
theorem final_eq_comb_fold (h : α) (ps : List (Aff α)) :
    final h ps = (ps.foldl Aff.comb Aff.one).apply h := by
  rw [← final_eq_comb_fold_aux, Aff.one_apply]

theorem length_seqScan (h : α) (ps : List (Aff α)) : (seqScan h ps).length = ps.length := by
  induction ps generalizing h with
  | nil => rfl
  | cons p ps ih => simp [seqScan, ih]

theorem seqScan_getElem?_final (h : α) (ps : List (Aff α)) (t : Nat) :
    (seqScan h ps)[t]? = if t < ps.length then some (final h (ps.take (t + 1))) else none := by
  induction ps generalizing h t with
  | nil => simp [seqScan]
  | cons p ps ih =>
    cases t with
    | zero => simp [seqScan, final]
    | succ t =>
      simp only [seqScan, List.getElem?_cons_succ, ih, List.length_cons, List.take_succ_cons,
        final_cons]
      split <;> split <;> first | rfl | omega

/-- **(b)** `h_t` is the combine folded over the prefix `p_0 … p_t`, applied to `h0`. -/
theorem seqScan_getElem? (h0 : α) (ps : List (Aff α)) (t : Nat) (ht : t < ps.length) :
    (seqScan h0 ps)[t]? = some (((ps.take (t + 1)).foldl Aff.comb Aff.one).apply h0) := by
  rw [seqScan_getElem?_final, final_eq_comb_fold]; simp [ht]

theorem seqScan_append (h : α) (xs ys : List (Aff α)) :
    seqScan h (xs ++ ys) = seqScan h xs ++ seqScan (final h xs) ys := by
  induction xs generalizing h with
  | nil => rfl
  | cons p xs ih => simp [seqScan, ih, final_cons]

theorem seqScan_getLastD (h : α) (ps : List (Aff α)) : (seqScan h ps).getLastD h = final h ps := by
  induction ps generalizing h with
  | nil => rfl
  | cons p ps ih =>
    cases ps with
    | nil => rfl
    | cons q qs =>
      have : (seqScan h (p :: q :: qs)) = p.apply h :: seqScan (p.apply h) (q :: qs) := rfl
      rw [this, final_cons, ← ih]
      simp [seqScan, List.getLastD]

/-- The final state is affine in the initial state, with slope the product of the `a`s. -/
theorem final_affine (c : List (Aff α)) (h : α) : final h c = prodA c * h + final 0 c := by
  induction c generalizing h with
  | nil => simp only [final_nil, prodA]; grind
  | cons p ps ih =>
    rw [final_cons, final_cons, ih (p.apply h), ih (p.apply 0)]
    simp only [prodA, Aff.apply]; grind

/-- Phase 1 computes the chunk's composed map: its summary applied to `h` is the
chunk's final state from `h`. -/
theorem summarize_apply (c : List (Aff α)) (h : α) : (summarize c).apply h = final h c := by
  simp only [summarize, Aff.apply, seqScan_getLastD]
  exact (final_affine c h).symm

theorem fold_comb_a (c : List (Aff α)) (r : Aff α) : (c.foldl Aff.comb r).a = prodA c * r.a := by
  induction c generalizing r with
  | nil => simp only [List.foldl_nil, prodA]; grind
  | cons p ps ih => simp only [List.foldl_cons, ih, Aff.comb, prodA]; grind

/-- …and the summary is the combine folded over the chunk (the monoid product). -/
theorem summarize_eq_fold (c : List (Aff α)) : summarize c = c.foldl Aff.comb Aff.one := by
  have ha : (c.foldl Aff.comb Aff.one).a = prodA c := by
    rw [fold_comb_a]; simp only [Aff.one]; grind
  have hu := final_eq_comb_fold (0 : α) c
  simp only [Aff.apply, Semiring.mul_zero] at hu
  rw [Aff.ext_iff']
  refine ⟨ha.symm, ?_⟩
  simp only [summarize, seqScan_getLastD, hu]
  rw [Semiring.add_comm, Semiring.add_zero]

/-! ## (c) chunked scan = sequential scan -/

/-- Phases 2 and 3 give the sequential scan for **any** split of time into chunks. -/
theorem chunked_eq_seq_of_split (cs : List (List (Aff α))) (h : α) :
    (List.zipWith (fun c h => seqScan h c) cs (carries h (cs.map summarize))).flatten
      = seqScan h cs.flatten := by
  induction cs generalizing h with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.map_cons, carries, List.zipWith_cons_cons, List.flatten_cons,
      summarize_apply, ih, List.flatten_cons, seqScan_append]

theorem chunksAux_flatten {β : Type} (L : Nat) (hL : 0 < L) :
    ∀ (n : Nat) (xs : List β), xs.length ≤ n → (chunksAux L n xs).flatten = xs := by
  intro n
  induction n with
  | zero => intro xs hx; cases xs with
    | nil => rfl
    | cons _ _ => simp at hx
  | succ n ih =>
    intro xs hx
    cases xs with
    | nil => rfl
    | cons x xs =>
      simp only [chunksAux, List.flatten_cons]
      rw [ih]
      · exact List.take_append_drop L (x :: xs)
      · simp only [List.length_drop, List.length_cons] at hx ⊢; omega

/-- Chunking and concatenating gives the input back (`L ≥ 1`). -/
theorem chunksOf_flatten {β : Type} (L : Nat) (hL : 0 < L) (xs : List β) :
    (chunksOf L xs).flatten = xs :=
  chunksAux_flatten L hL _ xs (Nat.le_refl _)

theorem chunksAux_shape {β : Type} (L : Nat) (hL : 0 < L) :
    ∀ (n : Nat) (xs : List β), ∀ c ∈ chunksAux L n xs, 0 < c.length ∧ c.length ≤ L := by
  intro n
  induction n with
  | zero => intro xs c hc; simp [chunksAux] at hc
  | succ n ih =>
    intro xs c hc
    cases xs with
    | nil => simp [chunksAux] at hc
    | cons x xs =>
      simp only [chunksAux, List.mem_cons] at hc
      rcases hc with rfl | hc
      · simp only [List.length_take, List.length_cons]; omega
      · exact ih _ c hc

/-- Every chunk has between `1` and `L` steps. -/
theorem chunksOf_shape {β : Type} (L : Nat) (hL : 0 < L) (xs : List β) :
    ∀ c ∈ chunksOf L xs, 0 < c.length ∧ c.length ≤ L :=
  chunksAux_shape L hL _ xs

/-- **(c)** The three-phase chunked scan equals the sequential scan, for every `L ≥ 1`. -/
theorem chunkedScan_eq_seqScan (L : Nat) (hL : 0 < L) (h0 : α) (ps : List (Aff α)) :
    chunkedScan L h0 ps = seqScan h0 ps := by
  simp only [chunkedScan]
  rw [chunked_eq_seq_of_split, chunksOf_flatten L hL]

/-! ## (d) causality -/

theorem seqScan_take (h : α) (ps : List (Aff α)) (n : Nat) :
    seqScan h (ps.take n) = (seqScan h ps).take n := by
  induction ps generalizing h n with
  | nil => simp [seqScan]
  | cons p ps ih =>
    cases n with
    | zero => simp [seqScan]
    | succ n => simp [seqScan, ih]

theorem ys_take (d h0 : α) (ins : List (Inp α)) (n : Nat) :
    ys d h0 (ins.take n) = (ys d h0 ins).take n := by
  induction ins generalizing h0 n with
  | nil => simp [ys, seqScan]
  | cons i is ih =>
    cases n with
    | zero => simp [ys, seqScan]
    | succ n =>
      have := ih (i.aff.apply h0) n
      simp only [ys, List.map_take] at this ⊢
      simp [seqScan, this]

theorem take_eq_of_agree {β : Type} (l1 l2 : List β) (t : Nat)
    (hag : ∀ i, i ≤ t → l1[i]? = l2[i]?) : l1.take (t + 1) = l2.take (t + 1) := by
  apply List.ext_getElem?
  intro i
  simp only [List.getElem?_take]
  split
  · exact hag i (by omega)
  · rfl

theorem getElem?_of_take_eq {β : Type} (l1 l2 : List β) (t i : Nat) (hi : i ≤ t)
    (h : l1.take (t + 1) = l2.take (t + 1)) : l1[i]? = l2[i]? := by
  have h1 : (l1.take (t + 1))[i]? = l1[i]? := by simp [List.getElem?_take]; omega
  have h2 : (l2.take (t + 1))[i]? = l2[i]? := by simp [List.getElem?_take]; omega
  rw [← h1, ← h2, h]

/-- **(d)** Causality of the states: inputs that agree up to `t` give states that agree up to `t`. -/
theorem seqScan_causal (h0 : α) (ps qs : List (Aff α)) (t : Nat)
    (hag : ∀ i, i ≤ t → ps[i]? = qs[i]?) :
    ∀ i, i ≤ t → (seqScan h0 ps)[i]? = (seqScan h0 qs)[i]? := by
  intro i hi
  apply getElem?_of_take_eq _ _ t i hi
  rw [← seqScan_take, ← seqScan_take, take_eq_of_agree ps qs t hag]

/-- **(d)** Causality of the readout `y_t = c_t h_t + d x_t`: `y[0..t]` is unchanged by any
change of the inputs after `t`. -/
theorem ys_causal (d h0 : α) (ins1 ins2 : List (Inp α)) (t : Nat)
    (hag : ∀ i, i ≤ t → ins1[i]? = ins2[i]?) :
    ∀ i, i ≤ t → (ys d h0 ins1)[i]? = (ys d h0 ins2)[i]? := by
  intro i hi
  apply getElem?_of_take_eq _ _ t i hi
  rw [← ys_take, ← ys_take, take_eq_of_agree ins1 ins2 t hag]

/-- The chunked kernel is causal too (it equals the sequential scan). -/
theorem chunkedScan_causal (L : Nat) (hL : 0 < L) (h0 : α) (ps qs : List (Aff α)) (t : Nat)
    (hag : ∀ i, i ≤ t → ps[i]? = qs[i]?) :
    ∀ i, i ≤ t → (chunkedScan L h0 ps)[i]? = (chunkedScan L h0 qs)[i]? := by
  rw [chunkedScan_eq_seqScan L hL, chunkedScan_eq_seqScan L hL]
  exact seqScan_causal h0 ps qs t hag

/-! ## (e) reverse scan, chunked reverse scan, adjoint identity -/

theorem revScan_headD (g : α) (ps : List (Aff α)) : (revScan g ps).headD g = revFinal g ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    show p.apply ((revScan g ps).headD g) = revFinal g (p :: ps)
    rw [ih]; rfl

theorem revFinal_cons (g : α) (p : Aff α) (ps : List (Aff α)) :
    revFinal g (p :: ps) = p.apply (revFinal g ps) := rfl

theorem revFinal_append (g : α) (xs ys : List (Aff α)) :
    revFinal g (xs ++ ys) = revFinal (revFinal g ys) xs := by
  simp [revFinal, List.foldr_append]

theorem revScan_cons (g : α) (p : Aff α) (ps : List (Aff α)) :
    revScan g (p :: ps) = p.apply (revFinal g ps) :: revScan g ps := by
  simp only [revScan, revScan_headD]

theorem revScan_append (g : α) (xs ys : List (Aff α)) :
    revScan g (xs ++ ys) = revScan (revFinal g ys) xs ++ revScan g ys := by
  induction xs with
  | nil => rfl
  | cons p xs ih =>
    rw [List.cons_append, revScan_cons, revScan_cons, ih, revFinal_append, List.cons_append]

/-- The reverse scan is the forward scan of the reversed list, reversed. -/
theorem revScan_eq_reverse (g : α) (ps : List (Aff α)) :
    revScan g ps = (seqScan g ps.reverse).reverse := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    rw [revScan_cons, List.reverse_cons, seqScan_append, ih, List.reverse_append]
    have : final g ps.reverse = revFinal g ps := by
      simp [final, revFinal, List.foldl_reverse]
    simp [seqScan, this]

theorem revFinal_affine (c : List (Aff α)) (g : α) : revFinal g c = prodA c * g + revFinal 0 c := by
  induction c with
  | nil => simp only [revFinal, List.foldr_nil, prodA]; grind
  | cons p ps ih =>
    rw [revFinal_cons, revFinal_cons, ih]
    simp only [prodA, Aff.apply]; grind

theorem revSummarize_apply (c : List (Aff α)) (g : α) :
    (revSummarize c).apply g = revFinal g c := by
  simp only [revSummarize, Aff.apply, revScan_headD]
  exact (revFinal_affine c g).symm

/-- Phases 2 and 3 of the reverse kernel give the reverse scan for any split into chunks. -/
theorem chunkedRev_eq_of_split (cs : List (List (Aff α))) (g : α) :
    (revCarries g (cs.map revSummarize)).1 = revFinal g cs.flatten ∧
    (List.zipWith (fun c g => revScan g c) cs (revCarries g (cs.map revSummarize)).2).flatten
      = revScan g cs.flatten := by
  induction cs with
  | nil => exact ⟨rfl, rfl⟩
  | cons c cs ih =>
    obtain ⟨ih1, ih2⟩ := ih
    simp only [List.map_cons, revCarries, List.zipWith_cons_cons, List.flatten_cons]
    rw [ih1, ih2, revSummarize_apply, revFinal_append, revScan_append]
    exact ⟨rfl, rfl⟩

/-- **(e)** The chunked reverse scan equals the sequential reverse scan, for every `L ≥ 1`. -/
theorem chunkedRevScan_eq_revScan (L : Nat) (hL : 0 < L) (g : α) (ps : List (Aff α)) :
    chunkedRevScan L g ps = revScan g ps := by
  simp only [chunkedRevScan]
  rw [(chunkedRev_eq_of_split _ g).2, chunksOf_flatten L hL]

theorem adjoint_cons (x : Step α) (rest : List (Step α)) :
    adjoint (x :: rest) = (x.c * x.gy + nextA rest * (adjoint rest).headD 0) :: adjoint rest := by
  simp only [adjoint, adjPairs, revScan, Aff.apply]
  congr 1; grind

theorem nextA_cons (x : Step α) (rest : List (Step α)) : nextA (x :: rest) = x.a := rfl

/-- The adjoint satisfies its defining recurrence `g_t = c_t gy_t + a_{t+1} g_{t+1}`, `g_T = 0`
(stated at every index). -/
theorem adjoint_getElem? (xs : List (Step α)) (t : Nat) (ht : t < xs.length) :
    (adjoint xs)[t]? = some ((xs[t]'ht).c * (xs[t]'ht).gy +
      nextA (xs.drop (t + 1)) * (adjoint (xs.drop (t + 1))).headD 0) := by
  induction xs generalizing t with
  | nil => simp at ht
  | cons x rest ih =>
    rw [adjoint_cons]
    cases t with
    | zero => simp
    | succ t =>
      simp only [List.getElem?_cons_succ, List.getElem_cons_succ, List.drop_succ_cons]
      exact ih t (by simp at ht; omega)

theorem adjoint_length (xs : List (Step α)) : (adjoint xs).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons x rest ih => simp [adjoint_cons, ih]

/-- **(e)** The adjoint identity
`Σ_t gy_t c_t h_t = Σ_t g_t u_t + g_0 a_0 h0`: `g` is the exact gradient of the
readout loss with respect to `u` (and `g_0 a_0` with respect to `h0`). -/
theorem adjoint_identity (h0 : α) (xs : List (Step α)) :
    sumL (List.zipWith (fun x h => x.gy * x.c * h) xs (stepStates h0 xs))
      = sumL (List.zipWith (fun g x => g * x.u) (adjoint xs) xs)
        + (adjoint xs).headD 0 * nextA xs * h0 := by
  induction xs generalizing h0 with
  | nil => simp only [stepStates, List.map_nil, seqScan, List.zipWith_nil_left, sumL, adjoint,
      adjPairs, revScan, List.headD_nil, nextA]; grind
  | cons x rest ih =>
    have hs : stepStates h0 (x :: rest) = (x.a * h0 + x.u) :: stepStates (x.a * h0 + x.u) rest := by
      simp [stepStates, seqScan, Step.aff, Aff.apply]
    rw [hs, adjoint_cons]
    simp only [List.zipWith_cons_cons, sumL, List.headD_cons, nextA_cons]
    rw [ih (x.a * h0 + x.u)]
    generalize sumL (List.zipWith (fun g x => g * x.u) (adjoint rest) rest) = S
    generalize (adjoint rest).headD 0 = G
    generalize nextA rest = n
    grind

/-- The adjoint computed by the chunked reverse kernel. -/
theorem chunked_adjoint (L : Nat) (hL : 0 < L) (xs : List (Step α)) :
    chunkedRevScan L 0 (adjPairs xs) = adjoint xs :=
  chunkedRevScan_eq_revScan L hL 0 _

theorem adjPairs_indep_u (xs ys : List (Step α))
    (h : xs.map (fun x => (x.a, x.c, x.gy)) = ys.map (fun x => (x.a, x.c, x.gy))) :
    adjPairs xs = adjPairs ys := by
  induction xs generalizing ys with
  | nil => cases ys with
    | nil => rfl
    | cons _ _ => simp at h
  | cons x rest ih =>
    cases ys with
    | nil => simp at h
    | cons y ys =>
      simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at h
      obtain ⟨⟨ha, hc, hg⟩, hrest⟩ := h
      have hn : nextA rest = nextA ys := by
        cases rest with
        | nil => cases ys with
          | nil => rfl
          | cons _ _ => simp at hrest
        | cons r rs => cases ys with
          | nil => simp at hrest
          | cons s ss =>
            simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at hrest
            exact hrest.1.1
      simp only [adjPairs, ih ys hrest, hn, hc, hg]

/-- The adjoint does not read `u` (it only needs `a`, `c`, `gy`). -/
theorem adjoint_indep_u (xs ys : List (Step α))
    (h : xs.map (fun x => (x.a, x.c, x.gy)) = ys.map (fun x => (x.a, x.c, x.gy))) :
    adjoint xs = adjoint ys := by
  simp only [adjoint, adjPairs_indep_u xs ys h]

end
end ClmVerify
