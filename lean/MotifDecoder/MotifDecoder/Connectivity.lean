import MotifDecoder.Invariant
/-!
Item 6: connectivity.
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

/-- Atoms `i` and `j` are joined by a bond (in either direction). -/
def Adj (bs : List Bond) (i j : Nat) : Prop := ∃ o, (i, j, o) ∈ bs ∨ (j, i, o) ∈ bs

/-- Reachability through bonds: the reflexive-transitive closure of `Adj`. -/
inductive Conn (bs : List Bond) : Nat → Nat → Prop
  | refl (i : Nat) : Conn bs i i
  | tail {i j k : Nat} : Conn bs i j → Adj bs j k → Conn bs i k

theorem Adj.symm {bs : List Bond} {i j : Nat} (h : Adj bs i j) : Adj bs j i := by
  obtain ⟨o, h⟩ := h; exact ⟨o, h.symm⟩

theorem Adj.mono {bs bs' : List Bond} {i j : Nat} (h : Adj bs i j) (hsub : ∀ b ∈ bs, b ∈ bs') :
    Adj bs' i j := by
  obtain ⟨o, h | h⟩ := h
  · exact ⟨o, .inl (hsub _ h)⟩
  · exact ⟨o, .inr (hsub _ h)⟩

theorem Conn.trans {bs : List Bond} {i j k : Nat} (h1 : Conn bs i j) (h2 : Conn bs j k) :
    Conn bs i k := by
  induction h2 with
  | refl => exact h1
  | tail _ ha ih => exact ih.tail ha

theorem Conn.single {bs : List Bond} {i j : Nat} (h : Adj bs i j) : Conn bs i j :=
  (Conn.refl i).tail h

theorem Conn.symm {bs : List Bond} {i j : Nat} (h : Conn bs i j) : Conn bs j i := by
  induction h with
  | refl => exact .refl _
  | tail _ ha ih => exact (Conn.single ha.symm).trans ih

theorem Conn.mono {bs bs' : List Bond} {i j : Nat} (h : Conn bs i j)
    (hsub : ∀ b ∈ bs, b ∈ bs') : Conn bs' i j := by
  induction h with
  | refl => exact .refl _
  | tail _ ha ih => exact ih.tail (ha.mono hsub)

theorem Adj.shift {bs : List Bond} {i j : Nat} (h : Adj bs i j) (n : Nat) :
    Adj (bs.map (shiftBond n)) (n + i) (n + j) := by
  obtain ⟨o, h | h⟩ := h
  · exact ⟨o, .inl (List.mem_map.2 ⟨_, h, rfl⟩)⟩
  · exact ⟨o, .inr (List.mem_map.2 ⟨_, h, rfl⟩)⟩

theorem Conn.shift {bs : List Bond} {i j : Nat} (h : Conn bs i j) (n : Nat) :
    Conn (bs.map (shiftBond n)) (n + i) (n + j) := by
  induction h with
  | refl => exact .refl _
  | tail _ ha ih => exact ih.tail (ha.shift n)

/-- A motif is connected: every atom is reachable from its atom 0 through the
motif's bonds. -/
def Motif.Connected (m : Motif) : Prop := ∀ i, i < m.size → Conn m.bonds 0 i

/-- Every motif of the vocabulary is connected. -/
def VocabConnected (V : Vocab) : Prop := ∀ m, m < V.size → (motifAt V m).Connected

/-- **Item 6.**  If every motif of the vocabulary is connected, then in every
reachable state every atom is connected to atom 0.  (In `start` there is no
atom; past `start` there is at least one, see `ReachBy.natoms_pos`.) -/
theorem ReachBy.connected (hC : VocabConnected V) {ts : List Token} {s : State}
    (h : ReachBy V B ts s) : ∀ i, i < s.natoms → Conn s.bonds 0 i := by
  induction h with
  | init => intro i hi; simp [State.init, State.natoms] at hi
  | @snoc ts s s' t hr hs ih =>
    have hinv := hr.inv
    cases hs with
    | pop => exact ih
    | atom => exact ih
    | bond => exact ih
    | motif => exact ih
    | @start m hp hm hb =>
      have h0 := hinv.start_eq hp
      subst h0
      intro i hi
      rw [natoms_startMotif] at hi
      have hi' : i < (motifAt V m).size := by simpa [State.init, State.natoms] using hi
      have := (hC m hm i hi').shift 0
      simp only [Nat.zero_add] at this
      exact this.mono (fun b hb => by simp [startMotif, State.init, State.natoms] at hb ⊢; exact hb)
    | @attach a o m b hp hb hf =>
      obtain ⟨_, _, _, _, _, hm, _, _⟩ := hinv.afterMotif hp
      have hsrc := hinv.src_lt hp
      intro i hi
      rw [natoms_attach] at hi
      have hold : ∀ x ∈ s.bonds, x ∈ (attach V s a o m b).bonds := by
        intro x hx; simp [attach, hx]
      have hnew : ∀ x ∈ (motifAt V m).bonds.map (shiftBond s.natoms),
          x ∈ (attach V s a o m b).bonds := by
        intro x hx; simp only [attach, List.mem_append]; exact .inl (.inr hx)
      by_cases hlt : i < s.natoms
      · exact (ih i hlt).mono hold
      · obtain ⟨j, rfl⟩ : ∃ j, i = s.natoms + j := ⟨i - s.natoms, by omega⟩
        have hj : j < (motifAt V m).size := by omega
        -- 0 ~ src (old bonds), src - dst (the attachment bond), dst ~ i (inside the motif)
        have c1 : Conn (attach V s a o m b).bonds 0 (s.top.1 + a) := (ih _ hsrc).mono hold
        have c2 : Adj (attach V s a o m b).bonds (s.top.1 + a) (s.natoms + b) :=
          ⟨o, .inl (by simp [attach])⟩
        have c3 : Conn (motifAt V m).bonds b j := (hC m hm b hb).symm.trans (hC m hm j hj)
        exact (c1.tail c2).trans ((c3.shift s.natoms).mono hnew)

/-- Past `start` there is at least one atom (for a well-formed vocabulary), so
"connected to atom 0" is not vacuous. -/
theorem ReachBy.natoms_pos (hV : VocabWF V) {ts : List Token} {s : State}
    (h : ReachBy V B ts s) (hne : s.phase ≠ .start) : 0 < s.natoms := by
  induction h with
  | init => exact absurd rfl hne
  | @snoc ts s s' t hr hs ih =>
    cases hs with
    | start hp hm hb => rw [natoms_startMotif]; have := (hV _ hm).pos; omega
    | pop hp hb => exact ih (by simp [hp])
    | atom hp ha hf => exact ih (by simp [hp])
    | bond hp h1 h3 hf => exact ih (by simp [hp])
    | motif hp hm he hb => exact ih (by simp [hp])
    | attach hp hb hf =>
      rw [natoms_attach]; have := ih (by simp [hp]); omega

/-- Item 6 as a statement about pairs: any two atoms are connected. -/
theorem ReachBy.connected_pair (hC : VocabConnected V) {ts : List Token} {s : State}
    (h : ReachBy V B ts s) {i j : Nat} (hi : i < s.natoms) (hj : j < s.natoms) :
    Conn s.bonds i j :=
  (h.connected hC i hi).symm.trans (h.connected hC j hj)

end MotifDecoder
