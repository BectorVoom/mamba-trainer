import MotifDecoder.Step
/-!
Item 1: the well-formedness invariant.

`Inv` needs no hypothesis on the vocabulary (it records what the guards of the
pending phases established, and that the frames lie inside the atoms).
`State.WF` is the invariant of the specification and needs `VocabWF`.
-/
namespace MotifDecoder

variable {V : Vocab} {B : Option Budget}

/-- What the guards have established in each phase. -/
def PhaseInv (V : Vocab) (B : Option Budget) (s : State) : Prop :=
  match s.phase with
  | .start => s = State.init
  | .body => s.stack ≠ []
  | .afterAtom a => s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ nth s.free (s.top.1 + a)
  | .afterBond a o =>
      s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ o ∧ o ≤ 3 ∧ o ≤ nth s.free (s.top.1 + a)
  | .afterMotif a o m =>
      s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ o ∧ o ≤ 3 ∧ o ≤ nth s.free (s.top.1 + a) ∧
        m < V.size ∧ fits B s.elements (motifAt V m) = true ∧
        ∃ b, b < (motifAt V m).size ∧ o ≤ nth (motifAt V m).free b
  | .done => s.stack = [] ∧ finished B s = true

structure Inv (V : Vocab) (B : Option Budget) (s : State) : Prop where
  phase : PhaseInv V B s
  /-- every stack frame lies inside the atoms -/
  frames : ∀ f ∈ s.stack, f.1 + f.2 ≤ s.natoms
  /-- with a budget, no element ever exceeds its target -/
  budget : ∀ b, B = some b → ∀ e, s.elements.count e ≤ b.count e

theorem stack_ne_of_lt_top {s : State} {a : Nat} (h : a < s.top.2) : s.stack ≠ [] := by
  intro hs; simp [State.top, hs] at h

theorem top_mem {s : State} (h : s.stack ≠ []) : s.top ∈ s.stack := by
  cases hs : s.stack with
  | nil => exact absurd hs h
  | cons f rest => simp [State.top, hs]

@[simp] theorem natoms_startMotif (s : State) (m : Nat) :
    (startMotif V s m).natoms = s.natoms + (motifAt V m).size := by
  simp [startMotif, State.natoms, Motif.size]

@[simp] theorem natoms_attach (s : State) (a o m b : Nat) :
    (attach V s a o m b).natoms = s.natoms + (motifAt V m).size := by
  simp [attach, State.natoms, Motif.size]

theorem fits_count {els : List Nat} {mt : Motif} (hf : fits B els mt = true)
    (h : ∀ b, B = some b → ∀ e, els.count e ≤ b.count e) :
    ∀ b, B = some b → ∀ e, (els ++ mt.elements).count e ≤ b.count e := by
  intro b hB e
  subst hB
  have := (fits_some_iff b els mt (h b rfl)).1 hf e
  rw [List.count_append]; omega

theorem Inv.init : Inv V B State.init :=
  ⟨rfl, by simp [State.init], by simp [State.init]⟩

theorem Inv.start_eq {s : State} (hi : Inv V B s) (hp : s.phase = .start) : s = State.init := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

theorem Inv.body_ne {s : State} (hi : Inv V B s) (hp : s.phase = .body) : s.stack ≠ [] := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

theorem Inv.afterMotif {s : State} {a o m : Nat} (hi : Inv V B s)
    (hp : s.phase = .afterMotif a o m) :
    s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ o ∧ o ≤ 3 ∧ o ≤ nth s.free (s.top.1 + a) ∧
      m < V.size ∧ fits B s.elements (motifAt V m) = true ∧
      ∃ b, b < (motifAt V m).size ∧ o ≤ nth (motifAt V m).free b := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

theorem Inv.afterAtom {s : State} {a : Nat} (hi : Inv V B s) (hp : s.phase = .afterAtom a) :
    s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ nth s.free (s.top.1 + a) := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

theorem Inv.afterBond {s : State} {a o : Nat} (hi : Inv V B s) (hp : s.phase = .afterBond a o) :
    s.stack ≠ [] ∧ a < s.top.2 ∧ 1 ≤ o ∧ o ≤ 3 ∧ o ≤ nth s.free (s.top.1 + a) := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

theorem Inv.done {s : State} (hi : Inv V B s) (hp : s.phase = .done) :
    s.stack = [] ∧ finished B s = true := by
  have := hi.phase; simp only [PhaseInv, hp] at this; exact this

/-- The source atom of a pending attachment is an existing atom. -/
theorem Inv.src_lt {s : State} {a o m : Nat} (hi : Inv V B s)
    (hp : s.phase = .afterMotif a o m) : s.top.1 + a < s.natoms := by
  obtain ⟨hne, ha, _⟩ := hi.afterMotif hp
  have := hi.frames _ (top_mem hne)
  omega

theorem Inv.step {s s' : State} {t : Token} (hi : Inv V B s) (h : Step V B s t s') :
    Inv V B s' := by
  cases h with
  | start hp hm hb =>
    refine ⟨?_, ?_, ?_⟩
    · show (startMotif V s _).stack ≠ []
      simp [startMotif]
    · intro f hf
      rw [natoms_startMotif]
      simp only [startMotif, List.mem_cons] at hf
      rcases hf with rfl | hf
      · exact Nat.le_refl _
      · have := hi.frames f hf; omega
    · exact fits_count hb hi.budget
  | pop hp hb =>
    refine ⟨?_, fun f hf => hi.frames f (List.mem_of_mem_tail hf), hi.budget⟩
    cases hs : s.stack.tail with
    | nil =>
      have : (popFrame s).phase = .done := by simp [popFrame, hs]
      simp only [PhaseInv, this]
      exact ⟨hs, hb hs⟩
    | cons f rest =>
      have : (popFrame s).phase = .body := by simp [popFrame, hs]
      simp only [PhaseInv, this]
      show s.stack.tail ≠ []
      simp [hs]
  | atom hp ha hf =>
    exact ⟨show _ ∧ _ ∧ _ from ⟨stack_ne_of_lt_top ha, ha, hf⟩, hi.frames, hi.budget⟩
  | @bond a o hp h1 h3 hf =>
    have := hi.phase; simp only [PhaseInv, hp] at this
    exact ⟨show _ ∧ _ ∧ _ ∧ _ ∧ _ from ⟨this.1, this.2.1, h1, h3, hf⟩, hi.frames, hi.budget⟩
  | @motif a o m hp hm he hb =>
    have := hi.phase; simp only [PhaseInv, hp] at this
    exact ⟨show _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ from
      ⟨this.1, this.2.1, this.2.2.1, this.2.2.2.1, this.2.2.2.2, hm, hb, he⟩, hi.frames, hi.budget⟩
  | @attach a o m b hp hb hf =>
    obtain ⟨_, _, _, _, _, _, hfit, _⟩ := hi.afterMotif hp
    refine ⟨?_, ?_, ?_⟩
    · show (attach V s a o m b).stack ≠ []
      simp [attach]
    · intro f hf
      rw [natoms_attach]
      simp only [attach, List.mem_cons] at hf
      rcases hf with rfl | hf
      · exact Nat.le_refl _
      · have := hi.frames f hf; omega
    · exact fits_count hfit hi.budget

theorem ReachBy.inv {ts : List Token} {s : State} (h : ReachBy V B ts s) : Inv V B s := by
  induction h with
  | init => exact Inv.init
  | snoc _ hs ih => exact ih.step hs

/-- The well-formedness invariant of the specification (item 1), plus: every
bond has order 1..3. -/
structure State.WF (s : State) : Prop where
  len : s.elements.length = s.free.length
  frames : ∀ f ∈ s.stack, f.1 + f.2 ≤ s.elements.length
  bonds : ∀ b ∈ s.bonds, b.1 < s.elements.length ∧ b.2.1 < s.elements.length ∧ b.1 ≠ b.2.1 ∧
    1 ≤ b.2.2 ∧ b.2.2 ≤ 3

theorem vocabWF_at (hV : VocabWF V) {m : Nat} (hm : m < V.size) : (motifAt V m).WF := hV m hm

/-- **Item 1.** Every reachable state is well-formed. -/
theorem ReachBy.wf (hV : VocabWF V) {ts : List Token} {s : State} (h : ReachBy V B ts s) :
    s.WF := by
  induction h with
  | init => exact ⟨rfl, by simp [State.init], by simp [State.init]⟩
  | @snoc ts s s' t hr hs ih =>
    have hi := hr.inv
    have hi' : Inv V B s' := hi.step hs
    refine ⟨?_, hi'.frames, ?_⟩
    · cases hs with
      | start hp hm hb =>
        simp only [startMotif, List.length_append]
        rw [ih.len, (hV _ hm).free_len]; rfl
      | pop => exact ih.len
      | atom => exact ih.len
      | bond => exact ih.len
      | motif => exact ih.len
      | attach hp hb hf =>
        obtain ⟨_, _, _, _, _, hm, _, _⟩ := hi.afterMotif hp
        simp only [attach, List.length_append, length_subAt]
        rw [ih.len, (hV _ hm).free_len]; rfl
    · cases hs with
      | pop => exact ih.bonds
      | atom => exact ih.bonds
      | bond => exact ih.bonds
      | motif => exact ih.bonds
      | @start m hp hm hb =>
        intro b hb
        have hn : (startMotif V s m).elements.length = s.natoms + (motifAt V m).size :=
          natoms_startMotif s m
        rw [hn]
        simp only [startMotif, List.mem_append, List.mem_map] at hb
        rcases hb with hb | ⟨b0, hb0, rfl⟩
        · exact BondOK.mono (ih.bonds b hb) (Nat.le_add_right _ _)
        · exact BondOK.shift ((hV _ hm).bonds_ok b0 hb0) _
      | @attach a o m b hp hb hf =>
        obtain ⟨_, _, h1, h3, _, hm, _, _⟩ := hi.afterMotif hp
        have hsrc := hi.src_lt hp
        intro x hx
        have hn : (attach V s a o m b).elements.length = s.natoms + (motifAt V m).size :=
          natoms_attach s a o m b
        rw [hn]
        simp only [attach, List.mem_append, List.mem_map, List.mem_singleton] at hx
        rcases hx with (hx | ⟨b0, hb0, rfl⟩) | rfl
        · exact BondOK.mono (ih.bonds x hx) (Nat.le_add_right _ _)
        · exact BondOK.shift ((hV _ hm).bonds_ok b0 hb0) _
        · exact ⟨by show s.top.1 + a < _; omega, by show s.natoms + b < _; omega,
            by show s.top.1 + a ≠ s.natoms + b; omega, h1, h3⟩

end MotifDecoder
