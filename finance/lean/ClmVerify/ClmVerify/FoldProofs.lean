import ClmVerify.Folds
import ClmVerify.WindowProofs
/-!
# C4 proofs: fold layout, purging, test segments, expanding scheme, kept prefixes
-/
namespace ClmVerify

/-! ## `firstIdx` -/

theorem firstIdx_some {p : Int → Bool} {l : List Int} {i : Nat} (h : firstIdx p l = some i) :
    i < l.length ∧ p (yr l i) = true ∧ ∀ j, j < i → p (yr l j) = false := by
  induction l generalizing i with
  | nil => simp [firstIdx] at h
  | cons y ys ih =>
    simp only [firstIdx] at h
    split at h
    · rename_i hp
      cases h
      exact ⟨by simp, by simpa [yr] using hp, fun j hj => by omega⟩
    · rename_i hp
      obtain ⟨k, hk, rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := ih hk
      refine ⟨by simp; omega, by simpa [yr] using h2, ?_⟩
      intro j hj
      cases j with
      | zero => simpa [yr] using hp
      | succ j => simpa [yr] using h3 j (by omega)

theorem firstIdx_none {p : Int → Bool} {l : List Int} (h : firstIdx p l = none) :
    ∀ j, j < l.length → p (yr l j) = false := by
  induction l with
  | nil => intro j hj; simp at hj
  | cons y ys ih =>
    simp only [firstIdx] at h
    split at h
    · cases h
    · rename_i hp
      have hn := ih (by simpa using h)
      intro j hj
      cases j with
      | zero => simpa [yr] using hp
      | succ j => simpa [yr] using hn j (by simp at hj; omega)

theorem firstIdx_le {p : Int → Bool} {l : List Int} {i j : Nat} (h : firstIdx p l = some i)
    (_hj : j < l.length) (hp : p (yr l j) = true) : i ≤ j := by
  obtain ⟨_, _, h3⟩ := firstIdx_some h
  refine Nat.le_of_not_lt fun hlt => ?_
  rw [h3 j hlt] at hp; cases hp

theorem testStart_spec {years : List Int} {Y : Int} {ts : Nat} (h : testStart years Y = some ts) :
    ts < years.length ∧ yr years ts = Y ∧ ∀ j, j < years.length → yr years j = Y → ts ≤ j := by
  obtain ⟨h1, h2, _⟩ := firstIdx_some h
  refine ⟨h1, by simpa using h2, fun j hj hy => firstIdx_le h hj (by simpa using hy)⟩

theorem testEnd_le (years : List Int) (Y : Int) : testEnd years Y ≤ years.length := by
  unfold testEnd
  cases h : firstIdx (fun y => decide (Y < y)) years with
  | none => simp
  | some te => simpa using Nat.le_of_lt (firstIdx_some h).1

/-- Days before `te` have `year ≤ Y`. -/
theorem lt_testEnd {years : List Int} {Y : Int} {j : Nat} (hj : j < testEnd years Y)
    (hN : j < years.length) : yr years j ≤ Y := by
  unfold testEnd at hj
  cases h : firstIdx (fun y => decide (Y < y)) years with
  | none => have := firstIdx_none h j hN; simp at this; omega
  | some te =>
    rw [h] at hj
    have := (firstIdx_some h).2.2 j (by simpa using hj)
    simp at this; omega

/-- Every day with `year > Y` is at or after `te`. -/
theorem testEnd_le_of_gt {years : List Int} {Y : Int} {j : Nat} (hN : j < years.length)
    (hy : Y < yr years j) : testEnd years Y ≤ j := by
  unfold testEnd
  cases h : firstIdx (fun y => decide (Y < y)) years with
  | none => have := firstIdx_none h j hN; simp at this; omega
  | some te => simpa using firstIdx_le h hN (by simpa using hy)

/-- `te < N` implies `year[te] > Y`. -/
theorem testEnd_gt {years : List Int} {Y : Int} (h : testEnd years Y < years.length) :
    Y < yr years (testEnd years Y) := by
  unfold testEnd at h ⊢
  cases hf : firstIdx (fun y => decide (Y < y)) years with
  | none => rw [hf] at h; simp at h
  | some te => simpa using (firstIdx_some hf).2.1

/-! ## What an emitted fold looks like -/

@[simp] theorem Fold.seg_zero (f : Fold) : f.seg 0 = f.train := rfl
@[simp] theorem Fold.seg_one (f : Fold) : f.seg 1 = f.val := rfl
@[simp] theorem Fold.seg_two (f : Fold) : f.seg 2 = f.cal := rfl
@[simp] theorem Fold.seg_three (f : Fold) : f.seg 3 = f.test := rfl

theorem mkFold_some {cfg : FoldConfig} {years : List Int} {Y : Int} {f : Fold}
    (h : mkFold cfg years Y = some f) :
    ∃ ts, testStart years Y = some ts ∧ cfg.calDays + cfg.valDays ≤ ts ∧
      cfg.minTrain ≤ f.train.len ∧ f.year = Y ∧
      f.cal = ⟨ts - cfg.calDays, ts⟩ ∧
      f.val = ⟨ts - cfg.calDays - cfg.valDays, ts - cfg.calDays⟩ ∧
      f.test = ⟨ts, testEnd years Y⟩ ∧
      f.train.stop = ts - cfg.calDays - cfg.valDays ∧
      (cfg.scheme = .expanding → f.train.start = 0) ∧
      (∀ d, cfg.scheme = .rolling d → f.train.start = ts - cfg.calDays - cfg.valDays - d) := by
  unfold mkFold at h
  split at h
  · cases h
  · rename_i ts hts
    cases hsch : cfg.scheme with
    | expanding =>
      simp only [hsch] at h
      split at h
      · rename_i hc
        cases h
        exact ⟨ts, hts, hc.1, hc.2, rfl, rfl, rfl, rfl, rfl, fun _ => rfl, (fun d hd => by cases hd)⟩
      · cases h
    | rolling d =>
      simp only [hsch] at h
      split at h
      · rename_i hc
        cases h
        exact ⟨ts, hts, hc.1, hc.2, rfl, rfl, rfl, rfl, rfl, (fun hd => by cases hd),
          (fun d' hd => by cases hd; rfl)⟩
      · cases h

theorem mem_folds {cfg : FoldConfig} {years : List Int} {f : Fold} :
    f ∈ folds cfg years ↔ ∃ Y, Y ∈ testYears cfg ∧ mkFold cfg years Y = some f := by
  simp [folds, List.mem_filterMap]

/-- The test year of an emitted fold is in `first_test_year ..= last_test_year` and occurs
in the calendar; emitted folds satisfy the emission conditions. -/
theorem fold_emitted {cfg : FoldConfig} {years : List Int} {f : Fold} (hf : f ∈ folds cfg years) :
    cfg.firstTest ≤ f.year ∧ f.year ≤ cfg.lastTest ∧
    (∃ t, t < years.length ∧ yr years t = f.year) ∧
    cfg.calDays + cfg.valDays ≤ f.test.start ∧ cfg.minTrain ≤ f.train.len := by
  obtain ⟨Y, hY, hm⟩ := mem_folds.1 hf
  obtain ⟨ts, hts, hc, hmin, hy, -, -, ht, -⟩ := mkFold_some hm
  have := testStart_spec hts
  simp only [testYears, List.mem_map, List.mem_range] at hY
  obtain ⟨i, hi, rfl⟩ := hY
  refine ⟨by omega, by omega, ⟨ts, this.1, by rw [hy]; exact this.2.1⟩, by rw [ht]; exact hc, hmin⟩

/-- **C4.1** Layout of an emitted fold (years sorted): `train`, `val`, `cal`, `test` are
adjacent intervals in this order inside `[0, N)`, `cal` and `val` have the configured
lengths, `test` is non-empty. -/
theorem fold_layout {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) :
    f.train.start ≤ f.train.stop ∧ f.train.stop = f.val.start ∧
    f.val.start ≤ f.val.stop ∧ f.val.stop = f.cal.start ∧
    f.cal.start ≤ f.cal.stop ∧ f.cal.stop = f.test.start ∧
    f.test.start < f.test.stop ∧ f.test.stop ≤ years.length ∧
    f.val.len = cfg.valDays ∧ f.cal.len = cfg.calDays := by
  obtain ⟨Y, -, hm⟩ := mem_folds.1 hf
  obtain ⟨ts, hts, hc, -, -, hcal, hval, htest, htr, hexp, hrol⟩ := mkFold_some hm
  obtain ⟨hN, hy, -⟩ := testStart_spec hts
  have hte := testEnd_le years Y
  -- `ts < te`: a day `≤ ts` has year `≤ Y`, so it is not the first day with year `> Y`
  have hlt : ts < testEnd years Y := by
    refine Nat.lt_of_not_le fun hle => ?_
    have h1 := testEnd_gt (Y := Y) (years := years) (by omega)
    have h2 := hs _ _ hle hN
    omega
  have htrs : f.train.start ≤ f.train.stop := by
    cases hsch : cfg.scheme with
    | expanding => rw [hexp hsch]; omega
    | rolling d => rw [hrol d hsch, htr]; omega
  rw [hcal, hval, htest, htr]
  simp only [Seg.len]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> first | trivial | omega

/-- The segments' start points never decrease (`val ≤ cal ≤ test`), and each segment
ends where the next begins — without any assumption on the calendar. -/
theorem fold_chain {cfg : FoldConfig} {years : List Int} {f : Fold} (hf : f ∈ folds cfg years) :
    f.train.stop = f.val.start ∧ f.val.stop = f.cal.start ∧ f.cal.stop = f.test.start ∧
    f.val.start ≤ f.cal.start ∧ f.cal.start ≤ f.test.start ∧ f.test.start < years.length := by
  obtain ⟨Y, -, hm⟩ := mem_folds.1 hf
  obtain ⟨ts, hts, -, -, -, hcal, hval, htest, htr, -⟩ := mkFold_some hm
  have := (testStart_spec hts).1
  rw [hcal, hval, htest, htr]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> (try simp only) <;> first | trivial | omega

/-- Segment `i` ends before segment `j > i` starts. -/
theorem seg_stop_le_start {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) {i j : Nat} (hij : i < j) (hj : j ≤ 3) :
    (f.seg i).stop ≤ (f.seg j).start := by
  have L := fold_layout hs hf
  have hi : i = 0 ∨ i = 1 ∨ i = 2 := by omega
  have hj' : j = 1 ∨ j = 2 ∨ j = 3 := by omega
  rcases hi with rfl | rfl | rfl <;> rcases hj' with rfl | rfl | rfl <;>
    first | omega | (simp only [Fold.seg_zero, Fold.seg_one, Fold.seg_two, Fold.seg_three]; omega)

/-- **C4.1** Pairwise disjoint. -/
theorem seg_disjoint {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) {i j t : Nat} (hij : i ≠ j) (hi : i ≤ 3) (hj : j ≤ 3) :
    ¬ ((f.seg i).mem t ∧ (f.seg j).mem t) := by
  rintro ⟨⟨_, h1⟩, ⟨h2, _⟩⟩
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · have := seg_stop_le_start hs hf h hj; omega
  · have := seg_stop_le_start hs hf h hi
    omega

/-- **C4.1** In order: every day of segment `i` is before every day of segment `j > i`. -/
theorem seg_ordered {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) {i j t t' : Nat} (hij : i < j) (hj : j ≤ 3)
    (ht : (f.seg i).mem t) (ht' : (f.seg j).mem t') : t < t' := by
  have := seg_stop_le_start hs hf hij hj
  simp only [Seg.mem] at ht ht'; omega

/-- **C4.1** Inside `[0, N)`. -/
theorem seg_inside {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) {i t : Nat} (hi : i ≤ 3) (ht : (f.seg i).mem t) :
    t < years.length := by
  have L := fold_layout hs hf
  simp only [Seg.mem] at ht
  have hi' : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
  rcases hi' with rfl | rfl | rfl | rfl <;> simp only [Fold.seg_zero, Fold.seg_one, Fold.seg_two, Fold.seg_three] at ht <;> omega

/-- **C4.1** Each segment is contiguous: its days are exactly `start, start+1, …, stop-1`. -/
theorem mem_days {s : Seg} {t : Nat} : t ∈ s.days ↔ s.mem t := by
  simp only [Seg.days, Seg.len, Seg.mem, List.mem_range']
  constructor
  · rintro ⟨i, hi, rfl⟩; omega
  · intro h; exact ⟨t - s.start, by omega, by omega⟩

/-- The test segment is exactly the days of the test year (years sorted). -/
theorem test_eq_year {cfg : FoldConfig} {years : List Int} (hs : Sorted years) {f : Fold}
    (hf : f ∈ folds cfg years) {t : Nat} (hN : t < years.length) :
    f.test.mem t ↔ yr years t = f.year := by
  obtain ⟨Y, -, hm⟩ := mem_folds.1 hf
  obtain ⟨ts, hts, -, -, hy, -, -, htest, -⟩ := mkFold_some hm
  obtain ⟨hts1, hts2, hts3⟩ := testStart_spec hts
  rw [htest, hy]
  simp only [Seg.mem]
  constructor
  · rintro ⟨h1, h2⟩
    have := hs _ _ h1 hN
    have := lt_testEnd h2 hN
    omega
  · intro hyt
    refine ⟨hts3 t hN hyt, Nat.lt_of_not_le fun hle => ?_⟩
    have hgt := testEnd_gt (years := years) (Y := Y) (by omega)
    have := hs _ _ hle hN
    omega

/-- Rolling scheme: the training segment has at most `train_days` days. -/
theorem rolling_train_len {cfg : FoldConfig} {years : List Int} {f : Fold} {d : Nat}
    (hsch : cfg.scheme = .rolling d) (hf : f ∈ folds cfg years) : f.train.len ≤ d := by
  obtain ⟨Y, -, hm⟩ := mem_folds.1 hf
  obtain ⟨ts, -, -, -, -, -, -, -, htr, -, hrol⟩ := mkFold_some hm
  simp only [Seg.len, htr, hrol d hsch]; omega

/-! ## Kept sets -/

theorem filter_range'_lt (x a n : Nat) :
    (List.range' a n).filter (fun t => decide (t < x)) = List.range' a (min (max x a) (a + n) - a) := by
  induction n generalizing a with
  | zero =>
    have : min (max x a) (a + 0) - a = 0 := by omega
    rw [this]; rfl
  | succ n ih =>
    rw [List.range'_succ, List.filter_cons, ih]
    by_cases hax : a < x
    · simp only [hax, decide_true, ↓reduceIte]
      have : min (max x a) (a + (n + 1)) - a = (min (max x (a + 1)) (a + 1 + n) - (a + 1)) + 1 := by
        omega
      rw [this, List.range'_succ]
    · simp only [hax, decide_false]
      have h1 : min (max x (a + 1)) (a + 1 + n) - (a + 1) = 0 := by omega
      have h2 : min (max x a) (a + (n + 1)) - a = 0 := by omega
      rw [h1, h2]; rfl

theorem keepRule_iff (h e s t : Nat) : keepRule h e s t = decide (t < s - 1 - h - e) := by
  simp only [keepRule]; apply decide_eq_decide.2; omega

theorem testKeepRule_iff (N h t : Nat) : testKeepRule N h t = decide (t < N - 1 - h) := by
  simp only [testKeepRule]; apply decide_eq_decide.2; omega

theorem keeps_eq (N h e : Nat) (f : Fold) (i : Nat) :
    f.keeps N h e i = fun t => decide (t < (if i < 3 then (f.seg (i + 1)).start - 1 - h - e
      else N - 1 - h)) := by
  funext t
  simp only [Fold.keeps]
  split
  · exact keepRule_iff _ _ _ _
  · exact testKeepRule_iff _ _ _

/-- **C4 kept prefix** The kept days of segment `i` are the contiguous prefix
`[a, kb)` of the segment, `kb = clamp(s_next - 1 - h - e, a, b)` (test:
`clamp(N - 1 - h, a, b)`), with `Nat` subtraction exactly as written. -/
theorem kept_eq_range (N h e : Nat) (f : Fold) (i : Nat) :
    f.kept N h e i = List.range' (f.seg i).start (f.keepEnd N h e i - (f.seg i).start) := by
  simp only [Fold.kept, Seg.days, keeps_eq, filter_range'_lt]
  congr 1
  simp only [Fold.keepEnd, clamp, Seg.len]
  split <;> omega

theorem mem_kept (N h e : Nat) (f : Fold) (i t : Nat) :
    t ∈ f.kept N h e i ↔ (f.seg i).mem t ∧ f.keeps N h e i t = true := by
  simp [Fold.kept, List.mem_filter, mem_days]

theorem mem_kept_iff_range (N h e : Nat) (f : Fold) (i t : Nat) :
    t ∈ f.kept N h e i ↔ (f.seg i).start ≤ t ∧ t < f.keepEnd N h e i := by
  rw [kept_eq_range, List.mem_range']
  constructor
  · rintro ⟨j, hj, rfl⟩; omega
  · intro hr; exact ⟨t - (f.seg i).start, by omega, by omega⟩

/-- **Leak injection** A day that violates its segment's keep rule is not kept. -/
theorem not_kept_of_violation (N h e : Nat) (f : Fold) (i t : Nat)
    (hv : f.keeps N h e i t = false) : t ∉ f.kept N h e i := by
  rw [mem_kept]; rintro ⟨-, hk⟩; rw [hv] at hk; cases hk

/-- Leak injection, concretely: a non-test sample whose label (plus embargo) reaches the next
segment, `s_next ≤ t + 1 + h + e`, is not kept. -/
theorem not_kept_of_label_reaches_next (N h e : Nat) (f : Fold) (i t : Nat) (hi : i < 3)
    (hv : (f.seg (i + 1)).start ≤ t + 1 + h + e) : t ∉ f.kept N h e i := by
  apply not_kept_of_violation
  simp only [Fold.keeps, hi, ↓reduceIte, keepRule]
  simp; omega

/-- Leak injection for the test segment: a sample whose label ends past `N - 1` is not kept. -/
theorem not_kept_test_of_label_past_end (N h e : Nat) (f : Fold) (t : Nat)
    (hv : N - 1 < t + 1 + h) : t ∉ f.kept N h e 3 := by
  apply not_kept_of_violation
  simp [Fold.keeps, testKeepRule]; omega

/-- **C4.2** Purging and embargo: for a kept sample `t` of segment `i` and any later segment
`S = seg j` of the same fold, every label day `l` of `t` satisfies `l < S.start - e`, i.e.
`l + e < S.start`, and so `l + e < d` for every decision day `d` of `S`. -/
theorem purge {cfg : FoldConfig} {years : List Int} {f : Fold} (hf : f ∈ folds cfg years)
    (N h e : Nat) {i j t l : Nat} (hij : i < j) (hj : j ≤ 3)
    (hk : t ∈ f.kept N h e i) (hl : l ∈ labelDays h t) :
    l + e < (f.seg j).start ∧ l < (f.seg j).start - e ∧
      ∀ d, (f.seg j).mem d → l + e < d := by
  have C := fold_chain hf
  rw [mem_kept] at hk
  have hi3 : i < 3 := by omega
  have hrule := hk.2
  simp only [Fold.keeps, hi3, ↓reduceIte, keepRule, decide_eq_true_eq] at hrule
  have hlab := mem_labelDays.1 hl
  -- the start of the next segment is at most the start of any later one
  have hnext : (f.seg (i + 1)).start ≤ (f.seg j).start := by
    have hi' : i = 0 ∨ i = 1 ∨ i = 2 := by omega
    have hj' : j = 1 ∨ j = 2 ∨ j = 3 := by omega
    rcases hi' with rfl | rfl | rfl <;> rcases hj' with rfl | rfl | rfl <;>
      first | omega | (simp only [Nat.reduceAdd, Fold.seg_one, Fold.seg_two, Fold.seg_three]; omega)
  refine ⟨by omega, by omega, ?_⟩
  intro d hd
  simp only [Seg.mem] at hd
  omega

/-- A kept test sample's label lies inside the calendar (`l ≤ N - 1`). -/
theorem kept_test_label_inside (N h e : Nat) (f : Fold) {t l : Nat}
    (hk : t ∈ f.kept N h e 3) (hl : l ∈ labelDays h t) : l ≤ N - 1 ∧ l < N := by
  rw [mem_kept] at hk
  have hrule := hk.2
  simp only [Fold.keeps, Nat.lt_irrefl, ↓reduceIte, testKeepRule, decide_eq_true_eq] at hrule
  have := mem_labelDays.1 hl
  omega

/-! ## Across folds -/

theorem testYears_pairwise (cfg : FoldConfig) : (testYears cfg).Pairwise (· < ·) := by
  simp only [testYears, List.pairwise_map]
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  simp only [List.getElem_range]
  omega

/-- **C4.3** Test segments of different emitted folds are pairwise disjoint and increasing
(no assumption on the calendar needed). -/
theorem folds_test_increasing (cfg : FoldConfig) (years : List Int) :
    (folds cfg years).Pairwise (fun f g => f.year < g.year ∧ f.test.stop ≤ g.test.start) := by
  unfold folds
  rw [List.pairwise_filterMap]
  apply (testYears_pairwise cfg).imp
  intro Y Y' hYY' f hf g hg
  obtain ⟨ts, -, -, -, hy, -, -, htest, -⟩ := mkFold_some hf
  obtain ⟨ts', hts', -, -, hy', -, -, htest', -⟩ := mkFold_some hg
  obtain ⟨hN', hyr', -⟩ := testStart_spec hts'
  refine ⟨by rw [hy, hy']; exact hYY', ?_⟩
  rw [htest, htest']
  exact testEnd_le_of_gt hN' (by rw [hyr']; exact hYY')

theorem folds_test_disjoint (cfg : FoldConfig) (years : List Int) :
    (folds cfg years).Pairwise (fun f g => ∀ t, ¬ (f.test.mem t ∧ g.test.mem t)) := by
  apply (folds_test_increasing cfg years).imp
  rintro f g ⟨-, h⟩ t ⟨⟨_, h1⟩, ⟨h2, _⟩⟩
  omega

/-- **C4.5** Expanding scheme: an earlier fold's training segment is contained in every later
fold's (in particular `train(k) ⊆ train(k+1)`). -/
theorem expanding_train_mono {cfg : FoldConfig} {years : List Int}
    (hsch : cfg.scheme = .expanding) (hs : Sorted years) :
    (folds cfg years).Pairwise (fun f g => ∀ t, f.train.mem t → g.train.mem t) := by
  unfold folds
  rw [List.pairwise_filterMap]
  apply (testYears_pairwise cfg).imp
  intro Y Y' hYY' f hf g hg t ht
  obtain ⟨ts, hts, -, -, -, -, -, -, htr, hexp, -⟩ := mkFold_some hf
  obtain ⟨ts', hts', -, -, -, -, -, -, htr', hexp', -⟩ := mkFold_some hg
  obtain ⟨hN, hyr, -⟩ := testStart_spec hts
  obtain ⟨hN', hyr', -⟩ := testStart_spec hts'
  have hle : ts ≤ ts' := by
    refine Nat.le_of_not_lt fun hlt => ?_
    have := hs _ _ (Nat.le_of_lt hlt) hN
    omega
  simp only [Seg.mem, hexp hsch, hexp' hsch, htr, htr'] at ht ⊢
  omega

end ClmVerify
