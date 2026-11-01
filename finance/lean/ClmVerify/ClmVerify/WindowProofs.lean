import ClmVerify.Windows
/-!
# C2, C3 proofs: no feature index reaches a label index; the class rule is a partition
-/
namespace ClmVerify

theorem mem_windowRows {W t r : Nat} (hW : W ≤ t + 1) :
    r ∈ windowRows W t ↔ t + 1 - W ≤ r ∧ r ≤ t := by
  simp only [windowRows, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨i, hi, rfl⟩; omega
  · intro hr; exact ⟨r - (t + 1 - W), by omega, by omega⟩

theorem mem_rowReads {K r i : Nat} (hK : K ≤ r) : i ∈ rowReads K r ↔ r - K ≤ i ∧ i ≤ r := by
  simp only [rowReads, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨j, hj, rfl⟩; omega
  · intro hi; exact ⟨i - (r - K), by omega, by omega⟩

/-- **C2** Every day the input window of decision `t` reads lies in `[t-W+1-K, t]`
(precondition `t ≥ W-1+K`, i.e. the window and its lookback start at day `≥ 0`), and
every such day is read: the window reads exactly that interval. -/
theorem mem_featureReads {K W t i : Nat} (hW : 1 ≤ W) (hv : W - 1 + K ≤ t) :
    i ∈ featureReads K W t ↔ t + 1 - W - K ≤ i ∧ i ≤ t := by
  simp only [featureReads, List.mem_flatMap]
  constructor
  · rintro ⟨r, hr, hi⟩
    rw [mem_windowRows (by omega)] at hr
    rw [mem_rowReads (by omega)] at hi
    omega
  · intro hi
    -- the row `max (t+1-W) i` reads `i`, and the row `t+1-W+K`… pick `r = min t (i + K)`
    refine ⟨min t (i + K), ?_, ?_⟩
    · rw [mem_windowRows (by omega)]; omega
    · rw [mem_rowReads (by omega)]; omega

/-- **C2** No feature of decision `t` reads a day after `t`. -/
theorem featureReads_le {K W t i : Nat} (hW : 1 ≤ W) (hv : W - 1 + K ≤ t)
    (hi : i ∈ featureReads K W t) : i ≤ t :=
  ((mem_featureReads hW hv).1 hi).2

theorem mem_labelDays {h t i : Nat} : i ∈ labelDays h t ↔ t + 1 ≤ i ∧ i ≤ t + 1 + h := by
  simp only [labelDays, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨j, hj, rfl⟩; omega
  · intro hi; exact ⟨i - (t + 1), by omega, by omega⟩

/-- **C3** Every price the label reads lies in the label window `[t+1, t+1+h]`. -/
theorem labelPriceReads_sub {h t i : Nat} (hi : i ∈ labelPriceReads h t) :
    i ∈ labelDays h t := by
  rw [mem_labelDays]
  simp only [labelPriceReads, List.mem_append, List.mem_cons, List.mem_map, List.mem_range,
    List.not_mem_nil, or_false] at hi
  rcases hi with (rfl | rfl) | ⟨j, hj, rfl⟩ <;> omega

/-- **C3** `label_start(t) = t+1`, `label_end(t) = t+1+h` are the ends of the label window. -/
theorem labelDays_bounds {h t i : Nat} (hi : i ∈ labelDays h t) :
    labelStart t ≤ i ∧ i ≤ labelEnd h t := by
  simpa [labelStart, labelEnd] using (mem_labelDays.1 hi)

theorem labelStart_mem (h t : Nat) : labelStart t ∈ labelDays h t := by
  rw [mem_labelDays]; simp [labelStart]

theorem labelEnd_mem (h t : Nat) : labelEnd h t ∈ labelDays h t := by
  rw [mem_labelDays]; simp [labelEnd]

/-- **C2 + C3** Every day a feature of the window reads is strictly before every day the
label reads: `max feature index ≤ t < t + 1 ≤ min label index`. -/
theorem feature_lt_label {K W h t i j : Nat} (hW : 1 ≤ W) (hv : W - 1 + K ≤ t)
    (hi : i ∈ featureReads K W t) (hj : j ∈ labelDays h t) : i < j := by
  have := featureReads_le hW hv hi
  have := (mem_labelDays.1 hj).1
  omega

/-! ## Label classes -/

section
open Lean.Grind
variable {α : Type} [CommRing α] [LE α] [LT α] [Std.IsLinearOrder α] [Std.LawfulOrderLT α]
  [OrderedRing α] [DecidableLT α]

omit [DecidableLT α] in
/-- `θ ≥ 0` whenever `kappa, m, sigma ≥ 0`. -/
theorem theta_nonneg {kappa m sigma : α} (hk : 0 ≤ kappa) (hm : 0 ≤ m) (hs : 0 ≤ sigma) :
    0 ≤ theta kappa m sigma := by
  have hms : 0 ≤ m * sigma := OrderedRing.mul_nonneg hm hs
  unfold theta; grind

omit [DecidableLT α] in
/-- With `θ ≥ 0` the BUY and SELL conditions exclude each other. -/
theorem not_buy_and_sell {θ r : α} (hθ : 0 ≤ θ) : ¬ (θ < r ∧ r < -θ) := by
  rintro ⟨h1, h2⟩; grind

omit [LE α] [Std.IsLinearOrder α] [Std.LawfulOrderLT α] [OrderedRing α] in
theorem classify_buy_iff {θ r : α} : classify θ r = .buy ↔ θ < r := by
  unfold classify; split <;> (try split) <;> simp_all

theorem classify_sell_iff {θ r : α} (hθ : 0 ≤ θ) : classify θ r = .sell ↔ r < -θ := by
  unfold classify
  split
  · simp only [reduceCtorEq, false_iff]; intro h; exact not_buy_and_sell hθ ⟨by assumption, h⟩
  · split <;> simp_all

omit [LE α] [Std.IsLinearOrder α] [Std.LawfulOrderLT α] [OrderedRing α] in
theorem classify_hold_iff {θ r : α} : classify θ r = .hold ↔ ¬ θ < r ∧ ¬ r < -θ := by
  unfold classify; split <;> (try split) <;> simp_all

/-- **C3** For `θ ≥ 0` exactly one of `r > θ` (BUY), `r < -θ` (SELL), neither (HOLD) holds,
and `classify` returns that class. -/
theorem classify_exactly_one {θ r : α} (hθ : 0 ≤ θ) :
    (classify θ r = .buy ∧ θ < r ∧ ¬ r < -θ) ∨
    (classify θ r = .sell ∧ r < -θ ∧ ¬ θ < r) ∨
    (classify θ r = .hold ∧ ¬ θ < r ∧ ¬ r < -θ) := by
  have hx : ¬ (θ < r ∧ r < -θ) := not_buy_and_sell hθ
  by_cases h1 : θ < r
  · left; exact ⟨classify_buy_iff.2 h1, h1, fun h2 => hx ⟨h1, h2⟩⟩
  · by_cases h2 : r < -θ
    · right; left; exact ⟨(classify_sell_iff hθ).2 h2, h2, h1⟩
    · right; right; exact ⟨classify_hold_iff.2 ⟨h1, h2⟩, h1, h2⟩

end
end ClmVerify
