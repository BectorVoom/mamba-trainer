import ClmVerify.Backtest
/-!
# C5 proofs: positions, trades, timing, accounting, skip rule
-/
namespace ClmVerify

-- The core ring classes carry `NatCast` as a field; use it for `trade : Nat` in money.
attribute [local instance] Lean.Grind.Semiring.natCast

/-! ## Positions and trades -/

theorem nextPos_le_one {prev : Nat} (hp : prev ≤ 1) (a : Action) : nextPos prev a ≤ 1 := by
  cases a <;> simp [nextPos] <;> omega

theorem tradeSize_le_one {prev p : Nat} (h1 : prev ≤ 1) (h2 : p ≤ 1) : tradeSize prev p ≤ 1 := by
  simp only [tradeSize]; omega

theorem tradeSize_self (p : Nat) : tradeSize p p = 0 := by simp [tradeSize]

/-- HOLD keeps the position and does not trade (one step). -/
theorem hold_step (prev : Nat) : nextPos prev .hold = prev ∧ tradeSize prev (nextPos prev .hold) = 0 :=
  ⟨rfl, tradeSize_self prev⟩

/-- **C5** `p(t) ∈ {0, 1}` (and `trade(t) ∈ {0, 1}`) for every decision. -/
theorem posTrades_mem {prev : Nat} (hp : prev ≤ 1) (as : List Action) :
    ∀ x ∈ posTrades prev as, (x.1 = 0 ∨ x.1 = 1) ∧ (x.2 = 0 ∨ x.2 = 1) := by
  induction as generalizing prev with
  | nil => simp [posTrades]
  | cons a as ih =>
    intro x hx
    simp only [posTrades, List.mem_cons] at hx
    rcases hx with rfl | hx
    · have h1 := nextPos_le_one hp a
      have h2 := tradeSize_le_one hp h1
      simp only; omega
    · exact ih (nextPos_le_one hp a) x hx

theorem length_posTrades (prev : Nat) (as : List Action) : (posTrades prev as).length = as.length := by
  induction as generalizing prev with
  | nil => rfl
  | cons a as ih => simp [posTrades, ih]

/-- **C5** HOLD never trades: a HOLD decision has `trade = 0`. -/
theorem hold_no_trade (prev : Nat) (as : List Action) (i : Nat) (h : as[i]? = some .hold) :
    ((posTrades prev as)[i]?).map Prod.snd = some 0 := by
  induction as generalizing prev i with
  | nil => simp at h
  | cons a as ih =>
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h; simp [posTrades, tradeSize_self, nextPos]
    | succ i =>
      simp only [List.getElem?_cons_succ] at h
      simp only [posTrades, List.getElem?_cons_succ]
      exact ih _ i h

/-- HOLD keeps the position: on the first decision it stays `prev` (`0` for a segment)… -/
theorem hold_keeps_first (prev : Nat) (a : Action) (as : List Action) (h : a = .hold) :
    ((posTrades prev (a :: as))[0]?).map Prod.fst = some prev := by
  subst h; simp [posTrades, nextPos]

/-- …and later it equals the previous decision's position. -/
theorem hold_keeps (prev : Nat) (as : List Action) (i : Nat) (h : as[i + 1]? = some .hold) :
    ((posTrades prev as)[i + 1]?).map Prod.fst = ((posTrades prev as)[i]?).map Prod.fst := by
  induction as generalizing prev i with
  | nil => simp at h
  | cons a as ih =>
    cases i with
    | zero =>
      cases as with
      | nil => simp at h
      | cons b bs =>
        simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at h
        subst h; simp [posTrades, nextPos]
    | succ i =>
      simp only [List.getElem?_cons_succ] at h
      simp only [posTrades, List.getElem?_cons_succ]
      exact ih _ i h

/-- **C5** Total trades `≤` number of non-HOLD actions. -/
theorem trades_le_nonhold {prev : Nat} (hp : prev ≤ 1) (as : List Action) :
    ((posTrades prev as).map Prod.snd).sum ≤ (as.filter (fun a => a != .hold)).length := by
  induction as generalizing prev with
  | nil => simp [posTrades]
  | cons a as ih =>
    have := ih (nextPos_le_one hp a)
    cases a with
    | hold => simp [posTrades, nextPos, tradeSize_self] at this ⊢; omega
    | buy =>
      have := tradeSize_le_one hp (nextPos_le_one hp .buy)
      simp only [posTrades, List.map_cons, List.sum_cons, List.filter_cons] at *
      simp; omega
    | sell =>
      have := tradeSize_le_one hp (nextPos_le_one hp .sell)
      simp only [posTrades, List.map_cons, List.sum_cons, List.filter_cons] at *
      simp; omega

/-! ## Timing -/

/-- **C5** Execution day `t + 1 > t`. -/
theorem execDay_gt (t : Nat) : t < execDay t := by simp [execDay]

/-- **C5** Every price the P&L of decision `t` reads has index `≥ t + 1` (so `> t`: the
decision cannot see it). -/
theorem pnlReads_ge {t i : Nat} (hi : i ∈ pnlReads t) : t + 1 ≤ i := by
  simp [pnlReads] at hi; omega

section
variable {α : Type} [Lean.Grind.CommRing α]

/-- **C5 skip rule** If `t + 2 > N - 1` the decision is not traded: no trade, no gross,
no cost, position unchanged. -/
theorem runDay_skip {N : Nat} (κ : α) {t : Nat} (prev : Nat) (a : Action) (ret : Option α)
    (h : t + 2 > N - 1) :
    runDay N κ t prev a ret = ⟨prev, 0, 0, 0, 0⟩ := by
  simp [runDay, h]

/-- A traded decision reads only prices inside the calendar: `t + 1 ≤ i ≤ N - 1`. -/
theorem runDay_reads_inside {N t : Nat} (h : ¬ t + 2 > N - 1) {i : Nat} (hi : i ∈ pnlReads t) :
    t + 1 ≤ i ∧ i ≤ N - 1 := by
  simp [pnlReads] at hi; omega

theorem runDay_traded {N : Nat} (κ : α) {t : Nat} (prev : Nat) (a : Action) (ret : Option α)
    (h : ¬ t + 2 > N - 1) :
    (runDay N κ t prev a ret).pos = nextPos prev a ∧
    (runDay N κ t prev a ret).trade = tradeSize prev (nextPos prev a) := by
  simp [runDay, h]

/-- A missing open gives `gross = 0`, but the position is recorded and the cost charged. -/
theorem runDay_missing {N : Nat} (κ : α) {t : Nat} (prev : Nat) (a : Action)
    (h : ¬ t + 2 > N - 1) :
    (runDay N κ t prev a none).gross = 0 ∧
    (runDay N κ t prev a none).pos = nextPos prev a ∧
    (runDay N κ t prev a none).cost = κ * ((tradeSize prev (nextPos prev a) : Nat) : α) := by
  simp [runDay, h]

theorem runDay_hold {N : Nat} (κ : α) (t prev : Nat) (ret : Option α) :
    (runDay N κ t prev .hold ret).pos = prev ∧ (runDay N κ t prev .hold ret).trade = 0 := by
  unfold runDay; split
  · exact ⟨rfl, rfl⟩
  · simp [nextPos, tradeSize_self]

/-- Per day: `net = gross - κ * trade` and `cost = κ * trade`. -/
theorem runDay_net {N : Nat} (κ : α) (t prev : Nat) (a : Action) (ret : Option α) :
    (runDay N κ t prev a ret).net =
      (runDay N κ t prev a ret).gross - κ * ((runDay N κ t prev a ret).trade : α) ∧
    (runDay N κ t prev a ret).cost = κ * ((runDay N κ t prev a ret).trade : α) := by
  unfold runDay; split
  · constructor <;> grind
  · exact ⟨rfl, rfl⟩

theorem runBacktest_cons (N : Nat) (κ : α) (t prev : Nat) (a : Action) (r : Option α)
    (ds : List (Action × Option α)) :
    runBacktest N κ t prev ((a, r) :: ds) =
      runDay N κ t prev a r :: runBacktest N κ (t + 1) (runDay N κ t prev a r).pos ds := rfl

/-- **C5** `p(t) ∈ {0, 1}` and `trade(t) ∈ {0, 1}` on every day of the ledger. -/
theorem runBacktest_mem {N : Nat} (κ : α) (ds : List (Action × Option α)) :
    ∀ t prev, prev ≤ 1 → ∀ r ∈ runBacktest N κ t prev ds,
      (r.pos = 0 ∨ r.pos = 1) ∧ (r.trade = 0 ∨ r.trade = 1) := by
  induction ds with
  | nil => intro t prev _ r hr; simp [runBacktest] at hr
  | cons d ds ih =>
    intro t prev hp r hr
    obtain ⟨a, x⟩ := d
    rw [runBacktest_cons, List.mem_cons] at hr
    have hday : (runDay N κ t prev a x).pos ≤ 1 ∧ (runDay N κ t prev a x).trade ≤ 1 := by
      unfold runDay; split
      · exact ⟨hp, Nat.zero_le _⟩
      · exact ⟨nextPos_le_one hp a, tradeSize_le_one hp (nextPos_le_one hp a)⟩
    rcases hr with rfl | hr
    · omega
    · exact ih _ _ hday.1 r hr

/-- **C5 accounting** `Σ net = Σ gross - κ · Σ trade`. -/
theorem accounting (N : Nat) (κ : α) (ds : List (Action × Option α)) :
    ∀ t prev, sumL ((runBacktest N κ t prev ds).map Row.net)
      = sumL ((runBacktest N κ t prev ds).map Row.gross)
        - κ * sumL ((runBacktest N κ t prev ds).map (fun r => (r.trade : α))) := by
  induction ds with
  | nil => intro t prev; simp only [runBacktest, List.map_nil, sumL]; grind
  | cons d ds ih =>
    intro t prev
    obtain ⟨a, x⟩ := d
    simp only [runBacktest_cons, List.map_cons, sumL]
    rw [ih, (runDay_net κ t prev a x).1]
    grind

/-- **C5** Total trades of the ledger `≤` number of non-HOLD actions. -/
theorem runBacktest_trades_le (N : Nat) (κ : α) (ds : List (Action × Option α)) :
    ∀ t prev, prev ≤ 1 →
      ((runBacktest N κ t prev ds).map Row.trade).sum ≤
        ((ds.map Prod.fst).filter (fun a => a != .hold)).length := by
  induction ds with
  | nil => intro t prev _; simp [runBacktest]
  | cons d ds ih =>
    intro t prev hp
    obtain ⟨a, x⟩ := d
    have hpos : (runDay N κ t prev a x).pos ≤ 1 := by
      unfold runDay; split
      · exact hp
      · exact nextPos_le_one hp a
    have := ih (t + 1) _ hpos
    have htr : (runDay N κ t prev a x).trade ≤ (if a != .hold then 1 else 0) := by
      cases a with
      | hold => simp [(runDay_hold κ t prev x).2]
      | buy =>
        unfold runDay; split
        · simp
        · simpa using tradeSize_le_one hp (nextPos_le_one hp .buy)
      | sell =>
        unfold runDay; split
        · simp
        · simpa using tradeSize_le_one hp (nextPos_le_one hp .sell)
    have hlen : (((a, x) :: ds).map Prod.fst |>.filter (fun a => a != .hold)).length
        = (if a != .hold then 1 else 0) + ((ds.map Prod.fst).filter (fun a => a != .hold)).length := by
      simp only [List.map_cons, List.filter_cons]
      split <;> simp_all <;> omega
    rw [hlen]
    simp only [runBacktest_cons, List.map_cons, List.sum_cons]
    generalize (if a != .hold then 1 else 0) = k at htr ⊢
    omega

/-- Away from the calendar end (`t + n + 1 ≤ N - 1` for `n` decisions), the ledger's positions
and trades are exactly `posTrades` (what `clmcheck backtest` prints). -/
theorem runBacktest_eq_posTrades (N : Nat) (κ : α) (ds : List (Action × Option α)) :
    ∀ t prev, t + ds.length + 1 ≤ N - 1 →
      (runBacktest N κ t prev ds).map (fun r => (r.pos, r.trade)) = posTrades prev (ds.map Prod.fst) := by
  induction ds with
  | nil => intro t prev _; rfl
  | cons d ds ih =>
    intro t prev hN
    obtain ⟨a, x⟩ := d
    simp only [List.length_cons] at hN
    have hskip : ¬ t + 2 > N - 1 := by omega
    obtain ⟨h1, h2⟩ := runDay_traded κ prev a x hskip (N := N) (t := t)
    simp only [runBacktest_cons, List.map_cons, posTrades, h1, h2]
    rw [ih (t + 1) _ (by omega)]

end
end ClmVerify
