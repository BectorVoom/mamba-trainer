# ClmVerify: Lean 4 model and proofs of the Mamba-CLM contracts

This project states the rules of [`finance/docs/CONTRACTS.md`](../../docs/CONTRACTS.md) in Lean 4 and proves what they guarantee. It covers C2/C3 (feature and label windows, label classes), C4 (walk-forward folds, purging, embargo), C5 (policy and backtest accounting) and C7 (selective scan: algebra, chunked kernel, causality, adjoint).

**What is and is not verified.** The proofs are about the Lean definitions in `ClmVerify/Scan.lean`, `Windows.lean`, `Folds.lean` and `Backtest.lean`. The Rust code (`finance/crates/*`, `src/tensor/ops/selective_scan.rs`) and the CubeCL GPU kernels are **not** proven. They are *tested* against the executable `clmcheck`, which runs exactly these definitions. This is the same arrangement as `lean/MotifDecoder`.

- **Lean**: `Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`, pinned in `lean-toolchain`.
- **Dependencies**: none (core Lean only). The algebra uses the core classes `Lean.Grind.CommSemiring`, `Lean.Grind.CommRing` and `Lean.Grind.OrderedRing`. `Int`, `Rat` and `Nat` (semiring) are instances.
- **Status**: every theorem below is proven. The build has zero errors and zero warnings. There is no `sorry`, no added `axiom` and no `native_decide`.
- **Size**: 1,848 lines (definitions 403, proofs 1,180, `Axioms.lean` 82, `Main.lean` 183).
- **Build**: `lake build` from a clean `.lake/` takes about 9 s on this PC, executable included.

## Layout

| File | Content |
|---|---|
| `ClmVerify/Scan.lean` | C7 definitions: `Aff`, `comb`, `seqScan`, chunked forward scan (`summarize`, `carries`, `chunkedScan`), readout `ys`, `revScan`, chunked reverse scan (`revSummarize`, `revCarries`, `chunkedRevScan`), `adjPairs`, `adjoint`. |
| `ClmVerify/ScanProofs.lean` | C7 theorems. |
| `ClmVerify/Windows.lean` | C2/C3 definitions: `windowRows`, `rowReads`, `featureReads`, `labelDays`, `labelPriceReads`, `Action`, `theta`, `classify`. |
| `ClmVerify/WindowProofs.lean` | C2/C3 theorems. |
| `ClmVerify/Folds.lean` | C4 definitions: `Seg`, `FoldConfig`, `firstIdx`, `testStart`, `testEnd`, `mkFold`, `folds`, `keepRule`, `testKeepRule`, `Fold.kept`, `Fold.keepEnd`. |
| `ClmVerify/FoldProofs.lean` | C4 theorems. |
| `ClmVerify/Backtest.lean` | C5 definitions: `nextPos`, `tradeSize`, `posTrades`, `execDay`, `pnlReads`, `runDay`, `runBacktest`, `runSegment`. |
| `ClmVerify/BacktestProofs.lean` | C5 theorems. |
| `ClmVerify/Axioms.lean` | `#print axioms` for 71 main theorems (built with the library). |
| `Main.lean` | `clmcheck`. It imports only `Scan`, `Folds` and `Backtest`, which hold definitions only. |
| `test/cases.txt`, `test/expected.txt` | 45 commands and the executable's output on them. |

## Theorems ↔ contract items

### C7 selective scan (one `b, d, n` lane, any commutative semiring)

| Item | Theorem(s) |
|---|---|
| (a) `⊕` is associative, identity `(1,0)`, it is composition | `Aff.comb_assoc`, `Aff.one_comb`, `Aff.comb_one`, `Aff.apply_comb` |
| (b) the fold of `⊕` over `p_0..p_t`, applied to `h0`, is `h_t` | `seqScan_getElem?`, `final_eq_comb_fold` |
| (c) the chunked scan equals the sequential scan for every `L ≥ 1` | `chunkedScan_eq_seqScan`. Supporting lemmas: `chunked_eq_seq_of_split` (phases 2+3 are correct for any split), `chunksOf_flatten`, `chunksOf_shape` (chunks have 1..L steps), `summarize_apply`, `summarize_eq_fold` (phase 1 computes the monoid product) |
| (d) causality | `seqScan_causal`, `ys_causal` (readout `y_t = c_t h_t + d x_t`), `chunkedScan_causal` |
| (e) `Σ gy_t c_t h_t = Σ g_t u_t + g_0 a_0 h0` | `adjoint_identity`; recurrence restated index-wise in `adjoint_getElem?` |
| (e) chunked reverse scan equals the sequential reverse scan | `chunkedRevScan_eq_revScan`, `chunkedRev_eq_of_split`, `chunked_adjoint` |
| extras | `revScan_eq_reverse`, `adjoint_indep_u` |

**Forward:** the chunked forward kernel is modelled as it runs. Phase 1 runs, per chunk, a local scan from 0 and the product of `a` (the CubeCL kernel multiplies the per-step factors, as the model does). Phase 2 is a sequential pass over the chunk summaries, giving each chunk's carried-in state. Phase 3 re-scans each chunk locally from that state.

**Backward: what is and is not covered** (corrected after the codex review of 2026-10-22). The CubeCL backward kernel does *not* run a three-phase reverse scan. It walks the checkpoint segments from last to first; per segment it recomputes the forward states from the segment's checkpoint and then runs the reverse adjoint recurrence sequentially through the segment, carrying `g` across segments. What the proofs give for it:

- the checkpoints the forward writes are the sequential states at segment starts (`chunkedScan_eq_seqScan`, with chunk = segment), so a segment recomputed from its checkpoint reproduces the sequential states;
- the recurrence it walks is `revScan`, and `adjoint_identity` shows that `g` is the exact gradient of `Σ gy·c·h` with respect to `u`.

The checkpoint layout, the padded last segment, the per-cube shared-memory reductions over channels and the parameter-gradient sums are **not** modelled; they are tested against the `f64` reference (`tests/selective_scan.rs`). `chunkedRevScan_eq_revScan` (a three-phase reverse scan) is proven but describes no kernel that exists today; it is what a chunk-parallel backward would rely on.

### C2, C3

| Item | Theorem(s) |
|---|---|
| The window reads exactly `[t-W+1-K, t]` (pre: `W ≥ 1`, `t ≥ W-1+K`) | `mem_featureReads`, `featureReads_le` |
| Label days are `[t+1, t+1+h]`; the prices read lie inside them | `mem_labelDays`, `labelDays_bounds`, `labelStart_mem`, `labelEnd_mem`, `labelPriceReads_sub` |
| Max feature index < min label index | `feature_lt_label` |
| Class rule: BUY iff `r > θ`, SELL iff `r < -θ` (θ ≥ 0), exactly one class | `classify_buy_iff`, `classify_sell_iff`, `classify_hold_iff`, `classify_exactly_one`, `not_buy_and_sell`, `theta_nonneg` |

### C4

`years : List Int`, `N = years.length`, `Sorted` = non-decreasing. Segments are 0 = train, 1 = val, 2 = cal, 3 = test.

| Item | Theorem(s) |
|---|---|
| What an emitted fold is | `mkFold_some`, `fold_emitted`, `mem_folds` |
| C4.1: contiguous, adjacent, ordered, inside `[0,N)`, pairwise disjoint; val/cal lengths; test non-empty | `fold_layout`, `fold_chain`, `mem_days`, `seg_stop_le_start`, `seg_ordered`, `seg_disjoint`, `seg_inside` |
| The test segment is exactly the days of year Y | `test_eq_year` |
| C4.2: a kept sample's label day `l` and any later segment `S`: `l + e < S.start`, `l < S.start - e`, and `l + e < d` for every day `d` of `S` | `purge` (no calendar assumption) |
| Kept test labels lie inside the calendar | `kept_test_label_inside` |
| C4.3: test segments of different folds are increasing and disjoint | `folds_test_increasing`, `folds_test_disjoint` |
| C4.5: Expanding, `train(k) ⊆ train(k')` for every later fold | `expanding_train_mono` |
| Rolling: `train.len ≤ train_days` | `rolling_train_len` |
| Kept set = `[a, clamp(s_next-1-h-e, a, b))` (test: `clamp(N-1-h, a, b)`) | `kept_eq_range`, `mem_kept`, `mem_kept_iff_range` |
| Leak injection | `not_kept_of_violation`, `not_kept_of_label_reaches_next`, `not_kept_test_of_label_past_end` |

`fold_layout`, `seg_*` and `test_eq_year` assume `Sorted years`; otherwise `te` could fall before `ts`.

### C5

| Item | Theorem(s) |
|---|---|
| `p(t) ∈ {0,1}`, `trade(t) ∈ {0,1}` | `posTrades_mem`, `runBacktest_mem` |
| HOLD never trades and keeps the position | `hold_no_trade`, `hold_keeps_first`, `hold_keeps`, `runDay_hold` |
| Execution day `t+1 > t`; P&L reads are at index `≥ t+1` | `execDay_gt`, `pnlReads_ge` |
| `Σ net = Σ gross − κ·Σ trade` | `accounting`, `runDay_net` |
| Trades ≤ non-HOLD actions | `trades_le_nonhold`, `runBacktest_trades_le` |
| Skip rule `t+2 > N-1`; traded days read only `[t+1, N-1]` | `runDay_skip`, `runDay_reads_inside` |
| Missing open: gross 0, position recorded, cost charged | `runDay_missing` |
| The ledger equals `posTrades` away from the calendar end | `runBacktest_eq_posTrades` |

## Modelling choices and simplifications

- **Exact algebra, not floats.** Floats are not a semiring because addition is not associative. Chunked = sequential holds exactly in exact arithmetic (checked at `Int` by `clmcheck`); for f32/f64 it holds only up to rounding, so Rust must compare with a tolerance. The same applies to `adjoint_identity`.
- **One lane.** `u = dt·B·x` and `a = exp(dt·A)` are inputs. The sum over `n` and the parameter-gradient formulas (`dx`, `ddt`, `dA`, `dBm`, `dCm`, `dDskip`) are not proven. `h[-1] = 0` is generalised to `h0`.
- **C2/C3: indices only.** Values and bar-level data validity are not modelled.
- **C5: `ret = O[t+2]/O[t+1] − 1` is data** (`Option α`; `none` = missing open). The portfolio average, the metrics and C6 are not modelled.
- **C4.4 and normalisation** (pipeline properties) are not modelled.
- **`clmcheck backtest`** ignores the calendar end, as specified; `runBacktest_eq_posTrades` links it to the full ledger.
- **`adjoint`** ignores `u`; `clmcheck` passes `u = 0` (`adjoint_indep_u`).

None of these weakens a requested theorem.

## Executable `clmcheck`

`.lake/build/bin/clmcheck` reads one command per line and answers each with result lines followed by `end`. Bad input gives `error <msg>` then `end`. Blank and `#` lines produce no output.

    folds N scheme train_days first_test last_test val_days cal_days h e min_train_days nyears y1 c1 … → fold Y train a b val a b cal a b test a b keep ta tb va vb ca cb xa xb
    backtest n a1 … an        → pos … / trade …
    scan n L h0 a1 u1 …       → seq … / chunk …
    adjoint n a1 c1 gy1 …     → g …
    adjointc n L a1 c1 gy1 …  → g … / chunk …   (extension, for a chunked backward kernel)

`folds` checks that `Σc = N`, that each `c ≥ 1` and that years strictly increase. Keep ranges are printed from `Fold.keepEnd`, the closed form `kept_eq_range` proves equal to the per-day rule.

Checks: `test/expected.txt` covers chunk sizes 1, n, > n and non-divisible; n = 0; skipped folds; expanding vs rolling; rolling clipped at 0; empty val/cal; `h + e` emptying keep ranges; negative years; malformed commands. A separate Python port of the contract text agreed with `clmcheck` on 3 × 3000 random commands (about 28k lines, about 760 folds).

## `#print axioms` (71 theorems)

| Axioms | Theorems |
|---|---|
| none | 1 |
| `propext` | 11 |
| `propext`, `Quot.sound` | 19 |
| `propext`, `Classical.choice`, `Quot.sound` | 40 |

These are Lean's three standard axioms; no `sorryAx` and no project axiom appear.
