No confirmed direct look-ahead leak was found in fold construction, features, labels, normalisation, or C5 timing. I found these issues:

1. **Major — Amihud feature violates the stated C1/C2 formula.**  
   [features.rs:251](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-data/src/features.rs:251), [features.rs:263](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-data/src/features.rs:263)  
   C1 defines `C = adj_close`; C2 specifies Amihud as `|r1[k]| / (C[k] * V[k])`. The implementation instead divides by `dollar_volume`, explicitly constructed from the reported unadjusted close ([bars.rs:489](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-data/src/bars.rs:489)).  
   Failing scenario: `close=100`, `adj_close=90`, `volume=1m`. The required denominator is `90m`; code uses `100m`, shifting that observation by `ln(0.9)`. This changes feature 12, normalisation, and external-baseline exports.  
   Fix: use `x.close[k] * x.volume[k]` in `amihud`, or deliberately amend C1/C2 to define an explicit reported-close exception. The current test locks in the divergence: [features.rs:37](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-data/tests/features.rs:37).

2. **Major — classification metrics silently accept NaN probabilities despite claiming they panic.**  
   [metrics.rs:49](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/metrics.rs:49), [metrics.rs:225](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/metrics.rs:225), [metrics.rs:292](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/metrics.rs:292)  
   `check_inputs` validates only lengths and labels. For one row `probs=[NaN, 0, 0], labels=[0]`, `argmax` selects BUY, so accuracy is reported as 1.0, while Brier/ECE become NaN; AUC may be `None` rather than rejecting the row. This can quietly corrupt a report.  
   Fix: reject non-finite probabilities up front; ideally also require finite probabilities in `[0,1]` and rows summing to one within tolerance. Add NaN/+∞/−∞ tests.

3. **Major — bootstrap and paired-comparison routines do not reject non-finite returns.**  
   [statistics.rs:124](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/statistics.rs:124), [statistics.rs:198](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/statistics.rs:198)  
   `StatsError::NonFinite` exists, but bootstrap paths never use it. A single NaN return yields NaN estimates and percentile bounds; an infinity can similarly poison Sharpe and paired differences. The result serializes as invalid statistical output rather than failing.  
   Fix: validate every return before resampling and return `StatsError::NonFinite(index)`; likewise validate per-year comparison inputs. Add tests for NaN and both infinities.

4. **Minor — OOF Parquet reader accepts semantically invalid OOF rows.**  
   [oof.rs:159](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/oof.rs:159)  
   `oof_from_batch` checks only column presence and Arrow physical type. It accepts, for example, `label=255`, NaN probabilities/logits, or nulls in semantically non-null numeric columns; the invalid label later panics metrics, and invalid probabilities trigger finding 2. The round-trip test only covers files emitted by the same writer ([oof.rs:31](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/tests/oof.rs:31)).  
   Fix: validate nullability, label range, finite values, and probability-row invariants while reading; add malformed external-Parquet tests.

Verified correct:

- All 20 feature indices/lookbacks, same-day cross-sectional construction, and window validity are causal; no feature reads a future bar.
- Label endpoints, strict BUY/SELL inequalities, stop inequality, and label existence are correct.
- Fold keep arithmetic matches Lean’s Nat-style saturating/clamped form; embargo is applied forward in the correct direction.
- Runtime leakage guard independently checks every later segment and test-label calendar bounds.
- Fold normalisation uses valid feature rows from training days only.
- Backtest correctly attributes decision `t` P&L to day `t+1`, handles the untraded tail, charges costs on missing-open trades, and does not label-gate trading.
- Sharpe, Sortino, drawdown, profit factor, tied AUC, Spearman IC, stationary-bootstrap transitions, paired resampling, temperature scaling, vector Newton scaling, and isotonic PAV logic conform to their stated formulas for finite valid inputs.
- Targeted `clm-data` and `clm-eval` test suites passed; several existing tests are strong, but the Amihud test currently enshrines the contract mismatch.