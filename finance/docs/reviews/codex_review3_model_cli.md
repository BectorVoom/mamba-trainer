No files modified. I found one launch-blocking reproducibility issue and several material reporting/fairness issues.

1. **Critical — stale/mixed outputs are silently resumed and reported as one run.**  
   [cv.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/cv.rs:131) skips any model solely because `model.json` exists. It neither compares the saved configuration, dataset version, source revision, fold metadata, nor normalizer against the current run. [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:91) then loads whatever artifacts happen to exist.

   Failing scenario: run half the folds; change `main.toml`, bars, or code; resume in the same output directory. Old folds/models are retained, new ones use the new setup, and the report presents the mixture as one experiment.

   Fix: make output directories immutable per run signature, or require exact equality of saved config, full data hash, source hash, fold metadata, and model configuration before resuming. Refuse to report mixed signatures.

2. **Major — the report silently aggregates different model/fold populations.**  
   [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:84) simply omits a missing fold/model; model-level metrics are calculated on each model’s surviving subset at [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:150). Pairwise comparisons are merely skipped when daily dates differ at [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:285).

   Failing scenario: Mamba+CLM finishes 2012–2025 but Mamba+MLP fails or is absent for 2024–25. Their headline Sharpe, calibration, cost, and regime tables cover different market periods yet appear side by side.

   Fix: require every configured model to have a valid artifact for every requested fold before producing the study report; alternatively calculate and prominently label a single common-fold intersection for all comparison tables.

3. **Major — external LightGBM predictions are not provenance-checked, and surplus rows are accepted.**  
   [cv.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/cv.rs:244) maps predictions by `(s,t)` and checks only that each expected key exists; it silently ignores extra rows. The only metadata consumed is an optional scalar `val_nll` at [cv.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/cv.rs:263). No dataset/fold/config/export hash, candidate count, seed, or selected-candidate evidence is verified.

   Failing scenario: `external_dir` points to predictions generated from an export that included calibration/test labels, or a combined prediction file. If it contains the required keys, Rust accepts it and reports it as the fair two-candidate baseline.

   Fix: export a manifest containing dataset hash, fold boundaries, normalizer hash, exact split key hashes, config hash, and seed; require LightGBM to propagate it and verify it exactly. Reject files whose key set is not exactly the requested split. Store both candidate validation losses and selected settings in the Rust result.

4. **Major — yearly results are assigned by P&L accrual year, not the fold’s test year.**  
   [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:215) groups returns by `years[d]`, where `d=t+1`. The backtest explicitly records an action made on `t` as P&L on `t+1` ([backtest.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/backtest.rs:131), [backtest.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-eval/src/backtest.rs:314)).

   Failing scenario: the final trade decision of the 2023 test fold accrues on the first trading day of 2024. It is reported in 2024’s “Sharpe per test year,” despite belonging to the 2023 fold. Paired yearly comparisons have the same error.

   Fix: retain fold/test-year ownership while concatenating daily series and group yearly fold comparisons by that ownership. If calendar-year accrual reporting is desired too, label it separately.

5. **Major — dataset provenance is incomplete even before considering resume.**  
   [data.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/data.rs:54) hashes symbols and present close values only. Features, labels, and trading also depend on open/high/low/adjusted-close/volume/presence. Further, `git_commit()` intentionally ignores untracked files at [cv.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/cv.rs:84).

   Failing scenario: corrected opens or volumes change labels/features/backtests while closes remain unchanged; the recorded `dataset_version` stays the same. Or a study implementation exists as untracked source and the run metadata reports a clean commit.

   Fix: hash canonical bytes for every input field and calendar/presence mask; record a source-tree hash or require a clean, fully tracked revision.

6. **Minor — configuration validation permits late panics/non-finite training.**  
   [config.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/config.rs:257) validates names and fold scheme only. [train.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-model/src/train.rs:192) performs modulo by `eval_every`.

   Failing scenario: `eval_every = 0` panics during a multi-hour run; `temperature = 0`, non-finite costs/tau values, zero window/kernel, or invalid dimensions can produce infinities or downstream shape/pooling failures.

   Fix: validate every numeric domain at load time: finite/positive temperatures, dimensions, window/kernel, batch/eval sizes, `eval_every >= 1`, non-empty finite tau grid, non-negative finite costs/weights, and valid fold lengths.

7. **Minor — reported action shares omit mandatory HOLDs on input-invalid test days.**  
   [report.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/report.rs:231) counts only OOF records. OOF test records are generated only for input-valid samples ([data.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/data.rs:112)). C5 requires invalid-input days to be HOLD, and the backtest correctly implements that through `None`.

   Failing scenario: a stock has many missing-bar windows. Trading includes its HOLD state, but the action-share table overstates BUY/SELL.

   Fix: calculate shares from the complete `security × test-day` decision matrix used by the backtest, treating absent predictions as HOLD.

8. **Minor — CLM action attributes misstate the configured decision threshold.**  
   [config.rs](/home/user/Documents/workspace/mamba-trainer/finance/crates/clm-cli/src/config.rs:384) passes only `2*kappa` to `action_attributes`, while the label threshold is `2*kappa + margin_mult*sigma_h`. The action encoder therefore advertises a fixed threshold that is not the target rule when `margin_mult > 0` (the supplied main configuration uses `0.5`).

   Failing scenario: the study describes CLM as conditioning actions on threshold/cost semantics, but its threshold attribute omits the volatility-dependent component.

   Fix: either remove that misleading static threshold attribute, or make action/state conditioning include the per-sample threshold consistently.

Verified correct:

- Per-fold normalization is fit only on valid training-day feature rows, then applied to all splits.
- Train/validation/calibration sample selection is purged and guarded; test trading uses all input-valid days rather than label-gated days.
- Candidate selection and early stopping use validation NLL; calibration, tau selection, and the calibration backtest use calibration data only. The calibration P&L range is safely before test prices.
- Mamba+MLP and Mamba+CLM receive the same per-fold/candidate seed and sampled split stream; the main configuration gives C and D matching encoder/training budgets.
- The CLM action cache is disabled under gradient mode and invalidated before evaluation/restoration; action-encoder parameters are visited by the module.
- InfoNCE multi-positive loss, SupCon diagonal masking/positive averaging, auxiliary MSE, and stable BCE are mathematically consistent with their stated objectives.
- Best validation weights are saved and restored after early stopping.
- `decide()` resolves HOLD and tie cases deterministically; backtest HOLD semantics, next-open execution, costs, and final untradeable-day handling match the contract.
