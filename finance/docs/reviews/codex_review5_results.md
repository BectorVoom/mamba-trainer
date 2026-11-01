- **Lines 3–9, 201–204 — major:** “trades the same,” “small positive average comes from lower turnover,” and “does not improve … trading in any of 12 paired comparisons” overstate the evidence. Lower turnover is correlated with the post-cost difference but is not a demonstrated cause; forecast results are mixed. Moreover, seed 3 reports a positive paired mean-daily-return difference of **+0.82 bp [+0.14, +1.56]** (unadjusted), even though its Sharpe interval includes zero.  
  **Should say:** “Across the reported Sharpe comparisons, all D−C 95% intervals include zero. D consistently has lower turnover; the larger post-cost than zero-cost average advantage is consistent with lower cost exposure, not proof of its cause. One seed has a nominally positive paired mean-return interval.”

- **Lines 11–15, 32 — critical:** “No model beats equal-weight buy-and-hold” and “linear … is level with it” are unsupported. The reports give no paired comparison, bootstrap interval, or hypothesis test against buy-and-hold; 0.97 versus 0.99 is only a point-estimate comparison.  
  **Should say:** “Buy-and-hold has the highest reported point-estimate Sharpe (0.99 versus linear’s 0.97); the study does not report uncertainty for that model-versus-buy-and-hold difference.”

- **Lines 13, 31, 153–154 — major:** “significantly” is only nominal. The document acknowledges seven unadjusted comparisons, so it cannot retain the categorical claim that all three baselines significantly beat C after considering multiplicity.  
  **Should say:** “The three baseline-vs-C paired bootstrap intervals exclude zero before multiplicity adjustment.”

- **Lines 16–17, 208 — major:** The constant-class-share log-loss range (1.077–1.079) is not reported in any generated report or JSON report. The claim that the best model is “0.011 below” it is therefore not auditable from the cited evidence.  
  **Should say:** Add a reported constant-prior baseline with its exact sample definition, or remove the numerical comparison.

- **Lines 25, 61–66, 84 — major:** “Fixed before any test fold was looked at” / “fixed in advance” is not established by the repository. TOML comments assert it, but provide no dated preregistration or immutable artifact. Also, the Rust external-model path does not enforce LightGBM’s claimed two-candidate tuning budget; it accepts externally supplied candidate losses without requiring two candidates or validating the selection protocol.  
  **Should say:** “Specified in the checked-in configuration” and, for LightGBM, “externally generated predictions whose fold manifest is verified”; do not claim preregistration or equal tuning enforcement without supporting artifacts.

- **Lines 110–114 — minor:** “Vector scaling and isotonic regression … overfit” is causal language not established by one held-out result. The numeric range is also loose: isotonic increases raw log loss by 0.0081 for B and 0.0098 for C, not at least 0.01. Vector scaling beats temperature-scaled log loss for C and D.  
  **Should say:** “In this test set, isotonic generally has higher log loss than raw outputs; temperature has the lowest ECE for each main-study model.”

- **Lines 130–139 — major:** “The ranking follows turnover almost exactly,” “the more a model trades, the worse it does,” and “one calibration year is too noisy” are causal/statistical conclusions not tested by the reports. Eight confounded model point estimates do not identify turnover as the cause.  
  **Should say:** “Across these models, lower turnover and higher Sharpe co-occur; this comparison does not isolate turnover’s effect. Threshold tuning was inconsistent in these folds.”

- **Lines 169–171 — major:** “D’s one consistent property is that it trades about 13% less. Its log loss is no better” suppresses mixed forecast evidence. D has lower calibrated log loss in seeds 1 and 3, d_state-32, horizon-20, embed-16, and multi-positive; the reported four-seed average is lower for D (1.0717 vs 1.0722).  
  **Should say:** “D consistently has lower turnover. Its calibrated log-loss results are mixed; its four-seed mean is slightly lower, without a reported uncertainty interval.”

- **Lines 191–197 — major:** The CUDA/wgpu reproducibility claim is not supported as written. `kaggle_head.md` repeats C’s metrics, but neither report records backend/device, and “1.0743 → 1.0733” is raw-to-temperature-calibrated log loss, not a cross-backend comparison. Matching rounded aggregate metrics does not show the backends “train the same model.”  
  **Should say:** “The two report files show identical rounded C metrics; backend-specific run metadata or deterministic weight/prediction comparisons would be required to claim cross-backend equivalence.”

- **Lines 205–212 — major:** “The Mamba encoder itself is the weak part,” “mostly finds noise,” and “what a weak, real signal would look like” are unsupported causal interpretations. The study compares whole model configurations, not an intervention isolating the encoder, and the synthetic learnability test only establishes that a planted synthetic signal can be learned.  
  **Should say:** “In these configurations, the last-row baselines have better reported forecast and trading point estimates than the sequence models; the study does not identify why.”

- **Lines 216–221 — major:** The stated three-runtime test coverage is not supported by the test code. `tests/selective_scan.rs` uses `Device::<Auto>::default()` and does not instantiate or require CPU, wgpu/Radeon, and CUDA/T4. Its half-precision test conditionally skips unsupported dtypes.  
  **Should say:** “The scan test suite compares the active backend’s kernels with an f64 reference; backend/device coverage and f16/bf16 execution require recorded test logs.”

- **Lines 223–235 — major:** The Lean coverage is overstated in two places. The Lean report explicitly says the checkpoint layout, padded segment, shared-memory reductions, and parameter-gradient sums are not modelled; “checkpointed states = sequential states” is an inference from the forward theorem, not a direct proof of the GPU backward implementation. Also, the Rust-vs-`clmcheck` differential tests are `#[ignore]` and cover folds/backtest rendering, not the scan kernel or all Rust behavior.  
  **Should say:** “Lean proves the stated abstract definitions. The GPU/Rust implementations are not proven; optional differential tests cover fold construction and action-ledger rendering, while scan kernels are separately numerical-reference tested.”

- **Lines 239–247 — minor:** The test count for clm-data is inaccurate as a source-level count: there are 40 `#[test]` cases (37 excluding the three conditional real-data tests), not simply 37. The two Lean-oracle differential tests in clm-eval are ignored by default.  
  **Should say:** “37 non-real-data clm-data tests plus 3 conditional real-data tests; 88 clm-eval tests, including two ignored Lean-oracle differential tests; 8 clm-model tests; 8 scan tests.”

- **Lines 41–46 — major:** “An audit recomputes every feature … and finds 0 mismatches on the real data” overstates the code. The real-data truncation audit samples roughly 50 dates, not every date, and returns early when `data/bars` is absent. No generated report records a successful real-data audit run.  
  **Should say:** “The repository contains a conditional real-data truncation audit over sampled dates; publish its executed output before claiming zero mismatches on the study data.”

- **Lines 257–258 — major:** `report` does not refuse a mismatched `run.json` signature. It checks deserialized config equality and dataset version, then explicitly permits a source-hash mismatch and merely logs it. `cv` does enforce its computed signature on resume.  
  **Should say:** “`cv` rejects resume when its computed config/data/source signature differs. `report` verifies stored config and dataset version, while recording—not rejecting—a source-hash difference.”

- **Omitted report finding — major:** The main report gives E a paired mean daily advantage over C of **+1.06 bp [+0.25, +1.83]**; seed 3 gives D **+0.82 bp [+0.14, +1.56]**. These are nominal, unadjusted, and do not make the Sharpe results significant, but they should be disclosed beside the negative Sharpe conclusion.
