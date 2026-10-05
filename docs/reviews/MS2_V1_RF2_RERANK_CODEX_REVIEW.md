# Codex review: fix task RF2 and the reranker/calibration experiment (P6.3, P7.9)

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree over d39ec34. Verdicts: Part A (RF2) reject — one finding (bf16 ranking terms narrowed before the sum) plus the documented synchronisation-test gap; Part B (reranker experiment) reject — seven findings. The fixes are task RKF3.

Static review of the uncommitted tree over `d39ec34`. No files modified; no Cargo commands or Rust tests run.

**PART A — RF2**

| Item | Status | Evidence |
|---|---|---|
| 1. N1 / I-C4: bf16 score gathering and producer test | **partly** | [ms2_pack.rs:405](src/tensor/ops/ms2_pack.rs:405) fixes the size-mismatched reinterpret, but narrows the f32 trace accumulator to bf16. The producer-through-ranking test exists and compares all packed fields with host `pack`, but deliberately uses only bf16-exact terms ([test:1341](tests/ms2_pack_kernels.rs:1341)). It misses the precision defect below. |
| 2. Sampler accumulator, other reinterprets, host readers, dtype test | **resolved** | [ms2.rs:7093](src/tensor/ops/ms2.rs:7093) decodes f32, widens the step term, and stores f32 bits. The remaining executable reinterprets in the requested files are size-compatible f32↔u32; `mixer_step.rs` has none. Host readers decode with `f32::from_bits`: `generate.rs:1726,1974`, `pack.rs:1024`, `twin.rs:1129`. [ms2_dtype.rs:732](tests/ms2_dtype.rs:732) checks launch errors and requires at least one finished trajectory, using in-range precursors. |
| 3. I-C8: host allocation preflight | **resolved** | [pack.rs:1800](src/models/ms2/pack.rs:1800) checks staging and packed-output products before the first staging allocation. Smaller assembled-field products are bounded by those checked widths. [ms2_pack.rs:1078](tests/ms2_pack.rs:1078) calls `pack` with empty vectors and oversized dimensions, requiring the address-domain error before length validation. |
| 4. N2: workspace evidence refusal | **resolved** | Both adapters reject `evidence=true`: [generate.rs:2339](src/models/ms2/generate.rs:2339) and [generate.rs:2385](src/models/ms2/generate.rs:2385). Both are exercised by the new test. |
| 5. N3: shared completeness rule and producer compatibility | **resolved** | [contract.rs:942](src/models/ms2/contract.rs:942) implements the shared rejection, called by both validators. The table producer starts with `complete=0` and sets only exhaustion when `joined > cap` ([ms2.rs:3468](src/tensor/ops/ms2.rs:3468)); enumeration likewise clears completeness on exhaustion ([ms2_enum.rs:1833](src/tensor/ops/ms2_enum.rs:1833)). I found no table cap-overflow production regression from this fix. |
| 6. R-A1: identity extent arithmetic | **resolved** | [identity.rs:507](src/models/ms2/identity.rs:507) and [ms2_identity.rs:557](src/tensor/ops/ms2_identity.rs:557) now perform extent division/subtraction in u32. Those operations match on validated layouts; the host additionally rejects oversized slices, while the device wrapper checks lengths before launch. |
| 7. R-D-sync: observable draining | **partly** | Production still calls `client.sync()` at [backend.rs:220](src/backend.rs:220). The new output check cannot distinguish a missing drain because subsequent flush/read drains work. The limitation is stated honestly in the module doc comment at [ms2_profile.rs:13](tests/ms2_profile.rs:13), and explicitly at line 1340. |
| 8. Metrics fixture | **resolved** | [ms2_metrics.rs:436](tests/ms2_metrics.rs:436) supplies non-sentinel formula provenance and nonzero counters, retaining `batch.validate()`. Validation was not weakened. |
| 9. Counter-test serialization | **resolved** | Every test body in both binaries takes the mutex: all 15 experiment tests and all 13 decoder tests. This covers every device-touching test. |

1. **P1 — bf16 packed ranking still disagrees with host packing.**  
   Location: [ms2_pack.rs:405](src/tensor/ops/ms2_pack.rs:405), with the neural-dtype score buffer declared at [generate.rs:138](src/models/ms2/generate.rs:138).

   **Failing scenario:** Two eligible trajectories have the same formula log-probability `−0.5`, with f32 trace accumulators `−1.00390625` for trajectory 0 and `−1.0009765625` for trajectory 1. Host scores are `−1.50390625` and `−1.5009765625`, so trajectory 1 wins. Both trace terms round to bf16 `−1`, making device scores tie at `−1.5`; trajectory 0 wins. Packed trace values also lose precision.

   **Minimal fix:** Preserve gathered ranking terms in an f32 score buffer, widening formula terms into it, and use that buffer for ranking and float packing. Extend the producer test with non-bf16-exact accumulated trace terms.

2. **Verification gap — the synchronization mutant still passes.**  
   Location: [ms2_profile.rs:1310](tests/ms2_profile.rs:1310).

   **Failing scenario:** Replace `client.sync()` with `Ok(())`, retaining the counter increment. Pending work can remain incomplete at the span boundary, but subsequent `check_launches` and readback complete it, so all assertions pass.

   **Minimal fix:** Add a deterministic runtime test seam that observes synchronization completion before another flush/read, or a supported backend test with a sync-only deferred failure. The current documentation correctly acknowledges the gap.

**Part A verdict: reject.** The primary bf16 host/device ranking contract remains broken.

**PART B — reranker and calibration experiment**

1. **P1 — missing checkpoint provenance silently bypasses the leakage guard.**  
   Location: [ms2_rerank_experiment.rs:590](examples/ms2_rerank_experiment.rs:590).

   **Failing scenario:** Load a table-source or older checkpoint with both `enum_fit_name` and `enum_fit_sha256` absent, then pass its actual training export as `--report`. Both `is_some_and` checks return false and the run reports held-out results. Moreover, these fields identify enumeration fitting, which can differ from generator training.

   **Minimal fix:** Require explicit generator-training provenance or a supplied fit export; refuse unverifiable provenance. Compare fit molecule keys against all three experiment splits, including provenance for fitted formula artifacts.

2. **P2 — valid cold GPU scoring fails the one-read assertion.**  
   Location: [ms2_rerank_experiment.rs:799](examples/ms2_rerank_experiment.rs:799).

   **Failing scenario:** On a GPU with an empty tuning cache, train with batches of 1,024 examples, then score 200 calibration examples. The new matrix shapes trigger autotuning, whose correctness check reads device data at [matmul.rs:3160](src/tensor/ops/matmul.rs:3160). The final logit download adds another read, and the driver aborts despite correct scoring.

   **Minimal fix:** Warm each actual scoring shape before measuring a repeated scoring call, then require exactly one read for that measured call.

3. **P2 — `--bootstrap 0` replaces measured points with zero.**  
   Locations: [rerank_eval.rs:285](src/models/ms2/rerank_eval.rs:285) and [ms2_rerank_experiment.rs:257](examples/ms2_rerank_experiment.rs:257).

   **Failing scenario:** A report has eligible positive candidates in every spectrum. With `--bootstrap 0`, reported top-1 precision becomes `0`, although it is `1`. A paired comparison with all differences equal to `1` likewise reports difference `0`.

   **Minimal fix:** Reject zero bootstrap repetitions at argument validation, or compute the point independently and represent the unavailable interval explicitly.

4. **P2 — NaN handling destroys pairing within molecules.**  
   Location: [rerank_eval.rs:264](src/models/ms2/rerank_eval.rs:264).

   **Failing scenario:** `a=[1, NaN]`, `b=[NaN, 0]`, molecules `[0,0]`. There is no spectrum with both measurements, yet the function returns a paired difference and interval of `(1,1,1)` by averaging different spectra independently.

   **Minimal fix:** Retain only groups defined in both rankings, calculate their paired differences, then average and resample molecules. Add an asymmetric-NaN test.

5. **P2 — valid negative candidates disappear from size strata.**  
   Location: [ms2_rerank_experiment.rs:643](examples/ms2_rerank_experiment.rs:643), consuming [metrics.rs:194](src/models/ms2/metrics.rs:194).

   **Failing scenario:** A finished, device-valid three-atom C–C–O candidate is generated under an incorrect retained formula for an oxygen-free parent. Replay under the *true parent* composition fails, so evaluation supplies `NotContained` and `atoms=0`. The driver retains the negative label but omits it from the 3–5 stratum. With one positive and this negative, both assigned probability `0.9`, that stratum reports Brier `0.01` instead of `0.41`.

   **Minimal fix:** Derive candidate size from its valid trace independently of the true-parent budget; use true-parent containment only for the label.

6. **P2 — calibration configuration keys collide across different search policies.**  
   Location: [ms2_rerank_experiment.rs:544](examples/ms2_rerank_experiment.rs:544).

   **Failing scenario:** Two enumeration runs have identical K, F and M but lane visit limits of `1` and `4096`. They produce different scored supports, yet receive identical `ConfigKeys`, so `validate_matches` accepts calibration across them.

   **Minimal fix:** Include the effective search limits and frozen search-artifact identity in `search_policy`, preferably through a canonical configuration fingerprint.

7. **P2 — report aggregation metadata overstates molecule weighting.**  
   Location: [ms2_rerank_experiment.rs:913](examples/ms2_rerank_experiment.rs:913).

   **Failing scenario:** Molecule A has nine spectra with perfectly ordered positive/negative examples; molecule B has one spectrum with reversed ordering. The pooled AUC is `0.9`, while the mean per-molecule AUC is `0.5`. The JSON reports the former under an aggregation statement claiming means over molecules.

   **Minimal fix:** Give AUC/base rate their own “pooled eligible examples” aggregation labels. Explicitly state that top-1 and precision metrics condition on spectra with at least one eligible candidate, and report the eligible molecule count.

The remaining protocol is sound on inspection: generation is frozen under `no_grad`; reranker updates use `rank` only; Platt parameters and their standardization use `calibration` only; held-out quality uses `report`. Epoch count and binning rules are fixed rather than selected using report labels. The three split-disjointness checks compare exported molecule keys, not local indices.

Eligibility excludes unfinished, invalid, trace/graph duplicates and counted containment work limits. Features follow §4.3, including no-evidence defaults. Raw and reranker rankings use identical examples, with raw score computed as the f32 formula-plus-trace sum.

For finite binary inputs, tied ROC AUC, top-1 and available-candidate precision estimators are correct. Nonempty bootstraps are deterministic molecule resamples with percentile intervals. ECE and Brier formulas are correct; equal-mass binning is a predefined estimator, not report-driven model selection. The pseudo-label description is explicit.

The serialized `report_boundaries_read_once` test makes non-vacuous zero-read/exactly-one-read assertions. The synthetic pipeline test does not exercise the driver’s leakage guards, export processing or cold GPU scoring.

**Part B verdict: reject.** Leakage validation and several reported-result paths require correction.