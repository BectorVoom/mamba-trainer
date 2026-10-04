# Codex review: standalone modules (assignment head, reranker and calibration, fingerprint head, baseline encoders, data tools, Python bindings)

Reviewer: codex exec (read-only), 2026-10-04. Verdicts: A accept-with-fixes, B reject, C reject, D accept-with-fixes, E reject, F reject; the findings are the follow-up tasks.

No files were modified and no cargo commands were run. Numerical counterexamples below were checked with read-only Python calculations; Rust/GPU tests were not executed. CodeGraph unexpectedly returned snippets from excluded files; those snippets were not used as review evidence.

**PART A — assignment head**

1. **Major — `src/models/ms2/assign.rs:329`: the “partial” proxy does not measure label retention.**  
   With `J=4`, one label that is fully retained gives mask `[1,0,0,0,0]`; the code reports partial=1, although nothing was dropped. Conversely, five labels with four retained give mask sum=4 and partial=0, although a label was dropped. This can reverse conclusions about supervision coverage when changing `J`. `tests/ms2_assign.rs:264` repeats the proxy, so its “independent” count reference cannot detect this error.  
   **Fix:** emit an actual partial-label flag from label matching and reduce it on device. Until then, expose the proxy under a separate name, not `assignment_label_partial`.

2. **Minor — `src/models/ms2/assign.rs:120`: the peak projection adds an undocumented bias.**  
   `LinearConfig::new(d,d)` enables bias, implementing `e·(Wx+b)/sqrt(d)`. For `d=1`, `e=1`, `x=0`, projection bias=1 and unassigned logit=0, the hypothesis probability is 0.731059; the specified formula gives 0.5. The reference at `tests/ms2_assign.rs:185` reproduces the bias.  
   **Fix:** disable projection bias, or explicitly revise the specification and add a specification-derived test.

The class-mask kernel and twin agree: kept-count clamping and unavailable-bit precedence are correct, and unassigned prevents an empty distribution. The loss uses a stable log-space sum and the eligible-peak denominator. The formula row network remains differentiable; its gradient tests explicitly exercise shared parameters. No device read was found in either head operation.

**PART B — reranker and calibration**

1. **Major — `src/models/ms2/calibration.rs:139`: undamped Newton iterations can diverge on ordinary finite inputs.**  
   Reproduced input: 17 negatives at logit 0, followed by positives at logits 1 and 100. The implementation returns approximately:
   ```
   a = 54249.708081
   b = -7778786.089641
   ```
   All 19 fitted probabilities become zero. With Platt smoothing, the targets are `1/19` and `0.75`; the final intercept-gradient magnitude is 2.394737, so this is not convergence. A damped reference gives approximately `a=0.0343723`, `b=-2.3033037`, with probabilities 0.090850, 0.093729 and 0.756563. Smoothed BCE sums are approximately 14,518,730 versus 6.042580.  
   **Fix:** add objective-based backtracking/damping and verify convergence before returning success. Add an imbalanced, high-leverage fixture; the current recovery and finiteness tests miss this.

2. **Major — `src/models/ms2/rerank.rs:402`: BCE has the wrong derivative at zero logits.**  
   `maximum(x,0)` routes a tie away from `x` (`src/autograd/ops.rs:265`), while `abs` has derivative zero at zero. Consequently, the composed derivative at `x=0` is `−label`, rather than `0.5−label`. For one unit-weight negative example, the loss is correctly `ln(2)` but the gradient is **0 instead of +0.5**. A positive example gets **−1 instead of −0.5**.  
   **Fix:** use a stable BCE operation with backward `sigmoid(x)−label`, or an equivalent smooth formulation. Add exact-zero-logit gradient tests.

3. **Major — `src/models/ms2/rerank.rs:467`: zero weight decay does not ensure no change on zero-weight batches.**  
   After any nonzero-gradient step, Adam retains momentum. A subsequent all-zero-weight batch still updates parameters. For scalar gradients `[1,0]`, default Adam betas and zero decay, the second update is approximately `−0.670058 × learning_rate`. `tests/ms2_rerank.rs:214` starts with fresh optimizer state and therefore misses this.  
   **Fix:** skip optimizer updates for batches with no eligible examples, preferably using the host eligibility information already available. Test a positive-weight step followed by zero-weight steps, including optimizer-state preservation.

4. **Major — `src/models/ms2/rerank.rs:220` and `src/tensor/ops/ms2_rerank.rs:182`: evidence features do not consistently implement §4.3.**  
   The upstream evidence buffer counts *all* qualifying peaks, whereas §2.4 returns at most four evidence records. Six qualifying peaks therefore produce feature 4=**1.5**, instead of retained evidence count/4=**1**. Also, count=0 and status=128 produce incomplete=**1**, although §4.3 specifies incomplete=0 when there is no evidence. These are realistic upstream states.  
   `tests/ms2_rerank_kernels.rs:104` generates both cases but compares against the same twin arithmetic.  
   **Fix:** distinguish total matches from retained evidence count, use the specified retained count, and apply the no-evidence defaults. Add hand-computed feature fixtures independent of the twin.

5. **Minor — `src/models/ms2/calibration.rs:102`: standardization does not guarantee finite results for finite inputs.**  
   For logits `[1e308,1e308]` and labels `[0,1]`, summation overflows, and the returned intercept becomes NaN. Squaring deviations can also overflow for values around `1e200`. The existing “extreme” fixture only reaches 1000.  
   **Fix:** compute mean and scale using overflow-resistant arithmetic; reject a non-finite fit rather than returning it.

The kernel otherwise matches its twin. Binary eligibility weights make the weight-sum guard correct under the documented 0/1 contract. `eligible_examples` correctly excludes and counts containment work-limit outcomes and preserves identity-unresolved candidates. Platt target smoothing, ECE and Brier formulas are correct for valid probabilities; size-stratum partitioning and all six configuration-key comparisons are implemented.

**PART C — fingerprint supervision**

1. **Major — `src/models/ms2/fingerprint.rs:302`: the fingerprint BCE has the same zero-logit gradient defect as Part B.**  
   For one spectrum, unit weight, zero logits and all-zero targets, loss=`ln(2)`, but every logit gradient is **0 instead of `0.5/1024`**. Zero pooled input also produces zero initial logits because the linear biases start at zero. Thus negative fingerprint bits receive no initial learning signal in this scenario.  
   **Fix:** share a correct stable BCE implementation and test zero logits with both target values, including the bit/spectrum normalization.

2. **Minor — `src/models/ms2/fingerprint.rs:309`: fractional-weight normalization disagrees with its documentation and test reference.**  
   The code divides by `max(1024·weight_sum,1)`, while the documented/reference expression is `1024·max(weight_sum,1)`. With one spectrum, weight=0.5 and zero logits, implementation loss=**0.693147**, while the documented reference gives **0.346574**. Existing fractional-weight tests have total weight greater than one.  
   **Fix:** choose and document the intended weighted-mean rule, implement it consistently, and test positive weight sums below one.

3. **Minor — `src/models/ms2/fingerprint.rs:166`: sidecar matching verifies keys but not parent identity/provenance.**  
   A sidecar with matching ordered keys but different stored SMILES—or an obsolete `source_sha256`—passes. For example, an export row keyed `MOL0001` containing `CCO` accepts a sidecar row with that key containing `CC(=O)O`.  
   **Fix:** compare stored SMILES and verify source hash where source bytes are available. Add same-key/wrong-parent fixtures.

The exporter correctly requests a radius-2, 1024-bit **bit** fingerprint through `GetFingerprint`, not a count fingerprint. Parsing the original SMILES is consistent with its documented aromatic-parent representation. Order is preserved, the head consumes the pooled spectrum vector, and nothing in these components assigns the parent fingerprint to fragments. Thresholded Tanimoto handles empty unions correctly.

**PART D — baseline encoder stacks**

1. **Minor — `src/models/ms2/baselines.rs:103`: the generic Mamba parameter formula counts the wrong output bias width.**  
   Output projection bias has `d_model` entries, not `d_inner`. With V0 dimensions and `bias=true`, the function reports **141,812** parameters per block; the constructed module has **141,684**. V0’s bias-disabled test cannot catch this.  
   **Fix:** add `in_w + ssm.d_model`; compare analytic and live counts across optional configurations.

2. **Minor — `src/models/ms2/baselines.rs:219`: the set stack’s output is permutation-equivariant, not invariant.**  
   Swapping two distinct valid peak embeddings swaps their output rows. `tests/ms2_baselines.rs:270` correctly tests this equivariance despite naming the test “invariance.” Only an invariant pooling operation produces an invariant spectrum vector.  
   **Fix:** correct the behavioral claim and separately test pooled-output invariance.

Padding selection precedes means, normalization and attention projections. Key masking prevents padded keys contributing, and all-padding outputs are selected to exact zero. The supplied tests cover poisoned padding and all-padding rows. V0 parameter matching is valid.

For a fair integration, keep peak selection/order, embeddings, metadata conditioning, output normalization, pooling/memory, decoder, objectives, training budget and candidate/search budgets identical. The parameter-matched forward stack also changes depth; results should identify that alongside removal of bidirectionality.

**PART E — data tools**

1. **Major — `tools/ms2/export_msgym.py:243`: fold-conflict detection is by SMILES, not molecule identity.**  
   Two SMILES variants sharing one identity block, one in `train` and one in `val`, escape conflict detection and enter different parts because `part_of` branches on fold. For example, `CCO` and `OCC` with the same key can enter fit/rank and calibration/report respectively. The advertised molecule-disjoint property is therefore assumed from the input, not enforced.  
   **Fix:** validate fold consistency by normalized identity block before either export or formula-table fitting. Add a cross-fold, same-block variant fixture.

2. **Major — `tools/ms2/formula_table_msgym.py:145`: accepted validation parts cannot supply table rows.**  
   `--split msgym-split-v1 --part calibration` retains validation structures for coverage, but table construction only iterates `train_smiles`. No train identity maps to calibration, so this writes an empty table even when calibration contains valid molecules. `report` behaves likewise.  
   **Fix:** either restrict table sources to fit/rank, or actually construct rows from the requested part. Test every accepted CLI choice and conflicting aliases.

3. **Minor — `tools/ms2/export_msgym.py:242` and `tools/ms2/formula_table_msgym.py:79`: first-block hashing depends on an unstated input-format assumption.**  
   The pinned TSV contains 14-character blocks, so it works there. A full InChIKey supplied through `--tsv` is hashed in full. For example, `VFMQMACUYWGDOJ-0000000000-N` maps to fit while `VFMQMACUYWGDOJ-0000000002-N` maps to rank, despite sharing the first block. Synthetic tests use only 14-character keys.  
   **Fix:** normalize and validate the first block before hashing, grouping and conflict checks.

The modulo rule implements the documented probabilistic 80/20 and 50/50 allocation, not exact finite-sample quotas. Processing is deterministic for fixed input and options. Test-fold chemistry and peaks are skipped; the whole-file provenance hash still includes test bytes, and row-based spectrum IDs change if rows are inserted. The audit correctly computes **molecule-weighted formula overlap over all in-domain distinct SMILES**, not unique-formula overlap or overlap limited to the sampled exports.

**PART F — Python bindings**

1. **Major — `bindings/python/src/ms2.rs:2112` and `bindings/python/python/mamba3_ms2/__init__.pyi:459`: §4.5 API parity remains incomplete.**  
   Python cannot call `Ms2Model.encode` or load a trained standalone model, upload reusable `DeviceSpectra`, or supply resident enumeration artifacts. `ModelConfig` construction also hardcodes `formula_artifacts=None` at line 890. A user can load a trainer checkpoint, but cannot obtain the specified inference model API from it.  
   **Fix:** expose these methods/types and their stubs, with checkpoint-to-inference and resident-input reuse tests. These are documented open items, but they prevent acceptance against §4.5.

2. **Minor — `bindings/python/tests/test_ms2.py:618`: error-parity coverage does not establish the promised class-and-message parity.**  
   Tests cover only some variants, use message fragments or `.*`, and do not compare against Rust-produced messages. A changed JSON error message would pass line 633. Shape, Rank, Autodiff and Backend parity are absent.  
   **Fix:** use Rust-generated expected exception classes/messages for every variant, with a test-only bridge where no public operation can safely trigger one.

Array intake correctly copies logical C-, Fortran- and strided layouts into owned Rust vectors before device operations. No borrowed NumPy buffer or Python API call was found inside detached closures; the captured pyclass data is Rust-owned and protected by the active borrow. Existing Rust errors use the shared mapping. Defaults generally delegate to Rust defaults.

The training parity test uses the same data, seed, learning rate and whole eight-spectrum batch. Rust shuffles its batch order while Python uses fixed order; align that order for reproducibility. It does not establish generation, encoding or resident-input parity.

**Verdicts**

- **A:** accept-with-fixes
- **B:** reject
- **C:** reject
- **D:** accept-with-fixes
- **E:** reject
- **F:** reject