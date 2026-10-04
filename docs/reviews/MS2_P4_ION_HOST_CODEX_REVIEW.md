# Codex review: host reference of fragment-ion assignment (H4)

Reviewer: codex exec (read-only), 2026-10-04. Verdict: accept-with-fixes; the five findings are fixed in the follow-up.

1. **Major — the per-peak implementation is not u32-only.**  
   **Locations:** `src/models/ms2/ion.rs:190`, `:224`, `:229`, `:299`; claim at `:4` and `:89`.

   **Concrete input:** parent `C1H4`, adduct 1, peak `17_038_576`, uncertainty 50, ppm tenths 100, default limits. Even this ordinary input executes:
   - `u64` addition, division and multiplication in the radix guard (`190–210`);
   - a `u64` visit index and mixed-radix remainder/division (`229–237`);
   - `usize` indexing, enumeration, capacity conversion and length comparison (`156`, `193`, `224`, `302`);
   - `u64` residual and decision-rule additions inside the called `chem::decide` (`chem.rs:497–500`).

   There is no direct `i64` arithmetic in `ion_assign`, but signed residual construction uses `i32` (`308–310`). Consequently, a u32 GPU kernel cannot copy this function line for line as claimed.

   **Fix:** implement the arithmetic lane with u32 radices and indices, explicitly handle `work_max == u32::MAX`, and use overflow-free u32 decision comparisons. Keep vector allocation/indexing in a host wrapper. Otherwise, withdraw the u32-only claim and identify this as a mathematical reference requiring translation.

2. **Major — the report omits the full evaluated-spectrum denominator.**  
   **Locations:** `examples/ms2_ion_report.rs:127`, `:134`, `:245`.

   **Concrete input:** two in-domain spectra, one labeled and one with unknown precision (`U = u32::MAX`). The second receives no targets and disappears before assignment. The report contains only `spectra_labeled = 1`; its unavailable rate cannot reveal the excluded spectrum. Similarly, `--limit-spectra 1` limits labeled spectra, potentially inspecting many more spectra.

   This is explicitly a conditional diagnostic, but contracts §7.1 require conditional metrics alongside the full denominator. The output gives no inspected/unlabeled counts or coverage rate.

   **Fix:** report inspected, labeled, unlabeled and applicable domain counts; include labeled coverage over inspected spectra. Name peak statistics as conditional on labeled spectra. Record the limit and its meaning. Add an explicit `oracle_conditioned: true` field: “true parent formula” is disclosed in source and JSON, but the required oracle-conditioned naming is absent, including from printed aggregate output.

3. **Major — label-cap losses disappear from anchored-retention statistics.**  
   **Locations:** `examples/ms2_ion_report.rs:153`, `:160`, `:205`, `:254`.

   **Concrete input:** one spectrum with 65 distinct recipe-anchor labels on 65 device-kept peaks, with every corresponding hypothesis retained. The label cap keeps 64. The report produces `anchored_peaks = 64`, fully-kept fraction 1, and dropped fraction 0. The omitted anchor is represented only by a spectrum-level overflow boolean.

   These numbers describe retention of **uploaded labels**, although the report describes anchored-peak label retention. A peak whose complete label set is only partly uploaded can likewise be classified fully kept.

   **Fix:** construct the uncapped deduplicated label set for diagnostic denominators. Report label-upload losses separately from hypothesis-cap/search losses. Preserve the existing capped-label statistics under explicitly conditional names, and report the overflow count rather than only its incidence.

4. **Minor — the superset test can silently lose entire shift/adduct categories.**  
   **Locations:** `tests/ms2_ion.rs:463`, `:492`.

   **Concrete failing implementation:** make `embedding_ion` return `None` for every `s != 0`, or for every negative adduct. Those cases are skipped before comparison. The final `checked > 100` assertion still has ample zero-shift/positive-adduct coverage to pass.

   The current fixture sweep does exercise both adducts and shifts −2 through +2; the defect is its regression protection. It also derives the expected mapping through the newly reviewed `embedding_ion` helper.

   **Fix:** independently derive expected composition/mass and whether hydrogen is negative. Assert coverage counts separately for each adduct and shift, and reject unexpected `None`.

5. **Minor — the absent-peak assertion is vacuous.**  
   **Location:** `tests/ms2_ion.rs:724`.

   **Concrete failing implementation:** map an absent anchor to raw index 0 instead of skipping it. The test supplies no deliberately absent anchor, and its assertion merely checks `< 300`, a property already guaranteed for every successful result of `raw_map`.

   **Fix:** add an anchor with an absent peak id and compare the complete output against the unchanged baseline. Also explicitly test negative hydrogen, nonzero anchor shifts and adduct 2 in `ion_labels`; the current label-construction fixture uses zero-shift positive-adduct peaks.

The remaining arithmetic review found no assignment acceptance/rejection defect within the specified searched support. Electron signs and target overflow checks are correct. Mixed-radix ordering and the exhaustion predicate are exact despite the guard continuing after `cut`. The iodine multiplication is safely guarded: `255 × 126_904_472 = 32_360_640_360` is never formed as u32 when it cannot fit. `E_ion` bounds every hypothesis’s arithmetic error; saturating window construction, hydrogen ceil/floor directions and per-hypothesis decisions are correct. Accepted and ambiguous counts remain separate, and the first-J prefix and status bits follow the specification.

Unchecked arithmetic in `ion_assign` includes the constant residual expression (`171`), hydrogen-cap additions (`219`), ceiling increment (`267`), guarded mass products/sums (`254`, `278`), residual multiplication (`287`) and signed residual negation (`310`). Each has a valid bound or preceding guard for these element/adduct tables. I found no reachable overflow in those operations. The u16 hydrogen clamp cannot exclude a representable atomic mass: even hydrogen alone exceeds u32 well before that clamp binds.

Label composition, negative-hydrogen omission, deduplication, sorting and overflow counting are correct. Mask states, partial overlap, unassigned one-hot rows and padding match §2.3.

Evidence bounds, shift sign, statuses and incomplete-support bit are correct. The residual/index tie-break is deterministic and documented in function comments, though absent from the architecture document. Two accepted hypotheses with identical heavy counts cannot arise from this assignment under the permitted tolerance: hydrogen spacing exceeds twice the maximum tolerance.

Padding/all-zero-parent unavailability and the ppm limit agree with the documents. Unknown-adduct handling is reasonable for these internal helpers; request-level metadata statuses remain the caller’s responsibility. Explicit status-0 records are a host-summary extension and must be filtered when producing the specified evidence records. The brute-force oracle independently implements masses, residual bounds, tolerance, electron correction and decisions. The report emits aggregate measurements only.

No files were modified and cargo was not run; this is a source review with independent arithmetic/fixture checks.

**Verdict: accept-with-fixes.**