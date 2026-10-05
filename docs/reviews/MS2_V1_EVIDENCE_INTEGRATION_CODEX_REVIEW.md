# Codex review: formula-evidence kernel fixes (re-review) and the Evidence feature layout integration (task E3)

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree over d39ec34. Verdicts: Part A (kernel fixes) accept-with-fixes — all four earlier findings closed, some boundary tests still missing or ineffective; Part B (integration) reject — two findings (donor peaks paired with the recipient m/z uncertainty in direct shuffled generation; one diagnostic denominator) and a list of untested properties. Fix task: E3F.

**Part A: accept-with-fixes. Part B: reject.**

Read-only review against HEAD `d39ec34` and the current §1.6. I did not modify files, run cargo, or execute tests. Concurrent ranking/packing and reranker edits are excluded.

**Part A — earlier findings**

| Earlier finding | Disposition |
|---|---|
| **1. `N=0` out-of-bounds normalization loads** | **Fixed.** The device wrapper refuses `N=0` before launch at [ms2_formula_evidence.rs:473](src/tensor/ops/ms2_formula_evidence.rs:473). The host twin asserts the same precondition at [formula_evidence.rs:210](src/models/ms2/formula_evidence.rs:210). This matches the updated spec. A device test checks refusal and zero launches. |
| **2. Selection omits `E_ion`** | **Closed by the explicit specification decision.** Selection tests `tol_p + U`; kernel 2 tests `tol_p + U + E_ion(c)`. A selected peak outside a candidate’s scope remains in the evidence denominator and is unexplained. Host and device regression cases now establish that behavior. |
| **3. Wrapping intermediate subtractions** | **Declined; no numerical defect demonstrated.** The wraps cannot change the final result in the supported input domain, for the reasons below. The requested guarded-operation form remains unimplemented, but I would not retain this as a result-correctness finding. |
| **4. Conditional `ev_w` load** | **Fixed.** The weight is loaded unconditionally before the mask-bit branch at [ms2_formula_evidence.rs:924](src/tensor/ops/ms2_formula_evidence.rs:924); the twin mirrors it. |

For finding 3:

- `tolerance-r` contributes only when `r <= tolerance`; `r-tolerance` contributes only when `r > tolerance`.
- A wrapped lower-window subtraction is replaced with zero.
- `lo_w-hm` contributes to `h_lo` only when `hm < lo_w`.
- A wrapped `hi_w-hm` occurs only when `in_hi=false`; consequently `has=false`, and no hydrogen hypothesis is evaluated.
- `h_hi-h_lo+1` contributes only when `h_lo <= h_hi`.

The discarded calculations contain no global-memory access or division by a possibly zero quantity. Thus the wraps described in the declined comment do not alter an explained bit, weight, or completeness result.

I found **no new numerical kernel defect** for selector-produced rows and supported composition counts.

The independent comparison is substantive: [kernel_twin_agrees_with_the_reference_index:828](tests/ms2_formula_evidence_ref.rs:828) compares the twin against an independently accumulated, mass-sorted index with binary search and guarded hydrogen intervals. It requires both explained and unexplained cases. Its `W=2^20` completes these small parents, so it establishes exhaustive explanation equivalence, **not truncated-prefix equivalence**.

**Disposition of the earlier missing-test list**

| Area | Added coverage | Still missing or ineffective |
|---|---|---|
| Selection | `N=0` refusal; revised `E_ion` scope example; positive-adduct target overflow and negative-adduct underflow; saturated selection width; zero intensity sum; order-sensitive normalization. | The normalization test checks the large peak’s weight bits, rather than bit equality of every resulting weight. |
| Evidence | Device budgets below/equal/above full visits; `half_p=m_H` and `m_H+1`; both adduct hydrogen caps; flag-1 zero-heavy candidate; empty evidence; slot 31; early explanation with `complete=0`. | Heavy-mass overflow; hydrogen-mass overflow; radix products at and above `2^32`; unknown `U` with **valid evidence rows**. The current unknown-`U` test also has empty evidence, so the walk is skipped. |
| Three-hydrogen interval | A test was added. | **The fixture does not exercise it.** [ms2_formula_evidence_kernels.rs:733](tests/ms2_formula_evidence_kernels.rs:733) calculates an interval at `hm=0`, described as visit 2 of C1. C1 has only visit 1, with `hm=12,000,000`; visit 2 does not exist, and the empty heavy vector is excluded. Use an actually visited nonempty heavy vector and center the window on its mass plus hydrogen mass. |
| Features | Productive residuals for both adducts and signs; unknown uncertainty/adduct; both parent-overflow directions; `w=0`; ordinary large-width clamping; counts 0, 1, 1023; count-feature comparisons using `to_bits()`. | Saturated residual width and the `4w` overflow boundary are not exercised on device. |
| Dispatch/output | Empty batch/candidate shapes; `P=0`/33 refusals; exact launch counts including a tail; chunk equality across spectrum boundaries. | Address-domain rejection remains untested. General kernel/twin float comparisons remain tolerance comparisons. |

These remaining boundary-test gaps are why Part A is **accept-with-fixes**, rather than an unconditional accept.

**Part B — findings**

1. **P1: Direct shuffled generation pairs donor peaks with recipient m/z uncertainty.**

   Location: [generate.rs:964](src/models/ms2/generate.rs:964), combined with [generate.rs:915](src/models/ms2/generate.rs:915).

   **Failing scenario:** Two spectra, `Control::ShuffledSpectrum`, `FormulaFeatures::Evidence`. Spectrum 0 has precursor `61,007,276`, adduct 1, known precursor uncertainty, and fragment uncertainty `u32::MAX`. Spectrum 1 has fragment uncertainty 0 and one intensity-1 peak at `59,999,451`. Fragment tolerance is 100 ppm-tenths; the table contains C5; `W >= 5`.

   Rotation uploads spectrum 1’s peak and uncertainty into device row 0. But the evidence helper constructs `spec` from the original host batch, giving row 0 `U=u32::MAX`.

   **Wrong result:** Row 0 selects no evidence; C5 receives `cand_ev=[0,0,0,1]`. With the donor’s uncertainty, the peak target is exactly `60,000,000`, tolerance 599, and C5 should receive `[1,1,1,1]`.

   The peaks themselves follow the donor correctly. The mismatch is the accompanying uncertainty. **Trainer training and trainer evaluation avoid this defect**, because they assemble a donor batch and suppress a second rotation.

   **Minimal fix:** Carry the uploaded/rotated m/z uncertainty with `DeviceSpectra`, or consistently pass the transformed host batch to downstream stages. Use that same source for both formula evidence and `generate_ion`; the ion stage has the same original-host-batch construction.

2. **P2: `evidence_peaks_mean` undercounts spectra with empty formula support.**

   Location: [train.rs:2394](src/models/ms2/train.rs:2394).

   **Failing scenario:** One spectrum with precursor `101,007,276`, adduct 1, known uncertainties, and one eligible intensity-1 peak at `59,999,451`. Use a table containing only C5 and a narrow precursor window, so `rows_scored=0`. Evidence selection still produces one valid evidence peak.

   **Wrong result:** Diagnostics substitute `nev=0` because no candidate is scored, and report `evidence_peaks_mean=0`. The documented mean over examined spectra is 1.

   The other denominators are correct: incompleteness uses scored slots; gold explanation uses spectra whose gold composition is scored; other explanation uses the other scored slots of those spectra.

   **Minimal fix:** Read `ev_peaks` in the existing batched diagnostic read and count valid flags per spectrum independently of candidate support. Keep the existing denominators for the candidate-based diagnostics.

**Remaining integration checks**

- **Counts compatibility:** The E3 changes preserve count-head parameters, names, initialization order, scoring operations and the count-feature launch. Evidence buffers are `None`. Missing serialized fields default to `Counts` and the historical work/jitter defaults. Earlier documents/checkpoints should therefore retain their layout by inspection; historical checkpoint loading is not actually tested here.
- **Search ordering:** Both production sources finish `cand` first—`formula_gather` for Table, `fill` then `cand_pad` for Enumerate—before evidence. Slices correctly select columns `0..10` and `10..16`. Outside the direct-shuffle finding, `spec=[U,0]` matches the ion stage.
- **Padding:** Kernels write exact zero features for padding. The evidence branch is added **before** masking. Its learned biases cannot preserve a padding contribution in scored support. The empty-support slot-0 fallback remains excluded by the reported candidate mask/top selection.
- **Head and gradients:** The implementation is the specified count embedding dot query plus `Linear(32→1)(SiLU(Linear(6→32)))`. Features are `Var::constant`. Output weight and bias start at zero.
  
  This is **not a dead start**: the output-weight gradient is a weighted sum of hidden activations and can be nonzero while the output weight is zero. `evidence_in` receives zero loss gradient until the output weight becomes nonzero. Nonzero hidden activations alone do not guarantee learning: identical activations across candidates cancel under softmax, and the shared output bias has no ranking signal.
- **Conditioning:** Generation gathers `scored.embedding`, which contains only count features. Composition teacher forcing calls the same row network on gold count features. `ScoredRowOrZero` also gathers that count-only embedding. Assignment reuses the count row network.
- **Inference boundary:** Evidence generation consumes spectra, formula artifacts and configuration, without gold compositions, targets or labels.
- **Jitter:** Host mutation precedes upload and all request-derived precursor computations. Training uses `(seed, 1+steps, set index)`; evaluation uses `(seed, 0, set index)`. Zero sigma is identity; the draw is rejection-truncated at three sigma. Enumeration/window construction, residuals, device peak filtering and metadata all receive the jittered precursor.
  
  Labels are deliberately frozen from the stored spectrum: their original preparation uses the stored precursor’s peak filter, but the recipe is not rerun during a step or jittered evaluation. Gold counts and targets remain unchanged. One ancillary calculation, `donor_stats`, runs before training jitter; its stored-precursor eligibility count is stale during that training preparation, but it is not exposed by the training loss report.
- **Evaluation driver:** Stored metrics use the original set. `_jitter2` passes use the cloned set jittered at sigma 2 with the checkpoint’s effective seed and fixed tag 0, independent of batch order. Donor assembly retains the recipient’s jittered precursor.
- **Reads:** The evidence path adds no device reads. Ordinary generation retains its final batched read; warmed non-report steps retain zero reads. Diagnostics explicitly read at evaluation.
- **Memory/preflight:** Estimates include evidence peaks, weights, candidate evidence, 16-wide features, the six-wide slice, branch activations and 257 branch parameters. Evidence work uses checked multiplication in `u64`; it cannot silently overflow. Counts evidence items are zero.
- **Checkpoints/flags:** Evidence parameters participate in save/load. Explicit layout mismatch returns `Error::Config` naming `formula_features`. All three requested driver flags are parsed and threaded.

**What the integration tests actually establish**

The file covers buffer presence and parameter names; manually assembled Table/Enumerate evidence pipelines against twins; approximate zero-output-branch equality; branch gradients after manually activating the output layer; a Table gold-composition inference-boundary check; one-read generation for both sources; zero-read warmed Enumerate training; jitter helper keying and donor assembly; version-1 config defaults; fresh Evidence weight round-trip; mismatch refusal in both directions; and overall training-loss decrease.

The following cases are **not established**:

- Historical Counts documents/checkpoints with all new fields absent, including earlier schema-2 documents.
- Counts bit equality against the previous implementation; the launch pin covers Table only.
- Production search-buffer correctness for both sources—the twin comparisons reconstruct the pipeline themselves.
- End-to-end donor **evidence** in training/evaluation and direct generation, with differing or unknown donor uncertainties.
- Padding masking with a **nonzero learned branch**, including empty support.
- A first optimizer update from zero output weights, followed by nonzero `evidence_in` gradients. The current gradient test manually activates the branch; overall loss decrease could come from the count path.
- Decoder/assignment embedding parity under Evidence in both teacher-conditioning modes.
- Enumerate inference independence from gold, and independence from changed labels/targets.
- Actual trainer step jitter keying and propagation through enumeration, residuals, metadata and a peak at the precursor-plus-2-Da boundary.
- Preservation of real nonempty labels/target batches under jitter.
- Diagnostic values, denominators, empty-support peak counts and read count.
- Driver flag/report behavior and stored-versus-`_jitter2` metrics.
- Workspace layout separation, exact estimate items, training memory, and overflow refusal.
- A trained Evidence checkpoint’s behavior after reload.

The appended footprint test checks a broad generation reserved-memory ratio when that measurement is available; it does not validate individual estimate items or training memory. All execution outcomes remain unverified in this review.