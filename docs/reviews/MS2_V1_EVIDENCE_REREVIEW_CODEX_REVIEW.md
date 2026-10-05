# Codex re-review: fix tasks E4F (evidence walk) and E3F (evidence integration)

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree over d39ec34. Verdicts: Part A (E4F) accept-with-fixes — the wrapped hydrogen ranges survive 71,265 independently checked endpoint combinations; one finding: in direct shuffled generation the dispatch bound uses a tolerance maximum taken from the batch before the peaks are rotated; some boundary tests still ineffective. Part B (E3F) accept-with-fixes — both earlier findings resolved; one finding: a hidden test hook clones every prepared batch in production; several of the new tests establish less than their names say. Fix task: E5F.

**Part A: accept-with-fixes. Part B: accept-with-fixes.**

Read-only review of the working tree over `d39ec34`. I did not modify files, run Cargo, or execute Rust tests. Enumeration-cache changes were excluded. I used independent Python arithmetic checks; test dispositions below describe what the assertions establish if they pass.

**Part A — E4F**

**P2: Direct shuffled generation supplies an underestimated `tol_max`.**  
Location: [generate.rs:1389](src/models/ms2/generate.rs:1389); sizing consumes it at [ms2_formula_evidence.rs:1170](src/tensor/ops/ms2_formula_evidence.rs:1170).

Concrete failing request:

- Table contains `C20H200`; its mass is `441,565,000`, giving `h_cap_max=203`.
- Both spectra have precursor `442,572,276`, adduct 1, known precursor uncertainty and fragment uncertainty 0.
- Row 0 contains 32 intensity-1 peaks at `50,000,000`, with fragment ppm-tenths **1000**.
- Row 1 contains 32 intensity-1 peaks at `400,499,451`, with fragment ppm-tenths **1**. Peak IDs are distinct.
- Use `ShuffledSpectrum`, at least 32 kept peaks, `formula_evidence_work_max=1`, and dispatch budget **192**.

The original batch’s maximum tolerance is **5000**. After rotation, row 0’s donor peaks retain row 0’s ppm, producing `t=400,500,000`, `tol=40,049`.

Dispatch charges:

```text
trials_bound(203, 5000) = 6
per_lane = 1 × 32 × 6 = 192
```

But each peak executes wrapped ranges **59..=69** and **187..=196**, totaling **21 trials**. The candidate lane executes **672 trials** in a job sized for 192. Acceptance at `(n=17,h=195)` does not shorten the current hydrogen loop.

**Wrong behavior:** the claimed dispatch-work upper bound is false. Explanation results remain unchanged in this example.

**Minimal fix:** compute and retain the tolerance bound from the exact uploaded batch in `DeviceSpectra`, alongside its uncertainty copy, and use that bound at search. Add a shuffled regression with unequal ppm values.

The ordinary trainer path is correct here: `prep.spectra` already contains donor peaks paired with recipient metadata before `max_fragment_tolerance()` runs.

**The wrapped walk survives the arithmetic attacks.**

For an accepted hypothesis, define:

```text
D = t + tol − (m′ + 12,000,000n + 1,007,825h).
```

Acceptance implies `0≤D≤2tol`, and therefore:

```text
X = 1,000,000(12n+h) + 7,825h+D.
```

This establishes the stated wrapped ranges and `s_max`. It also gives **`Q=12n+h+s`**, so an accepted hypothesis cannot require `s>Q`. Extra ranges beyond `Q` cannot pass the exact carbon test.

I independently checked **71,265 endpoint combinations**, including residue boundaries, consecutive wraps, `h=0/h_cap`, and `D=0/2tol`, without finding an omitted hydrogen. The literal `D` endpoints cannot themselves be accepted physical hypotheses because the arithmetic bound is positive; they still correctly exercise the enclosing range argument.

Other conclusions:

- The residue-only modular computation correctly handles mathematical `t+tol` exceeding `u32::MAX`.
- Under the scope gate, `h_cap≤65535` and `tol≤1,007,825`. Thus `s_max≤514`, the hydrogen/tolerance sum is at most **514,827,025**, and `Rm+1,000,000s` fits `u32`.
- The largest wrapped-product estimate is **133,385**. When it exceeds the plain count, the kernel selects the plain range. Actual admitted trials are at most **65,536** per peak/visit.
- Both hydrogen loops and the wrap loop cover their inclusive endpoints without counter overflow.
- The host helpers’ arithmetic fits `u64` even for arbitrary `u32` arguments. The wrapper checks overflow of the subsequent `work_max × P × trials_bound` product.
- Given **both true bounds**, monotonicity makes the dispatch estimate valid. The hydrogen clamp alone cannot compensate for underestimated tolerance.

**Hydrogen-bound provenance is sound for normal production paths.** Table candidates come from table counts; enumeration candidates obey the packed domain hydrogen maximum. Both providers add 3, covering either adduct. Training gold compositions enter conditioning and ion assignment, not the candidate buffer passed to formula evidence. Generation refuses `oracle_formula`. I found no supported production path where the new hydrogen clamp engages.

The hidden slow variants share that clamp, so fast/slow equality alone would not detect a bad provider. Their counters count hydrogen trials; they do not measure GPU instructions or time.

**Disposition of the previous dispatch finding:** substantially fixed, but **not closed** because of the shuffled tolerance counterexample. The wrapper also retains the documented `max(1, …)` exception: a single lane can exceed the configured budget. It does not implement the earlier review’s proposed strict per-lane refusal.

The architecture paragraph at `docs/MS2_V1_ARCHITECTURE.md:318–326` still describes the old fast/plain choice and dispatch denominator without hydrogen trials. Its cost and launch-count claims need updating.

| Previously missing Part A test | Disposition |
|---|---|
| Accepted representable hypothesis with both `t+tol>MAX` and `t+delta>MAX` | **Established:** host test at `tests/ms2_formula_evidence.rs:1918`, with device coverage at `tests/ms2_formula_evidence_kernels.rs:1751`. |
| Non-carbon mass overflow | **Still ineffective:** the N1000 fixture accepts at `d_N=1`, then exits before reaching overflowing `d_N≥307`. Keep another peak unexplained to force those visits. |
| Non-carbon mass fits, carbon addition would overflow | **Still missing targeted coverage.** |
| Hydrogen addition overflow | **Established:** H5000 fast/slow comparison runs the hydrogen ranges beyond the representable mass limit. |
| Cap saturation near 65535, both adducts | **Established for saturation arithmetic and twin/device agreement**, accepting through small hydrogen counts. A hypothesis at hydrogen 65535 cannot have representable mass. |
| `residual+bound==tol` and immediately ambiguous neighbor | **Established:** `tests/ms2_formula_evidence.rs:2031`. |
| Residual-ceiling transitions | **Partial:** asserted arithmetic crosses the ceiling, but changing tolerance from 2 to 0 does not isolate the changed bound. Use the same tolerance, such as 1, on both sides. |
| Non-vacuous accepted endpoints at tolerances 3912/3913 | **Established on host; device explicitly covers the 3913 endpoints.** |
| Hydrogen-only parent; `c[C]=0,V=1` zero-length prefix | **Established on host.** Device explicitly covers hydrogen-only; the exact zero-carbon prefix lacks targeted device coverage. |
| Physical work and dispatch sizing | **Substantive coverage added:** original slow count, wrapped count bounds, launch boundaries and clamp behavior. It misses transformed shuffled tolerances and does not establish a strict dispatch ceiling. |

**Part B — E3F**

**Previous finding 1: resolved.** `DeviceSpectra::upload` copies the uploaded uncertainty, and `evidence_spec` supplies formula evidence, generation ion assignment, training assignment and assignment evaluation. The direct-shuffle tests exercise the original donor/recipient counterexample through production search and full generation.

**Previous finding 2: resolved.** Diagnostics count valid `ev_peaks` independently of candidate support. The empty-support regression establishes the formerly incorrect mean of 0 is now 1.

**P2: The hidden jitter hook copies and retains every production batch.**  
Location: [train.rs:1467](src/models/ms2/train.rs:1467).

Concrete scenario: a normal `Counts` trainer runs a prefix for `B=1024`, `n_raw=512`, without invoking any test hook. The assignment clones and retains the entire `SpectrumBatch`. The three peak arrays alone add **6 MiB** of host storage, plus metadata; every subsequent prefix repeats the allocation/copy.

**Wrong behavior:** test instrumentation adds unconditional production copying and retained memory, including with jitter disabled. It does not change numerical results or device-read counters.

**Minimal fix:** make capture explicitly opt-in, defaulting off, or gate it behind a test-support feature. Capture only when enabled.

The uploaded uncertainty copy itself stays consistent under supported reuse: encoding controls do not rewrite `DeviceSpectra`’s raw peaks, and each production preprocessing call constructs a fresh upload. Resident generation uses that upload for both downstream stages before releasing it; its leased bucket owns the resulting buffers. I found no lease-induced uncertainty mismatch.

`debug_search_buffers` performs no automatic reads or retention. It deliberately returns the **last cached bucket**, which need not be the last-used bucket after a cache hit, and cannot inspect a currently leased bucket. Its fresh-workspace tests are valid; it should not be used as evidence about reused or resident requests.

| Requested integration property | What the new test establishes |
|---|---|
| Historical documents | Literal schema-1/schema-2 configuration defaults and loading/training a **current checkpoint with fields stripped**. It does not establish loading an actual historical checkpoint or Counts parity with HEAD. |
| Production search buffers | **Established for Table and Enumerate fixtures:** actual production buffers are compared with twins. This is stronger than reconstructing the pipeline. |
| Padding with a live branch | **Established:** explicitly nonzero output weights, populated and empty support, exact masked values, and padding-feature perturbation invariance. |
| Leaving the zero point | **Established:** first output-weight movement and later input-weight movement with decay disabled. **Loss attribution is not established:** Evidence gets two additional updates, decoder initialization differs, and the count path remains trainable. Compare identical trained weights with the branch enabled versus ablated. |
| Conditioning parity | Production teacher conditioning is compared in both modes, and generation conditioning is compared at the zero branch. Assignment coverage calls `embed_rows` directly; it does not run the assignment integration. Learned-branch conditioning parity remains untested. |
| Enumeration inference boundary | Gold compositions and parent graphs change while generated outputs remain equal. Labels change from absent to empty; independence from meaningful nonempty labels/targets remains untested. |
| Jitter through a trainer step | **Actual step keying reaches the captured prepared host batch.** Enumeration metadata and peak filtering are then recreated separately. The test does not inspect the trainer’s uploaded metadata, enumeration inputs, residual features or kept peaks. Labels are absent, so nonempty label/target preservation is not established. |
| Diagnostics | Empty-support counts, gold/other fractions, deterministic values and one logical/runtime read are established. Incompleteness is tested only at zero; its nonzero numerator/denominator remains untested. |
| Trained-checkpoint reload | Generation and next-step losses are compared after training. **A trained live evidence branch is not established:** the “moved” assertion accepts initially random `evidence_in` weights, and the fixture gives each spectrum one supported formula, yielding no formula-ranking signal. Require changed/nonzero output weights on competing support. |
| Workspace buckets and estimate | Counts/Evidence generation bucket separation and five buffer-size items are established. Branch activation bytes are checked against a formula; parameter count is checked separately. Training memory, estimate overflow refusal, repeated bucket reuse and resident leases remain untested. |

Driver flag/report behavior and stored-versus-`_jitter2` metrics also remain untested.

These conclusions are static review and independent arithmetic evidence; CPU/GPU execution and Python/Rust API parity were not verified.