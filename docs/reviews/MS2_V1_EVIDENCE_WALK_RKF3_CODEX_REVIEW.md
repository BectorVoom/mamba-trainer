# Codex review: the explained-peak walk without the carbon digit (task E4) and re-review of fix task RKF3

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree over d39ec34. Verdicts: Part A (E4) accept-with-fixes — no counterexample to the predicate, the single-carbon-count argument, the modular residue or the fast hydrogen range (4,886 endpoint combinations checked independently); one finding: the dispatch budget does not count hydrogen trials, which the slow path multiplies by up to the hydrogen cap; a list of missing boundary tests. Part B (RKF3) reject — the bf16 ranking defect and findings B2 to B7 are resolved; B1 (generator provenance) is still bypassable when the checkpoint records nothing. Fix tasks: E4F, and checkpoint provenance in T1A with the driver rule in T5.

**Part A: accept-with-fixes.** I found no counterexample to the explained-peak predicate, but the dispatch budget substantially understates physical work.

**Part B: reject.** The bf16 ranking defect is fixed, and findings B2–B7 are resolved. B1 remains bypassable despite improved reporting.

Read-only review over HEAD `d39ec34`; no files changed, no cargo run. I excluded the integration edits and inspected only the requested score-buffer declaration in `generate.rs`.

### Part A — finding

**P2 — dispatch budgeting omits hydrogen trials.**  
Locations: [ms2_formula_evidence.rs:596](src/models/ms2/formula_evidence.rs:596), [ms2_formula_evidence.rs:836](src/tensor/ops/ms2_formula_evidence.rs:836), [wrapper:1058](src/tensor/ops/ms2_formula_evidence.rs:1058).

Concrete scenario:

- Candidate `[C100,H200,N7,O7,F7,P3,S0,Cl0,Br0,I0]`, mass `1,837,461,030`.
- Adduct 1, `U=0`, ppm-tenths `100`; 32 distinct peak positions with m/z `50,499,451`.
- Thus `t=50,500,000`, `tol=504`, `J=W=2048`, hydrogen cap `203`.
- No hypothesis accepts; a small read-only calculation checked this. Consequently, early exit never applies.

The fast precondition fails. The lane executes **13,369,344 hydrogen trials**, approximately **40.23 million source-level integer quotients**, although dispatch accounts for only **65,536 peak tests**. The default 4,096-lane launch permits approximately **164.8 billion quotients** for this scenario. These are arithmetic counts, not measured hardware division instructions or a demonstrated timeout.

**Wrong behavior:** a launch admitted under the documented “few divisions per peak test” cost model can perform hundreds of times that work.

**Minimal fix:** include hydrogen-trial work in dispatch sizing and enforce a per-lane work ceiling before launch. If the required predicate must remain available for these inputs, provide a cheaper bounded fallback; otherwise refuse an excessive bucket explicitly. Update the architecture’s cost claim.

### Part A — mathematical checks

The predicate survives the requested attacks:

| Check | Conclusion |
|---|---|
| Hydrogen cap | Both twin and kernel use `min(c[H]+h_pos+2,65535)`, matching `ion_assign`. |
| Empty heavy vector | Excluded by `n1 != 0 || noncarbon_nonzero`; hydrogen alone cannot explain a peak. |
| Scope and uncertainty | Candidate scope uses saturating `tol+U+E_ion`; hypothesis bound uses saturating addition of `U`. |
| Hypothesis bound | Implements exactly `ceil((res′+33h+421)/1000)+U`. |
| Mass overflow | Guarded hydrogen base and bounded carbon sum reject overflowing masses. Reordering nonnegative mass terms does not change whether the total fits. |
| Three-slot ion cap | A closed window of length at most `2m_H` contains at most three integer hydrogen counts. Every accepted mass lies inside that window. |
| Unique carbon count | `delta≤tol≤m_H`, hence `2delta≤2,015,650<12,000,000`. |
| Saturated carbon upper endpoint | Clamping `t+delta` to `MAX` removes only unrepresentable masses. It cannot remove a representable accepted carbon multiple. |
| Carbon lower endpoint | `lhs≥t.saturating_sub(delta)` is exact; `lhs≤top≤MAX` establishes that its sum fits. |
| Modular residue | Residue operands stay below two million; their calculation equals mathematical `(t+tol−m′) mod 1,000,000`, including overflowing `t+tol`. Negative `X` is excluded first. |
| Fast hydrogen range | Under its strict precondition, `7825h+D<1,000,000`, so the residue cannot wrap. No accepted hydrogen lies outside the range. |

I additionally checked **4,886** fast-range endpoint combinations independently, including tolerances `3912/3913`, `h=0/h_cap`, and `D=0/2tol`. The largest fast range has **64 hydrogen values**; the unrestricted `while` loop covers it.

Budget arithmetic is correct: the guarded product decides `J≤W` exactly, including equality. The prefix relation is `V(c[C]+1)−1`, with `V=min(J,W)`. When `c[C]=0,V=1`, that prefix contains zero nonempty heavy vectors and explains nothing. Early success leaves completeness determined by `J≤W`.

The kernel has six array bindings, disjoint output rows, unconditional inner-walk peak/weight loads, and guarded mass products. Its scalar-expanded integer operations agree with the twin for supported counts. This is static agreement; I did not execute CPU/GPU kernels.

### Part A — worst-case cost and tests

For `B=16,M=2048,P=32,W=2048`, default dispatch produces **eight launches of 4,096 lanes**.

Counting source-level quotients, excluding remainder operations and compiler lowering:

- Fast path: at most 64 hydrogen trials per peak test; approximately **17 million quotients per lane**.
- Production count ceiling `H≤1023`: slow path permits 1,027 trials per peak test; a conservative bound is approximately **269 million quotients per lane**, **1.10 trillion per default launch**.
- General `u16` compositions: 65,536 trials per peak test. At most 4,262 hydrogen bases can fit `u32`; the corresponding conservative bound is approximately **5.13 billion quotients per lane**, **21.0 trillion per default launch**.

The tests provide useful independent coverage: exhaustive small-parent comparisons against `EvidenceIndex` and `ion_assign`, budget boundaries against ion prefixes, and poisoned device outputs against twins. **Fast-versus-slow twin equality alone is not independent**, because both share the carbon solver.

Missing targeted cases:

- Actual `t+tol>MAX` and `t+delta>MAX`, with an accepted representable hypothesis.
- Non-carbon mass overflow; non-carbon mass fitting while carbon addition overflows; hydrogen addition overflow.
- Hydrogen cap saturation near `65535`, for both adducts.
- Accepted hypotheses precisely at `residual+bound==tol`, alongside the immediately ambiguous neighbor and residual-ceiling transitions.
- Non-vacuous accepted fast-range endpoint cases at tolerances `3912/3913`; several existing residue-edge cases cannot accept any nonempty heavy vector.
- Hydrogen-only nonempty parent and the `c[C]=0,V=1` zero-length ion prefix.
- Deterministic physical-work coverage for the slow fallback and dispatch sizing.

### Part B — disposition of the prior review

| Prior finding | Disposition |
|---|---|
| Part A finding 1: bf16 trace narrowing | **Resolved.** Gathered scores remain f32; the new producer test uses the exact failing trace values and checks packed fields. |
| B1: generator provenance/leakage | **Partly resolved; still blocking.** Supplied-fit overlap and recorded-identifier checks exist, but missing provenance is accepted. |
| B2: cold scoring reads | **Resolved.** Each actual scoring shape is warmed and read before measuring the repeated call. |
| B3: bootstrap zero | **Resolved.** CLI rejects zero; the helper preserves its measured point. |
| B4: asymmetric NaNs | **Resolved.** Differences are formed only for jointly defined spectra, then averaged by molecule. |
| B5: negative candidates omitted from strata | **Resolved.** Size comes from the candidate trace independently of true-parent replay. |
| B6: search-policy collision | **Resolved for the reported counterexample.** Keys include effective work limits and frozen artifact identities. |
| B7: misleading aggregation metadata | **Resolved.** Pooled AUC/base rate and molecule-averaged precision have separate labels and eligible counts. |

For `E=f32`, the reviewed score-buffer changes preserve the same term bits, sum order, comparisons, and packed values. Neural-dtype reranker scores are explicitly widened with `f32::cast_from` in both ranking and float packing. The workspace estimate now prices scores at four bytes per element.

`trace_atom_count` agrees with device validation **for eligible, device-valid records**: both count every kind-2 action, including the root addition. START, ring closure, and STOP do not add atoms.

### Part B — remaining finding

**P1 — an unrelated supplied fit export still bypasses leakage validation.**  
Locations: [rerank_eval.rs:363](src/models/ms2/rerank_eval.rs:363), [experiment:638](examples/ms2_rerank_experiment.rs:638).

**Failing scenario:** checkpoint G was trained on export A but records neither fit identifier. Pass A as `--report` and an unrelated, disjoint export B as `--generator-fit`. All overlap checks pass, `check_generator_fit` returns `Unrecorded`, and the driver produces report metrics from generator-training data.

**Wrong result:** the supplied-fit disjointness checks cannot establish held-out generator evaluation. Recorded enumeration-fit identifiers also do not establish separate neural-training provenance.

**Minimal fix:** require verifiable generator-training provenance, checked separately from fitted-artifact provenance. Refuse missing provenance for held-out reporting, or require an explicit unverified mode whose output clearly disclaims held-out validation.

The JSON **does disclose** `"supplied by the caller, not recorded in the checkpoint"` at [experiment:1121](examples/ms2_rerank_experiment.rs:1121). That is improved transparency, but it neither closes the bypass nor explicitly states that generator-training leakage remains unchecked.