# MS2 design review and disposition

Date: 2026-10-02. Review agent: Claude Code 2.1.286 with explicitly requested `claude-opus-5-5`, high effort, tools disabled, and no fallback model. The completed response reports that model and no error. [Invocation metadata](reviews/MS2_SUBSTRUCTURE_OPUS55_METADATA.json) records the review-input hash; [original review](reviews/MS2_SUBSTRUCTURE_OPUS55_REVIEW.md) preserves the response.

The reviewer assessed the initial documents. The revisions below were made afterward by the primary agent, with source checks where recommendations depended on implementation details. This is a design review, not a second independent approval of the revised documents, a benchmark, or implemented work.

Documents: [design](MS2_SUBSTRUCTURE_DESIGN.md), [tasks](MS2_SUBSTRUCTURE_TASKS.md).

## Accepted changes

| Finding | Revision | Verification / task |
|---|---|---|
| Too much engineering before learnability | V0 composed trainable GPU slice before broad search, beam, sparse relations, slots, and custom attention | V0.1-V0.7 and reordered milestones |
| Unspecified graph target and traversal | Versioned target recipe q, canonical traversals, parent containment versus experimental/pseudo fragment labels, explicit loss weighting | P0.7, P1.9, P7.1-P7.2 |
| Redundant/ambiguous H and attachment actions | Joint atom-type convention; residual open valence at STOP; unknown missing-bond partitions remain explicit | P0.8, P1.8 |
| Large beam-state traffic | Independent stratified sampling first; beam is optional; immutable ancestry/finished storage if enabled | P5.6, P5.9-P5.10, O11 |
| Arbitrary T=128 increases submission work | T=2+A+R_max for baseline grammar; A=32/R_max=8 gives 42 tokens, bucket 48 | P0.8, P5.8, P8.10 |
| Missing train/inference mask parity | Exact device versus offline legality bitsets and identical normalization | P1.9, V0.4, P5.3 |
| Missing decoder scan/step parity | Compare parallel teacher-forced logits with recurrent logits, all carries and atom memory | V0.4, P5.2 |
| Graph-state network blocks a simple teacher-forced path | Atom creation-state memory first; incremental graph network optional | P5.2 |
| Formula retained probability undefined | Conditional probabilities with streaming normalizer only over fully scored enumerated support; coverage measured separately | P4.1, P4.3 |
| Formula search can be bounded in memory but slow | Explicit visited/joined/scored work limits; benchmark indexed tables against bounded enumeration | P0.8, P4.9 |
| Small attention matrices may not justify custom kernels | Direct shared per-layer K/V attention first; slots/online softmax only if measured | V0.1, P4.6-P4.8 |
| Sparse backward cannot assume portable float atomics | Source/destination index orientations and gather-based backward | P4.5 |
| Ranking and calibration targets unclear | Separate reranker, formula prior in trace comparison, named containment target, per-size reliability, versioned calibration | P5.5, P6.3, P7.9 |
| Memory estimate omitted large carry from headline number | Explicit current h/last_u/angle accounting, d_inner invariant, sampling/beam budgets | P2.2; checked against source and arithmetic |
| Launch-bound performance not budgeted | Launches per step/call, active-work fraction, CPU submission time, B*K amortization | P8.10; actual bottleneck remains to be measured |
| Complexity in workspace and training loss | Preallocated tensors and one stream first; arenas/leases/contrastive encoder/checkpointing conditional | P2.3, P7.2, P7.5 |
| Missing low-level correctness guards | Poisoned output/padding tests, runtime error checks, stable spectrum RNG IDs, isolated global-counter tests | P2.8-P2.9, P3.4, P5.5, P8.5 |

## Suggestions qualified or not adopted

| Suggestion | Decision and reason |
|---|---|
| Always use u32 microdaltons and delete mass-boundary ambiguity | Add a u32 fast path only with domain/intermediate/error proofs. Widening a search window prevents exclusion but does not prove that every match is inside the original tolerance. Preserve boundary uncertainty and upstream precision limits. Wide arithmetic is conditional rather than a V0 prerequisite. |
| Replace the existing carry with factorized previous inputs in memory estimates | Current `src/ssm/scan.rs` stores last_u as a full tensor the same size as h. Factorization is an optional proven/profiled change (O12), not a baseline capability. |
| Full flip and delta=0 are enough for identity padding | Not accepted without full-cache proof. last_u, angle, and convolution history must also be preserved; the current trapezoidal recurrence can observe altered carry after padding. |
| Ports directly determine hydrogen shifts and ion formulas | Residual valence describes attachment capacity, not a unique ionization/rearrangement mechanism. Require supported parent-to-ion mappings and separate confidence. |
| Use known precursor formula as normal MVP input | The user did not provide molecular formula. V0 ranks a small formula table on-device; known-formula mode remains a clearly labeled diagnostic/upper-bound experiment, never normal inference. |
| Train the reranker on validation data and calibrate there | Use out-of-fold training predictions or a separate ranking-training split; preserve independent calibration and test data to avoid leakage. |
| An OTHER formula logit establishes unsearched posterior mass | An optional supervised out-of-domain/reject head may be calibrated, but it cannot prove how much unenumerated formula probability was recovered. Keep support/exhaustion metadata. |
| Closed-form hydrogen enumeration always replaces tables | Keep as a benchmark alternative; its work can still be large and integer-window completeness needs proof. No universal speed advantage is assumed. |
| Derived action cap guarantees successful chemistry | It guarantees bounded runtime for the defined grammar. Unsupported composition/charge, invalid outputs, and no-valid-action cases still require explicit failure handling. |
| Specific binding defects, dropped launches, Metal bottleneck, MIMO state multiplier asserted from project notes | These statements were not established by the supplied review context. Convert them into tests/profile questions, not claims of existing bugs or measured behavior. Audit actual MIMO state shapes before estimating rank scaling. |

## Concrete cache calculation

For B=8, K=32, Ld=4, h=8, p=32, s=64, FP32, and no convolution, current h occupies 64 MiB, last_u another 64 MiB, and angle 1 MiB. One bank is 129 MiB; two banks are 258 MiB. Full-bank ancestry read/write traffic is 258 MiB per step, approximately 32.25 GiB at 128 steps or 12.09 GiB at 48 steps. These figures exclude weights, other activations, graph state, and runtime overhead. The shape uses d_inner=h*p=256. They are checked arithmetic, not measured speedups.

## Validation performed for this revision

The documentation links, Markdown fences/whitespace, task identifiers, pending statuses, and cache arithmetic are checked. Implementation tasks remain pending. No model training, kernel benchmark, or CPU/GPU implementation test was run for these documentation-only edits. Required future tests are described in the task document.
