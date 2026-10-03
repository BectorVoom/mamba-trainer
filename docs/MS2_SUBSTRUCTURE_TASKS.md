# MS2-to-substructure implementation tasks

Status (2026-10-03): in progress. P0, P1 and P3.1–P3.5 are done; P2 is done except P2.3 and P2.6; V0.1–V0.5 are
done, V0.5 with the results in [V0 results](#v0-results). A checked box has its
deliverable and the evidence named in the [progress log](#progress-log); an unchecked box is not done. Training and
GPU results exist only for the V0 slice on the CPU runtime and the Apple M1 (wgpu/Metal); there is no release-quality
accuracy result.

Design and contracts: [MS2_SUBSTRUCTURE_DESIGN.md](MS2_SUBSTRUCTURE_DESIGN.md).

Review decisions: [MS2_SUBSTRUCTURE_REVIEW.md](MS2_SUBSTRUCTURE_REVIEW.md).

## Execution rules

- Use Rust and CubeCL for the model, training operations, and on-device GPU generation pipeline.
- Read the [test guidelines](test_guidline.md); execute correctness tests on both CPU and an actual GPU and verify Rust/Python parity.
- Use CodeGraph before locating or understanding indexed code. Preserve unrelated work already in the workspace.
- Add independent references before custom fused kernels. Benchmark only kernels that pass correctness gates.
- Use a profiling tool for every performance optimization and retain before/after evidence. A proposed speedup is not a measured result.
- Follow the [manual sources in the design](MS2_SUBSTRUCTURE_DESIGN.md#10-sources-used), checking APIs against the pinned dependency.
- Keep all task boxes unchecked until the listed deliverable and acceptance evidence exist. Record unavailable hardware/data as a limitation, not a passed check.

## Milestones and dependencies

| Phase | Depends on | Deliverable |
|---|---|---|
| P0 | None | Input/chemistry contract and evaluation protocol |
| P1 | P0 | CPU oracle, fixtures, numerical precision rules |
| P2 | P0, precision reference from P1 | GPU workspace and profiling infrastructure |
| P3 | P1, P2 | Device preprocessing and Mamba-3 encoder |
| V0 | P1/P2 core and P3 composed encoder | Small trainable on-device vertical slice and learnability controls |
| P4 | V0 | Production formula hypotheses and evidence; optional sparse/slot experiments |
| P5 | V0, P4 | Full graph decoder; optional beam selection |
| P6 | V0, P5 | Extended GPU validation, graph identity, evidence API |
| P7 | Starts in V0; extended alongside P4 to P6 | Backward coverage, training scale, calibration and checkpoints |
| P8 | P2, integrated baseline | Profile-driven memory and speed improvements |
| P9 | P6 to P8 | Quality evaluation and release evidence |

P1 and P2 infrastructure can progress together after contracts stabilize; precision kernels depend on the independent numerical reference. V0 consumes the minimum decoder/validator/backward portions of P5/P6/P7, not their full implementations. P8 profiling begins immediately; advanced kernel optimization waits for the V0 learnability gate. No exact-identity, general aromaticity, beam-search, or custom-attention project blocks the first trainable model.

Critical path: contract and label audit -> independent oracle -> composed trainable GPU slice -> overfit and molecule-disjoint controls -> bounded production search/evidence -> profile-selected optimization -> full release gates. Optional experiments are not release prerequisites when the simpler path meets the measured quality/performance targets.

## P0 — Freeze contracts and scope

- [x] P0.1 Locate the real `prep_peaks` implementation and FPNet input schema. Document dtype, mass units, intensity transforms, masks, sorting, filtering, and precision loss. Add an original-mass sidecar requirement if necessary.
- [x] P0.2 Define serializable `SpectrumBatch`, `ChemistryDomain`, `ModelConfig`, `GenerationConfig`, and `CandidateBatch` schemas, including units, missing-value semantics, capacities, and version fields.
- [x] P0.3 Define allowed elements, charge/valence states, bonds, attachment ports, adduct transformations, ion/parent distinction, isotope handling, and hydrogen bookkeeping. Record unsupported cases.
- [x] P0.4 Define dataset-level mass accuracy, collision-energy units, and merge policy. Resolve energy count 8 versus 8+. Define behavior when those fields are unknown.
- [x] P0.5 Inventory paired spectra/structures and fragment labels with provenance and licenses. Freeze molecule/scaffold/instrument splits and structural-pretraining leakage checks.
- [x] P0.6 Define the confidence target, substructure-size-aware metrics, abstention behavior, and baseline comparisons. Record initial target devices and user-configurable memory/search limits.
- [x] P0.7 Freeze V0/V1 domains and coverage denominators, the versioned q(target|spectrum) recipe, pseudo-label provenance, evidence anchors, missing-label treatment, canonical traversal, and containment versus ion-assignment metrics.
- [x] P0.8 Set formula-search work counters (visited/joined/scored rows), table bytes, target counts, joint atom-type vocabulary and factor order, and T=2+A+R_max. Define residual attachment valence/unknown bond partitions. Include complete carries and h*p=d_inner in feasibility estimates before choosing B/K.

Acceptance: a valid request can be interpreted without hidden mass/energy conventions; each unsupported or missing input has explicit behavior. No model implementation depends on guessed preprocessing.

## P1 — Independent chemistry and numerical references

- [x] P1.1 Implement CPU decimal/FP64 references for adduct-to-parent mass, ion mass, tolerance calculations, composition conservation, and charge/hydrogen transformations.
- [x] P1.2 Implement the supported graph-action grammar, graph validator, and exact identity reference. Cross-check chemically valid/invalid examples with a selected offline chemistry toolkit.
- [x] P1.3 Specify integer mass scale, quantization-error bounds, supported ranges, limb/native-wide representation, and signed overflow behavior. Validate boundary ambiguity/retry rules.
- [x] P1.4 Build deterministic fixtures for positive/negative adducts, unknown adducts, intrinsic charge, multi-charge/multiplicity where supported, neutral losses, and hydrogen transfers.
- [x] P1.5 Add failure fixtures: non-finite values, negative intensity, zero/empty intensity, empty spectra, invalid IDs, polarity conflicts, missing energy versus measured zero, out-of-domain elements, duplicate peaks, and over-capacity inputs.
- [x] P1.6 Build reference formula tables and deterministic sparse-edge construction with explicit enumeration limits. Test table exhaustion and formula absence separately.
- [x] P1.7 Add graph fixtures covering open ports, rings, symmetric isomers, atom reorderings, duplicate bonds, disconnected graphs, valence errors, incomplete generation, and same-formula distinct structures.
- [x] P1.8 Define separate partial/final graph validity, fixed parent-convention H counts, and STOP-time residual attachment valence. Test ambiguous bond partitions without double-counting hydrogens and ports.
- [x] P1.9 Build canonical action targets and independent per-prefix legality bitsets from the frozen recipe. Require every in-domain training target to replay legally; report dropped-target rates. Cross-check rule-supported parent-to-ion mappings without assuming open valence implies a unique hydrogen shift.

Acceptance: references expose ambiguous/unsupported cases instead of silently coercing them; expected results are independently derived, not copied from GPU outputs.

## P2 — Workspace, device boundary, and profiling

- [x] P2.1 Audit reusable model, scan, matmul, normalization, index, RNG, autograd, and optimizer APIs. Record allocation/read behavior and current MIMO execution paths.
- [x] P2.2 Add a checked memory estimator covering weights, chemistry tables, projections, h/last_u/angle/conv carries, sampling versus optional beam banks, graph state, immutable ancestry/finished results, action/evidence scratch, output, and training state. Reproduce the design's 129/258 MiB cache arithmetic and account for cache-gather bytes per step.
- [ ] P2.3 Implement shape-bucketed preallocated typed tensors with reusable outputs and bounded retention on one stream. Add aligned arenas/concurrent leases only if measured overhead or serving requirements justify them; keep ownership explicit.
- [x] P2.4 Add capability checks for dtype, wide integer arithmetic, plane/shared-memory operations, timestamps, and relevant backend limits. Establish CPU and one real GPU as mandatory initial targets.
- [x] P2.5 Prove whether single-u32 arithmetic meets all V0 mass/intermediate/tolerance bounds. Test exact integer arithmetic plus independent decimal acceptance intervals; add native-wide/two-u32 arithmetic only for a demonstrated domain need. Preserve precision-limit/boundary statuses.
- [ ] P2.6 Add `profile_ms2_substructure` with stage spans, CubeCL `client.profile`, timing-method metadata, synchronized wall-clock measurements, warmup, cold-start reporting, and machine-readable output.
- [x] P2.7 Instrument allocations, launch counts, all device reads, upload/download bytes, logical live bytes, peak live bytes, and reserved bytes. Reuse existing backend counters; cover direct runtime calls too.
- [x] P2.8 Add tests for workspace reuse, alternating buckets, repeated fixed-shape calls, OOM preflight, and alias violations; test concurrent leases when introduced. Poison new-kernel outputs and validate complete logical writes/runtime errors on CPU and GPU.
- [x] P2.9 Isolate process-global counter tests; distinguish production reads from profiling synchronization. Verify warmup settling and at least 200 repeated requests without growth, then alternating-bucket behavior.

Acceptance: the estimator is reconciled with measured allocations; profiler metadata distinguishes device/system timers. Missing measurements are labeled unavailable. Workspace allocation happens outside the decoder hot loop.

## P3 — Device preprocessing and spectrum encoder

- [x] P3.1 Implement K01/K02: device valid-length handling, normalization, sort/select, original-ID mapping, mass features, metadata embedding/modulation, and valid-position reversal.
- [x] P3.2 Verify exact mass decisions use the integer sidecar; neural feature paths use FP32 before optional reduced-precision projection.
- [x] P3.3 Compose bidirectional SISO Mamba-3 blocks with trapezoidal/rotational behavior and context modulation. Reuse existing block/scan implementations where possible.
- [x] P3.4 Test padding independence with NaN/large-value poison, select-based masks, full-cache preservation (including last_u), batch permutation, deterministic ties, independent-spectrum reset, missing metadata, and short/long sequences on CPU/GPU. Delta=0 alone is not a padding proof.
- [x] P3.5 Compare sequence scans with explicitly stepped references, including all recurrent carries and reverse-direction behavior. Test gradients on small cases.
- [ ] P3.6 Save baseline stage profiles for N in {64,128,256,512} and B in {1,8,32}, recording memory-limit exclusions.

Acceptance: encoder outputs and gradients satisfy documented tolerances on both backends, with zero intermediate host reads. Padding cannot influence valid outputs.

## V0 — First trainable GPU vertical slice

- [x] V0.1 Use a small audited domain, N=128, d=128, s=32, two encoder/decoder blocks, F<=4 and K=8. Derive A/R_max/T from trace coverage; use FP32, direct composed attention, and shared per-layer K/V caches.
- [x] V0.2 Rank a bounded fixture formula table on-device; maintain conditional scores and explicit absent-gold/search-exhaustion diagnostics. Oracle-formula mode is separately labeled and excluded from ordinary quality results.
- [x] V0.3 Implement minimum graph actions, partial/final masks, independent stratified sampling, owned recurrent state, and supported-domain GPU validation. No beam gathering, sparse relation network, or general graph canonicalizer is required yet.
- [x] V0.4 Add teacher-forced graph loss and backward propagation immediately. Use identical training/inference masks verified against P1.9 bitsets, creation-state atom memory, and the design's per-spectrum q-weighted loss. Compare parallel/stepped decoder logits and share encoder/KV work across target graphs.
- [x] V0.5 Overfit 32-128 paired examples, then run a molecule-disjoint pilot with metadata-only and shuffled-spectrum controls. Compare against a structure-prior baseline and report statistical uncertainty instead of inventing a success threshold after seeing results.
- [ ] V0.6 Return structural proposals with evidence/identity-resolution statuses; exact action-trace duplicates may be removed, but unresolved graph duplicates remain visible. Confirm all online inference decisions stay on-device.
- [ ] V0.7 Capture end-to-end forward/backward/generation profiles and memory/read/allocation counts. Set pilot quality and hardware-specific latency/memory acceptance targets before production optimization.

Acceptance: a complete trainable CPU/GPU path passes reference/gradient tests and the overfit check; the held-out controls establish whether spectra improve the intended target. Lack of experimental ion labels is reported explicitly. If the pilot fails, investigate representation, labels, and objective before expanding kernels or search complexity. V0 is an internal feasibility milestone, not completion of the final evidence-aware output.

## P4 — Chemical hypotheses and sparse memory

- [ ] P4.1 Implement K03 indexed on-device formula-table joins, exact-mass filters, neural ranking, streaming log-partition when defined, running top-F, and search-exhaustion/error flags. Bound visited/scored candidates as well as scratch memory; report conditional retained probability only for fully scored enumerated sets.
- [ ] P4.2 Add fragment-ion assignment distributions with explicit noise/unassigned state and supported charge/adduct/hydrogen alternatives.
- [ ] P4.3 Test formula recall before and after each pruning stage. Compare mass-boundary decisions and uncertainty flags with P1 references.
- [ ] P4.4 Run the optional sparse-relation quality ablation using composed operations before specializing K05 edge lookup/selection/message passing. If retained, avoid dense N-by-N tensors and test relation-degree/overflow bounds.
- [ ] P4.5 If sparse relations improve quality, implement destination/source index orientations and gather-based forward/backward without requiring float atomics. Budget construction/storage; optional supported atomic variants need deterministic-reference parity.
- [ ] P4.6 If slot compression is retained after P4.8, implement K06 pooling/evidence access with stable masked softmax. Add online-softmax only with measured benefit; define all-masked behavior and verify gradients.
- [ ] P4.7 Share slots, keys/values, and evidence across candidates using spectrum IDs. Assert no physical K-fold replication.
- [ ] P4.8 Profile dense/composed direct or slot attention against custom online-softmax at actual N/r/B; add custom forward/backward only if quality and measured memory/time justify it.
- [ ] P4.9 Compare indexed tables with bounded closed-form-hydrogen enumeration. Verify all integer counts permitted by the tolerance, domain completeness against tiny brute force, deterministic capacity overflow, and visited/scored work. Select by recall, bytes, and measured latency.

Acceptance: outputs expose search truncation and unassigned peaks; formula/sparse workspace remains bounded. Slot compression has a measured quality ablation against direct evidence access.

## P5 — Graph decoding and device candidate selection

- [ ] P5.1 Implement graph state with bounded atom/edge/port capacities, composition budgets, formal charges, hydrogen states, and device-owned alive/finished masks.
- [ ] P5.2 Implement K07 causal decoding, shared direct/optional compact cross-attention, atom creation-state memory, and fixed-order factorized action/pointer heads. Add an incremental graph network only as an ablation. Require teacher-forced versus stepped logit parity covering all carries and graph memory.
- [ ] P5.3 Implement K08 device legality masks and stable conditional log probabilities. Match offline bitsets exactly and normalize training/inference over the same legal support. Handle all-invalid actions without NaNs or fabricated probabilities.
- [ ] P5.4 Extend seeded independent sampling first. Implement top-k/beam as an optional quality/latency experiment; compare conditional field scores and beam selection against exhaustive tiny-graph enumeration.
- [ ] P5.5 Allocate K across formulas as a total budget; include the formula log-prior in trace score comparisons and preserve provenance. Key counter RNG by stable spectrum ID, testing that unrelated batch neighbors do not change a trajectory on the same backend.
- [ ] P5.6 If beam mode is retained, implement K09 ancestry/cache gathering with disjoint banks and all carries. Test duplicated parents, reordered beams, finished beams, graph state, RNG identity, and reset/reuse. Sampling is not required to allocate beam banks.
- [ ] P5.7 Use fixed-step dispatch and absorbing finished masks with zero per-step host reads. Report active/finished work per step; compare optional compaction including extra launches/gathers.
- [ ] P5.8 Test derived T and ring/atom caps against all supported target traces, and preserve explicit failure statuses for unsatisfiable formulas, no-valid-action states, and malformed requests. On a tiny domain, enumerate traces including absorbing failure outcomes, check normalized probabilities, and test fixed-seed sample frequencies and batch independence.
- [ ] P5.9 Verify the owned-cache in-place step against the out-of-place reference, including old last_u/angle reads. Keep explicit extra banks in the budget until in-place correctness is established.
- [ ] P5.10 For optional beam mode, implement immutable time-indexed parent/action records, bounded completed-result storage, and device traceback. Surviving/finished traces must remain correct after live slots are reordered or recycled.

Acceptance: tiny-model candidates and scores match a CPU exhaustive/reference search; beam siblings cannot alias mutable cache state. Completed graphs are distinguished from truncated histories.

## P6 — Validation, identity, evidence, and APIs

- [ ] P6.1 Implement K10 GPU validation for declared connectivity, bond uniqueness, open valence, charge/H, composition, and ring rules. Start with offline-normalized explicit bond orders and exact labeled-graph identity; aromatic/resonance equivalence is a separately tested domain extension.
- [ ] P6.2 Extend V0 trace equality to structural-hash prefilter and bounded exact labeled-graph equality. Test forced collisions and symmetry; distinguish graph validity from identity resolution and preserve candidates when equality is unresolved.
- [ ] P6.3 Add local evidence/mass residuals and an on-device reranker trained on out-of-fold training samples or a distinct ranking-training split. Keep calibration/test data separate and avoid demanding that a subgraph explain all peaks.
- [ ] P6.4 Implement on-device ranking/compaction and packed output with validation, truncation, mass-ambiguity, and deduplication statuses.
- [ ] P6.5 Add one final batched readout per request/device stream; serialize graphs/evidence on the host only after that boundary. Test a fully device-resident output mode as well.
- [ ] P6.6 Expose matched Rust/Python configurations and methods for resident inputs, encoding, generation, evidence, dtype/device errors, and search limits. Validate C/Fortran/non-contiguous array handling and release the GIL around waits only when ownership/lifetimes permit; verify these risks instead of assuming existing defects.
- [ ] P6.7 Verify no CPU toolkit/top-k/isomorphism/early-stop callback is used inside GPU inference. Keep offline toolkit comparison as a test/data-preparation facility.
- [ ] P6.8 Define and validate parent-subgraph-to-ion atom mappings for supported transformations. Do not attach experimental fragment confidence to unanchored proposals or unsupported rearrangements.

Acceptance: validated output conforms to P0 chemistry, and any incomplete guarantees are explicit. Hash collisions never remove distinct candidates. API parity includes error semantics, not only successful inference.

## P7 — Training and persistence

- [ ] P7.1 Extend V0 targets using frozen H/open-valence semantics and size/rarity/anchor sampling. Use canonical traversals initially; random traversal is a later experiment. Keep containment, pseudo-fragment labels, and experimental assignments distinct.
- [ ] P7.2 Implement q-weighted teacher-forced graph loss, defined formula/assignment losses, and evidence loss only where labels support it. V0 needs no contrastive structure encoder; add contrastive loss only after defining the encoder, positives/negatives, and memory cost. Verify accumulation normalization against the mathematical objective.
- [ ] P7.3 Add optional whole-parent fingerprint supervision without treating it as each fragment's fingerprint.
- [ ] P7.4 Add and verify backward implementations for new kernels, including fused/reference output and gradient parity and small finite-difference tests where appropriate.
- [ ] P7.5 Select activation-checkpoint boundaries from measured memory profiles; implement graph-length buckets and correctly weighted microbatch gradient accumulation. Verify RNG/mask replay, shared spectrum-encoding gradients across targets, and full-batch parity.
- [ ] P7.6 Keep gradient checks, clipping, loss scaling, and optimizer updates on-device. Batch metric readout at explicit reporting boundaries.
- [ ] P7.7 Validate FP32 first; then test supported BF16/FP16 modes, finite loss/gradients, overflow handling, and held-out quality. Keep exact-mass decisions independent of neural dtype.
- [ ] P7.8 Save/load weights, preprocessing/chemistry versions, model/search configuration, optimizer state, RNG state, and precision settings. Verify resume behavior and Rust/Python checkpoint interoperability.
- [ ] P7.9 Preserve the V0 learnability controls during scaling. Calibrate held-out outputs after deployed selection/deduplication with a named target; version by domain, K/F, precision, ranking, and search policy.

Acceptance: trainable operations have working gradients on CPU/GPU, checkpoints preserve semantics, and a small overfit fixture succeeds. Full quality claims require real held-out data.

## P8 — Profile-driven optimization experiments

Each experiment records commit/diff, device/backend/driver, configuration, seed, weights, correctness results, cold/warm state, timer type, repeats, p50/p95 wall time, kernel time, launches, reads, transfer bytes, allocation calls, peak/reserved memory, and a conclusion. Store reports under a dedicated benchmark-results directory when implemented.

| Experiment | Change | Required evidence |
|---|---|---|
| O1 | Hoist invariant uploads, batch dispatch and final reads | Transfer/read reduction plus end-to-end timing |
| O2 | Preallocated workspace and shared candidate memory | Allocation counts, peak bytes, no growth across repeated calls |
| O3 | Coalesced SoA/features, valid-length reverse indexing | Memory-traffic attribution or same-traffic reference; parity |
| O4 | Fuse pointwise/projection epilogues and legality/selection | Launch/traffic reduction without register-pressure regression |
| O5 | Tiled sparse gather and online slot attention | Peak-memory reduction and forward/backward parity |
| O6 | Tune GEMM/scan/reduction/top-k geometry and vector width | Per-device/shape winner, tail and capability correctness |
| O7 | MIMO rank 4 versus SISO, optional | Audit actual state/kernel path first; native rank-aware kernel required before claiming MIMO hardware efficiency |
| O8 | Reduced neural precision and checkpointing | Training/inference memory, finite numerics, held-out quality |
| O9 | Device beam compaction or tiled graph comparisons | Improved end-to-end time including extra launches/gathers |
| O10 | Source-versioned compilation/autotune caching | Cold-start improvement, cache invalidation test, identical results |
| O11 | Independent sampling versus optional narrow beam | Matched-quality/compute curves, complete-cache bytes and ancestry traffic |
| O12 | Factorized previous-input carry, optional SISO experiment | Recurrence/rotation/checkpoint parity and measured byte/recompute tradeoff; full last_u remains budgeted until verified |

- [ ] P8.1 Capture the composed baseline before tuning; profile formula search and beam-state movement as separate stages.
- [ ] P8.2 Execute O1/O2 before low-level arithmetic tuning. Re-profile after each accepted change.
- [ ] P8.3 Execute O3 to O6 against the current measured bottleneck. Retain simpler paths where optimized variants regress.
- [ ] P8.4 Evaluate O7 to O9 as algorithmic/precision tradeoffs with quality checks, not automatic wins.
- [ ] P8.5 Implement bounded autotune keys/variants; reset mutable state and poison outputs between trials. Verify each winner on CPU/GPU before caching; reject unsupported configurations before launch.
- [ ] P8.6 Execute O10 with source/dependency/compiler/backend cache invalidation; do not reuse stale compiled kernels.
- [ ] P8.7 Validate fixed-shape and alternating-bucket memory stability, steady-state decoder allocation count, zero intermediate reads, and final batched read count.
- [ ] P8.8 Publish GPU results on available hardware and explicit capability gaps for other advertised targets. CPU-only measurements cannot complete the GPU milestone.
- [ ] P8.9 Execute O11 before investing in beam-specific kernels. Measure fused gather-and-step only if beam quality justifies its memory/traffic; validate complete-cache and history parity.
- [ ] P8.10 Record launches/step and L_call=L_preprocess+L_encoder+L_search+T*L_step+L_finalize. Attribute submission/device idle time; compare B*K amortization and T=domain bound before micro-tuning arithmetic.
- [ ] P8.11 Consider O12 only if measured cache traffic dominates and the factorization is proven for the current formulation. Compare saved bytes with recomputation; do not silently substitute factorized cache estimates.

Acceptance: every retained optimization has correctness and measured benefit at the relevant workload, with tradeoffs documented. No numerical latency/speedup claim is made from asymptotic complexity alone.

## P9 — Evaluation and release gate

- [ ] P9.1 Evaluate molecule-disjoint, scaffold-disjoint, and instrument holdouts; audit training/pretraining leakage and preprocessing statistics.
- [ ] P9.2 Report formula recall at each search/pruning stage, conditional versus full-dataset substructure metrics, size/information-aware precision/coverage at K, validity/identity status, attachment/evidence correctness, calibration, and domain/search abstention. Keep excluded-domain examples in the full-dataset denominator.
- [ ] P9.3 Compare FPNet where available, set encoder, and matched Transformer; ablate formulas, sparse relations, bidirectionality, slots/evidence, MIMO, and precision.
- [ ] P9.4 Report latency-throughput-memory-quality curves across B/N/K/A/T/search budgets; include preprocessing, upload, formula search, decoder, validation, and readout in end-to-end latency.
- [ ] P9.5 Run required CPU and GPU suites and Rust/Python parity checks. Record exact commands and hardware; resolve failures before marking complete.
- [ ] P9.6 Document supported chemistry/dtypes/devices, limits, uncertainty semantics, benchmark reproduction, checkpoints, and example use in both APIs.
- [ ] P9.7 Update the design to match the implemented architecture and measured decisions. Record unresolved domain/precision/performance limitations explicitly.

Release gate: chemically defined outputs, explicit unsupported/ambiguous statuses, trainable and tested kernels, complete device-residency accounting, stable memory, API parity, reproducible GPU profiles, and held-out quality evidence. A model that only emits valid graphs but lacks substructure accuracy evidence is not complete.

## Verification commands

These targets exist. MS2 verification currently runs in a working copy, `/Users/ods/Documents/mamba-trainer-ms2-verify`
(this tree without `target/` and `.git`), because unrelated in-progress Graph Mamba edits in this tree call a
`Var::reverse_bands_ragged` that does not exist yet, so the crate does not compile here; the copy adds
`src/autograd/verify_shim.rs` for that wrapper only, and every verified MS2 file is synced back unchanged.

```sh
# Host references (P0/P1; no device code)
cargo test --release --no-default-features --features cpu --test ms2_chemistry --test ms2_targets --test ms2_contract --test ms2_dataset --test ms2_bounds --test ms2_contain --test ms2_metrics
# Device suites: run each twice, with --features cpu and with --features wgpu (Metal); --test-threads 1 on wgpu
cargo test --release --no-default-features --features wgpu --test ms2_kernels --test ms2_kernel_launches --test ms2_encoder --test ms2_formula --test ms2_decoder --test ms2_generation --test ms2_workspace --test ms2_experiment -- --test-threads 1
# Counter and memory tests, each in its own binary
cargo test --release --no-default-features --features wgpu --test ms2_counters --test ms2_encoder_footprint --test ms2_formula_footprint --test ms2_decoder_footprint --test ms2_generation_footprint --test ms2_footprint -- --test-threads 1 --nocapture
# Profiles, labels and experiments
cargo run --release --no-default-features --features wgpu --example profile_ms2_substructure -- --mode both --n 64,128,256,512 --b 1,8,32 --out <json>
cargo run --release --no-default-features --features cpu --example ms2_label_report -- --input <export.json> --out <json>
cargo run --release --no-default-features --features wgpu --example ms2_experiment -- --train <export> --validation <export> --table <formula_table_v0.json> --control none|shuffled|metadata|prior --steps 6000 --batch 16 --lr 1e-3 --seed 1 --eval-every 1500 --save <ckpt> --out <json>
cargo run --release --no-default-features --features wgpu --example ms2_experiment -- ... --load <ckpt> --diagnose --out <json>
# Python reference and data tools (RDKit environment)
uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/export_casmi.py --data <casmi data> --name <name> --train-molecules N --validation-molecules M
uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/label_specificity.py --export <export.json> --out <json>
```

Not yet present: a Python binding for MS2 (`bindings/python/tests/test_ms2.py`, P6.6) and a CUDA/HIP run. The
export data is CC BY-NC and stays in the CASMI data directory; reports in `bench/results/ms2/` hold aggregates only.
Mark unsupported dtype/backend pairs as explicit capability cases; do not treat skipped required GPU execution as a
passing GPU test.

## Progress log

Evidence for every checked box. Dates are when the evidence was produced.

| Task | Date | Deliverable and evidence |
|---|---|---|
| P0.1 | 2026-10-03 | `prep_peaks` located (`Enveda_CASMI/.../v4b__code__casmi__fpnet6.py:28`) and documented with measured effects: [MS2_CONTRACTS.md](MS2_CONTRACTS.md) §2; the original-mass sidecar is required (§3.1) |
| P0.2 | 2026-10-03 | Schemas: contracts §3, implemented in `src/models/ms2/contract.rs` with `tests/ms2_contract.rs` |
| P0.3 | 2026-10-03 | Chemistry domain: contracts §4, from [casmi_audit.json](../bench/results/ms2/casmi_audit.json); unsupported cases in §4.6 |
| P0.4 | 2026-10-03 | Mass accuracy, energy units and count, merge policy (unsupported in V0): contracts §6. Energy count resolved: 8 means 8 or more |
| P0.5 | 2026-10-03 | Data inventory, terms, frozen splits and the instrument holdout protocol: contracts §1 |
| P0.6 | 2026-10-03 | Confidence target, metrics, abstention, controls, devices and limits: contracts §10 and §3.4 |
| P0.7 | 2026-10-03 | Domains and denominators, recipe `q-cut-v1`, provenance, anchors, missing labels, canonical traversal: contracts §7, measured in [target_pilot.json](../bench/results/ms2/target_pilot.json) |
| P0.8 | 2026-10-03 | Search counters and algorithm, table bytes, vocabulary, factor order, `T = 22`, carry estimates: contracts §9, [formula_table.json](../bench/results/ms2/formula_table.json) |
| P1.1 | 2026-10-03 | `src/models/ms2/chem.rs`; `tests/ms2_chemistry.rs`: element table, decimal parsing, adduct and parent masses, 100 ion cases against exact decimal intervals, tolerance (fixture and a 200,000-pair sweep of the 32-bit form), decision rule |
| P1.2 | 2026-10-03 | `graph.rs`, `grammar.rs`: grammar, validator, canonical trace. Cross-checked with RDKit 2026.03.3 through the fixture ([make_fixtures.py](../tools/ms2/make_fixtures.py)): 536 exhaustive canonical traces, identity partitions of 797 subgraphs, 8 out-of-domain molecules, atom permutations |
| P1.3 | 2026-10-03 | Integer scale, error bounds, ranges and overflow behaviour: contracts §5; checked arithmetic in `chem.rs`; boundary ambiguity cases in the fixture |
| P1.4 | 2026-10-03 | Fixture `tests/fixtures/ms2/chemistry_v0.json` (28 molecules, both adducts, shifts −2 to 2, intrinsic charge and other exclusions as out-of-domain cases); multi-charge and multimers are unsupported by contract |
| P1.5 | 2026-10-03 | `contract.rs` `SpectrumBatch::validate`; `tests/ms2_contract.rs`: one test per failure (non-finite, negative, zero, empty, ids, polarity, unknown versus measured-zero energy, capacity, duplicates, poisoned padding) |
| P1.6 | 2026-10-03 | `formula.rs`: table, window search with work limits, exhaustion distinct from absence, mass overflow, loss edges; `tests/ms2_targets.rs` |
| P1.7 | 2026-10-03 | Graph fixtures: rings up to 4 closures (pyrene, 22-token trace), symmetric and same-formula isomers, reorderings, duplicate bonds, disconnection, valence errors, incomplete traces, 18 invalid traces |
| P1.8 | 2026-10-03 | Partial and final validity, parent hydrogens, residual valence at STOP with unknown partition: contracts §4.5, `TraceState`, the open-valence tests |
| P1.9 | 2026-10-03 | Canonical targets and legality masks against the fixture; on 300 real training spectra the Rust and Python references agree on every field ([Rust](../bench/results/ms2/labels_pilot300_rust.json), [Python](../bench/results/ms2/labels_pilot300_python.json)): 1,984 targets, all replay legally under the parent budget, 0 canonicalization failures, longest trace 22, 82.3% of spectra labeled, mean dropped weight 0.7%. The hydrogen-shift mapping is a mass relation only (contracts §4.3) |
| P2.1 | 2026-10-03 | [MS2_P2_AUDIT.md](MS2_P2_AUDIT.md) |
| P2.7 | 2026-10-03 | `runtime_read_count`, `upload_bytes`, `download_bytes`, `allocation_calls`, `memory_snapshot` in `src/backend.rs`, counted at every direct runtime call; `tests/ms2_counters.rs` (exact deltas, own binary) passes on CPU and wgpu |
| P2.2 | 2026-10-03 | `Ms2MemoryEstimate::{generation, training}` with checked arithmetic, the design's 129/258 MiB cache figure (`tests/ms2_workspace.rs`), a per-step cache-gather item, and retained-activation items derived from shapes. `tests/ms2_footprint.rs`: training in-use peak / estimate = 0.95, 0.95, 0.94 (wgpu) and 1.24 at B = 4, 8, 16 (CPU), band [0.67, 1.5]; generation reserved increase / estimate 1.67. The ~20% CPU residual is unmodelled forward scratch, measured, not a factor |
| P2.4 | 2026-10-03 | `Ms2Capabilities::probe/check` (`workspace.rs`): dtype support, binding limit, plane size, reserved-bytes reporting and the timing method probed from `client.profile` (wgpu/Metal: `DeviceTimestamps`; CPU: `SystemTime`); no wide-integer arithmetic is needed (P2.5); `tests/ms2_workspace.rs`, CPU and wgpu |
| P2.5 | 2026-10-03 | `tests/ms2_bounds.rs`: every bound of architecture §6.4 computed from the domain tables (precursor, adducted parent range, table mass, saturating window bounds, 16-atom ion, tolerance intermediates under checked arithmetic) plus a device `formula_window` edge test; sentinel cases in `tests/ms2_formula.rs`. Single `u32` suffices for V0; CPU and wgpu |
| P2.8 | 2026-10-03 | Workspace reuse, alternating buckets and 200 repeated calls (`tests/ms2_generation_footprint.rs`), memory-limit preflight with unchanged allocation/launch counters, bucket reallocation and alias refusals (`tests/ms2_generation.rs`), poisoned outputs in every kernel test; CPU and wgpu |
| P2.9 | 2026-10-03 | Counter tests each in their own binary (`ms2_counters`, `*_footprint`); warmed calls settle after two warm-ups (1 read per `generate`, 0 per training step, the 36 cold-start reads of the first training step are autotune); profiling syncs are counted apart from production reads in `profile_ms2_substructure` |
| P3.1 | 2026-10-03 | `src/tensor/ops/ms2.rs`: five-kernel peak selection (eligibility, normalisation, top-N, m/z order, raw-index mapping, valid-length reversal), peak and metadata features; `tests/ms2_kernels.rs` against host twins and against the P1 `filter_peaks`, CPU and wgpu |
| P3.2 | 2026-10-03 | Exact-mass decisions only on the integer sidecar; neural features from integers in `f32` (integer-modulo Fourier phase); request validation rejects intensities a fast-math device would treat as non-finite |
| P3.3 | 2026-10-03 | `src/models/ms2/encoder.rs`: two bidirectional blocks of unidirectional `Mamba3Block`s with per-spectrum valid-length reversal and context conditioning |
| P3.4 | 2026-10-03 | `tests/ms2_encoder.rs`: NaN/`u32::MAX`-poisoned padding and extra ineligible peaks give bit-identical outputs; alone-versus-batch and permutation; missing metadata; lengths 1 to 128; selections by `select_valid`. Spectra carry no recurrent state between each other, so there is nothing to reset |
| P3.5 | 2026-10-03 | Block outputs against explicitly stepped forward and reverse references (L = 1, 5, N); contracted-config `apply` versus `step` with every carry compared; finite-difference gradients; CPU and wgpu |
| V0.1 | 2026-10-03 | `ModelConfig::v0()` (N=128, d=128, s=32, two encoder and two decoder blocks, F=4, K=8, A=16, R_max=4, T=22 from trace coverage in contracts §9), FP32, direct composed cross-attention; per-layer K/V computed once per call and held in `DecoderState` |
| V0.2 | 2026-10-03 | `formula_window` → `FormulaHead::score` → `formula_top` on device with work counters, absent-gold slot (`u32::MAX`) and exhaustion status; `top_count` counts written entries only; `oracle_formula` is `Error::Unsupported` in V0; `tests/ms2_formula.rs`, CPU and wgpu |
| V0.3 | 2026-10-03 | `sample_step` (shared `#[cube]` legality helpers, inverse-CDF draws keyed per architecture §3.6), `init_trajectories` (budgets from the integer counts), `validate_trajectories`, `generate.rs` with carry freeze; `tests/ms2_generation.rs` (twin bit-equality, exhaustive 33-trace frequencies on sequential ids within 4σ, stream independence, batch independence, carry freeze, teacher/stepped log-prob agreement), CPU and wgpu |
| V0.4 | 2026-10-03 | `decoder.rs` teacher forcing + `graph_loss` (q-weighted, divisor B), replay masks equal to `TraceState` (P1.9 bitsets) and shared with the sampler, creation-state atom memory, stepped parity within 1e-4, finite-difference gradients on exercised rows; `tests/ms2_decoder.rs`, `tests/ms2_decoder_footprint.rs` (0 reads, constant launches per training step), CPU and wgpu |
| V0.5 | 2026-10-03 | Overfit fixture, 3,524-spectrum pilot and 36,974-spectrum scale run with real, shuffled (molecule-aware donors), metadata-only and structure-prior models on wgpu/M1, bootstrap intervals over molecules; diagnostics and label-specificity check. At scale, real spectra beat every control on held-out NLL with non-overlapping intervals. [V0 results](#v0-results), `bench/results/ms2/v0_*.json` |

## Partially done and open items

- **P2.3** (shape-bucketed workspace): `GenerationWorkspace` and the trainer keep per-bucket buffers with bounded
  retention (at most 4 generation buckets, transparent reallocation on a new bucket); arenas and concurrent leases
  are not built, as the plan allows until measured overhead justifies them. Not checked: no measurement yet shows the
  bucket policy is sufficient for serving.
- **P2.6** (profile driver): `examples/profile_ms2_substructure.rs` records cold and warm wall time, launches, reads,
  bytes, peak reserved bytes, refused configurations and the timing method (wgpu/Metal: `DeviceTimestamps`; CPU:
  `SystemTime`), but per-stage times are synchronised wall clock: CubeCL 0.10's `ComputeClient::profile`
  (`cubecl-runtime-0.10.0/src/client.rs:886`) needs a `Send` closure, the model holds `Rc` handles, and the
  start/end token API exists only on the server trait (`server/base.rs:397,400`). Measured on CPU at N = 128, B = 8:
  generation encoder 48.5 ms, formula search 3.9 ms, decode 221 ms (10.5 ms per step); training forward 92.5 ms,
  backward plus optimizer 171 ms.
- **P3.6** (baseline stage profiles over N ∈ {64, 128, 256, 512}, B ∈ {1, 8, 32}): the driver exists and ran on CPU
  for N ∈ {64, 128}, B ∈ {1, 8}; the full grid on wgpu has not been run.
- **V0.6**: candidates carry validity, truncation, trace-duplicate and request statuses, and generation makes all
  online decisions on the device with one read; there is no evidence status yet.
- **V0.7**: per-call counters exist (a warmed training step: 0 reads, 4,653 launches on wgpu; a warmed `generate`:
  1 read, 2,272 launches, 95 per decode step; reserved bytes flat over 200 calls and alternating buckets; training
  step about 0.29 s and generation about 0.10 s per 8-spectrum call on M1), but the wgpu profile grid and the
  latency/memory targets are not set.
- **Reviews**: V0-C (sampler, generation) and V0-D to V0-G (evaluation, trainer, diagnostics) have not had a codex
  review.

## V0 results

All runs on the Apple M1 (wgpu/Metal), V0 model config, AdamW lr 1e-3, batch 16, seed 1, the same step budget for
every model of a comparison. Intervals are 95% bootstrap intervals over molecules; metrics are conditional on
in-domain molecules (the exports sample in-domain structures only). Data: `export_casmi.py` exports with
`uniform-in-domain-v2` sampling (molecules uniform, then each molecule's spectra uniform over its in-domain rows; the
first exporter took each molecule's first rows in file order and was replaced, contracts §7.1).

### Overfit fixture (V0.5, part 1)

128 labeled spectra of 128 molecules (`overfit_train.json`), 3,000 steps;
[none](../bench/results/ms2/v0_overfit128_none.json), [shuffled](../bench/results/ms2/v0_overfit128_shuffled.json).
Criteria of architecture §7, declared before the run. Teacher-forced NLL per token 4.73 → 0.123 [0.112, 0.135],
2.6% of the initial value: **met**. Coverage above the shuffled control: **not met** — 0.706 [0.666, 0.748] versus
0.697 [0.651, 0.742]; precision 0.502 versus 0.472. The decoder conditions on the formula, which comes from the
precursor mass the control keeps, and among 128 molecules with one spectrum each the formula identifies the
molecule, so this fixture cannot show spectrum dependence. (Its shuffled control is valid: one spectrum per
molecule, so in-batch rotation never pairs siblings.)

### Pilot, 3,524 training spectra (V0.5, part 2)

Train: 3,524 labeled spectra of 1,907 molecules; validation: 740 spectra of 379 held-out molecules
(`pilot_validation.json`); 6,000 steps (about 27 epochs). Reports: `bench/results/ms2/v0_pilot_*.json`, diagnostics
`v0_pilot_*_diagnose.json`.

| Model | Validation NLL/token | Train NLL/token | Donor − own peaks, validation | Coverage at K=8 | Precision |
|---|---|---|---|---|---|
| Real spectra | 2.318 [2.225, 2.417] | 0.349 | +0.129 [0.089, 0.171] | 0.0060 [0.0020, 0.0109] | 0.0067 |
| Shuffled (donor-evaluated) | 2.317 [2.222, 2.414] | 0.401 | −0.001 [−0.007, 0.005] | 0.0016 [0.0003, 0.0033] | 0.0081 |
| Metadata only | 2.217 [2.115, 2.323] | 0.577 | 0 (exact) | 0.0005 [0.0001, 0.0011] | 0.0067 |
| Structure prior | 2.007 [1.916, 2.106] | 0.682 | 0 (exact) | 0.0001 [0.0000, 0.0003] | 0.0045 |

Declared primary comparison: real spectra do not lower the held-out NLL below any control, at the final step or at
any of the four evaluations (every model is best at step 1,500). Diagnosis:

- **The shuffled control was first evaluated wrongly.** `batch::rotate_peaks` gives row `b` the peaks of row
  `(b + 1) % B` with no molecule check, and validation spectra are evaluated in molecule order, so the "shuffled"
  model mostly saw its sibling spectrum (found by the codex review of the follow-up proposal, verified in the code).
  V0-G added molecule-aware donors (`ExperimentSet::donor_map`); the table shows the re-evaluated control.
- **The peak path is connected and used**: only the real model reacts to donor peaks (mostly atom type +0.112 and
  STOP +0.083); the two NLL recomputations agree to 1e-6, so the metric is computed as specified.
- **Overfitting dominates**: every model fits its training spectra far better than held-out ones — including the
  metadata-only model, which sees no peaks and identifies training molecules from the high-frequency precursor
  features — and the model with the least identifying input, the structure prior, has the best validation NLL.
- **The labels are spectrum-specific** (`tools/ms2/label_specificity.py`,
  [result](../bench/results/ms2/label_specificity_pilot_validation.json), all 740 validation spectra): own peaks
  explain 51.7% of the intensity against 5.7% for another molecule's peaks in the same precursor window and 0.2% for
  random peaks; the retained `q` overlaps a sibling spectrum's by 0.56 against 0.054 and 0.036. The labels choose
  about 10 of about 52 candidate graphs per molecule by exact mass, so the objective is learnable from the spectrum.

A first reading attributed the failure to the model's inability to extract exact-mass matches and proposed
[decoder mass evidence](MS2_V0_MASS_EVIDENCE.md); the review and the diagnostics put overfitting first, and the
proposal is on hold. A V0-G bug evaluated the blinded models' donor inputs with their peak path switched on; it was
fixed in `train.rs` with the regression test `donor_peaks_keep_each_models_own_blinding`, and the numbers above come
from the fixed code.

### Scale experiment, 36,974 training spectra (V0.5, part 3)

Same protocol and the same `pilot_validation.json`, training on `scale20k_train.json` (36,974 spectra of 19,114
molecules; 6,000 steps of 16 is about 2.7 epochs over its labeled spectra, against about 27 in the pilot).
Reports: `bench/results/ms2/v0_scale20k_*.json`, diagnostics `v0_scale20k_*_diagnose.json`.

| Model | Validation NLL/token | Train NLL/token | Donor − own peaks, validation | Coverage at K=8 | Precision |
|---|---|---|---|---|---|
| Real spectra | **1.144** [1.107, 1.180] | 1.032 | **+0.269** [0.249, 0.288] | **0.0124** [0.0056, 0.0211] | 0.0113 [0.0080, 0.0155] |
| Shuffled | 1.239 [1.203, 1.277] | 1.152 | −0.000 [−0.001, 0.001] | 0.0001 [0.0000, 0.0003] | 0.0032 |
| Metadata only | 1.251 [1.212, 1.291] | 1.152 | 0 (exact) | 0.0006 [0.0002, 0.0012] | 0.0069 |
| Structure prior | 1.241 [1.206, 1.278] | 1.142 | 0 (exact) | 0.0005 [0.0000, 0.0015] | 0.0064 |

With ten times the data the train/validation gap closes (1.03 versus 1.14), every validation curve is still falling
at step 6,000, and the declared primary comparison now favours real spectra: their held-out NLL per token is
lower than each control's with non-overlapping intervals (1.144 against 1.239 shuffled, 1.251 metadata-only and
1.241 structure prior), and only the real model reacts to donor peaks (+0.269, mostly atom type +0.236 and STOP
+0.208). Secondary: coverage 0.0124 against at most 0.0006 for the controls, precision 0.011 against at most 0.007 —
spectrum-dependent, but about 1% in absolute terms, so generation quality is far from useful.

Provenance of this result: the protocol (models, controls, budget, seed, validation set, metrics) is the one
declared in architecture §7, but the training-set size was changed after the 3,524-spectrum pilot failed and its
diagnostics pointed at overfitting; the 3,524-spectrum result stands as reported. V0.5 is checked because the
deliverable — overfit check, molecule-disjoint pilot, all controls, structure prior, intervals — exists and the
held-out controls now establish that spectra improve the intended target at this scale.

Next, following V0's acceptance text: train longer and on more data (the curves have not converged), then decide on
the [mass-evidence proposal](MS2_V0_MASS_EVIDENCE.md) with the evidence-only ablation the review asked for, and set
the V0.7 latency/memory baselines from the wgpu profile grid (P3.6).

## Review history

P1 was reviewed by codex together with the contracts (second review); its findings were fixed and re-verified,
and the fixed code has not been re-reviewed. Host tests: `cargo test --release --no-default-features --features cpu
--test ms2_chemistry --test ms2_targets --test ms2_contract --test ms2_dataset` (68 tests). P1 is host-only code,
so there is no GPU run for it.

P0 was reviewed twice by codex ([first](reviews/MS2_CONTRACTS_CODEX_REVIEW.md), [second](reviews/MS2_CONTRACTS_CODEX_REVIEW_2.md));
the contracts are at revision 3, which answers the second review and has not itself been re-reviewed.
The V0 implementation specification is [MS2_V0_ARCHITECTURE.md](MS2_V0_ARCHITECTURE.md)
([review](reviews/MS2_V0_ARCHITECTURE_CODEX_REVIEW.md)).

V0-B (decoder, grammar replay) was reviewed by codex: no blocker; its five real findings (formula `top_count` with
non-finite scores, unchecked `q` in `TargetBatch`, replay limit validation, vacuous finite-difference entries,
undefined distribution rows) were fixed and tested. The supervisor's check of V0-C found three defects, fixed and
tested: draws keyed by `trajectory ^ id_lo` shared random streams across sequential spectrum ids (architecture §3.6
records the corrected keying), copied hash functions, and element budgets recovered from float features; V0-B's
`step_logits` read the token to the host every step, rebuilt atom memory with O(rows²) launches and recomputed
cross-attention keys and values per step, also fixed. The [mass-evidence proposal](MS2_V0_MASS_EVIDENCE.md) was
reviewed by codex; its findings (the sibling-spectrum control, the missing diagnostics, per-step replay snapshots,
the permissive shift bound and peak cap, the oracle check's expected value) are recorded there and led to the
diagnostics above. Review notes are kept outside the repository in the session working directory.
