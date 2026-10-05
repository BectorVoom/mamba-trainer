# MS2-to-substructure implementation tasks

Status (2026-10-04, 16:20): in progress. P0, P1, P3 and V0 are done; P2 is done except P2.3 (the decode loop
still allocates). Of the later phases, P5.1–P5.3, P5.8, P6.1 and P8.1 are done; everything else in P4 to P9 is
unchecked. Most of the remaining P4 to P7 code now exists and is wired into generation and training — the
enumerating formula source, trajectory allocation, graph identity, ranked and packed output, the ion-assignment
head with its loss and evidence — with the reranker, calibration, fingerprint head and baseline encoders as
standalone modules. What keeps the boxes open is listed item by item in
[Partially done and open items](#partially-done-and-open-items): GPU runs of the latest changes, review findings
not yet re-reviewed, and experiments not yet run. V0.5 has its results in [V0 results](#v0-results) and
[MassSpecGym results](#massspecgym-results-linux-radeon-860m), V0.7 its
[baselines and targets](#baselines-and-targets-v07).

A checked box has its deliverable and the evidence named in the [progress log](#progress-log); an unchecked box is
not done. GPU training results exist for the V0 slice, for conditioning on the parent composition and for the
enumerating formula source (held-out formula recall at F = 4 of 0.48 against 0.23 with the train-formula table),
on a Radeon 860M (wgpu/Vulkan, Linux) and, for V0, an Apple M1 (wgpu/Metal); there is no release-quality accuracy
result: held-out coverage and precision of the generated candidates are a few percent, and a shuffled-spectrum
control shows the formula ranking does not yet use the peaks (0.45 without them).

V1 design and its reviews: [MS2_V1_ARCHITECTURE.md](MS2_V1_ARCHITECTURE.md),
[reviews](reviews/MS2_V1_ARCHITECTURE_CODEX_REVIEW.md).

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
- [x] P2.6 Add `profile_ms2_substructure` with stage spans, CubeCL `client.profile`, timing-method metadata, synchronized wall-clock measurements, warmup, cold-start reporting, and machine-readable output.
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
- [x] P3.6 Save baseline stage profiles for N in {64,128,256,512} and B in {1,8,32}, recording memory-limit exclusions.

Acceptance: encoder outputs and gradients satisfy documented tolerances on both backends, with zero intermediate host reads. Padding cannot influence valid outputs.

## V0 — First trainable GPU vertical slice

- [x] V0.1 Use a small audited domain, N=128, d=128, s=32, two encoder/decoder blocks, F<=4 and K=8. Derive A/R_max/T from trace coverage; use FP32, direct composed attention, and shared per-layer K/V caches.
- [x] V0.2 Rank a bounded fixture formula table on-device; maintain conditional scores and explicit absent-gold/search-exhaustion diagnostics. Oracle-formula mode is separately labeled and excluded from ordinary quality results.
- [x] V0.3 Implement minimum graph actions, partial/final masks, independent stratified sampling, owned recurrent state, and supported-domain GPU validation. No beam gathering, sparse relation network, or general graph canonicalizer is required yet.
- [x] V0.4 Add teacher-forced graph loss and backward propagation immediately. Use identical training/inference masks verified against P1.9 bitsets, creation-state atom memory, and the design's per-spectrum q-weighted loss. Compare parallel/stepped decoder logits and share encoder/KV work across target graphs.
- [x] V0.5 Overfit 32-128 paired examples, then run a molecule-disjoint pilot with metadata-only and shuffled-spectrum controls. Compare against a structure-prior baseline and report statistical uncertainty instead of inventing a success threshold after seeing results.
- [x] V0.6 Return structural proposals with evidence/identity-resolution statuses; exact action-trace duplicates may be removed, but unresolved graph duplicates remain visible. Confirm all online inference decisions stay on-device.
- [x] V0.7 Capture end-to-end forward/backward/generation profiles and memory/read/allocation counts. Set pilot quality and hardware-specific latency/memory acceptance targets before production optimization.

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

- [x] P5.1 Implement graph state with bounded atom/edge/port capacities, composition budgets, formal charges, hydrogen states, and device-owned alive/finished masks.
- [x] P5.2 Implement K07 causal decoding, shared direct/optional compact cross-attention, atom creation-state memory, and fixed-order factorized action/pointer heads. Add an incremental graph network only as an ablation. Require teacher-forced versus stepped logit parity covering all carries and graph memory.
- [x] P5.3 Implement K08 device legality masks and stable conditional log probabilities. Match offline bitsets exactly and normalize training/inference over the same legal support. Handle all-invalid actions without NaNs or fabricated probabilities.
- [ ] P5.4 Extend seeded independent sampling first. Implement top-k/beam as an optional quality/latency experiment; compare conditional field scores and beam selection against exhaustive tiny-graph enumeration.
- [ ] P5.5 Allocate K across formulas as a total budget; include the formula log-prior in trace score comparisons and preserve provenance. Key counter RNG by stable spectrum ID, testing that unrelated batch neighbors do not change a trajectory on the same backend.
- [ ] P5.6 If beam mode is retained, implement K09 ancestry/cache gathering with disjoint banks and all carries. Test duplicated parents, reordered beams, finished beams, graph state, RNG identity, and reset/reuse. Sampling is not required to allocate beam banks.
- [ ] P5.7 Use fixed-step dispatch and absorbing finished masks with zero per-step host reads. Report active/finished work per step; compare optional compaction including extra launches/gathers.
- [x] P5.8 Test derived T and ring/atom caps against all supported target traces, and preserve explicit failure statuses for unsatisfiable formulas, no-valid-action states, and malformed requests. On a tiny domain, enumerate traces including absorbing failure outcomes, check normalized probabilities, and test fixed-seed sample frequencies and batch independence.
- [ ] P5.9 Verify the owned-cache in-place step against the out-of-place reference, including old last_u/angle reads. Keep explicit extra banks in the budget until in-place correctness is established.
- [ ] P5.10 For optional beam mode, implement immutable time-indexed parent/action records, bounded completed-result storage, and device traceback. Surviving/finished traces must remain correct after live slots are reordered or recycled.

Acceptance: tiny-model candidates and scores match a CPU exhaustive/reference search; beam siblings cannot alias mutable cache state. Completed graphs are distinguished from truncated histories.

## P6 — Validation, identity, evidence, and APIs

- [x] P6.1 Implement K10 GPU validation for declared connectivity, bond uniqueness, open valence, charge/H, composition, and ring rules. Start with offline-normalized explicit bond orders and exact labeled-graph identity; aromatic/resonance equivalence is a separately tested domain extension.
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

- [x] P8.1 Capture the composed baseline before tuning; profile formula search and beam-state movement as separate stages.
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

These targets exist. Two machines have run them: an Apple M1 (wgpu/Metal; paths under `/Users/ods/...`, CASMI data)
and a Linux PC with a Radeon 860M (wgpu through Vulkan/RADV; MassSpecGym data in `data/ms2/`). On the Linux PC the
GPU feature is `wgpu` (WGSL): `--features vulkan` (CubeCL's SPIR-V path) crashes inside the driver's SPIR-V front end
(`radv_shader_spirv_to_nir`, Mesa 26.2.3) on the first MS2 peak-selection kernel and is an open capability gap, not
a passed target.

```sh
# Host references (P0/P1, P4 host twins; no device code)
cargo test --release --no-default-features --features cpu --test ms2_chemistry --test ms2_targets --test ms2_contract --test ms2_dataset --test ms2_bounds --test ms2_contain --test ms2_metrics --test ms2_formula_enum --test ms2_ion --test ms2_identity --test ms2_allocate --test ms2_pack --test ms2_calibration
# Device suites: run each twice, with --features cpu and with --features wgpu; --test-threads 1 on wgpu
cargo test --release --no-default-features --features wgpu --test ms2_kernels --test ms2_kernel_launches --test ms2_encoder --test ms2_formula --test ms2_decoder --test ms2_generation --test ms2_workspace --test ms2_experiment --test ms2_profile -- --test-threads 1
# V1 kernels against their host twins, and V1 modules (same two backends)
cargo test --release --no-default-features --features wgpu --test ms2_enum_kernels --test ms2_enum_integration --test ms2_identity_kernels --test ms2_ion_kernels --test ms2_pack_kernels --test ms2_packed --test ms2_rerank_kernels --test ms2_rerank --test ms2_assign --test ms2_baselines --test ms2_fingerprint --test ms2_dtype -- --test-threads 1
# Counter and memory tests, each in its own binary
cargo test --release --no-default-features --features wgpu --test ms2_counters --test ms2_encoder_footprint --test ms2_formula_footprint --test ms2_decoder_footprint --test ms2_generation_footprint --test ms2_footprint --test ms2_footprint_v0 --test ms2_launch_budget -- --test-threads 1 --nocapture
# Profiles, labels, formula sources and experiments
cargo run --release --no-default-features --features wgpu --example profile_ms2_substructure -- --mode both --n 64,128,256,512 --b 1,8,32 --stability 200 --out <json>
cargo run --release --no-default-features --features cpu --example ms2_label_report -- --input <export.json> --out <json>
cargo run --release --no-default-features --features cpu --example ms2_formula_report -- --train <export> [--ratio-train <export>] --validation <export> --table <table.json> --out <json>
cargo run --release --no-default-features --features cpu --example ms2_ion_report -- --input <export.json> --out <json>
cargo run --release --no-default-features --features wgpu --example ms2_experiment -- --train <export> --validation <export> --table <table.json> --control none|shuffled|metadata|prior --steps 6000 --batch 16 --lr 1e-3 --seed 1 --eval-every 1500 [--gold-conditioning composition|row] [--formula-window M] [--formula-source table|enumerate --enum-fit <train export>] [--allocation round-robin|proportional] [--identity trace|graph] [--returned R] --save <ckpt> --out <json>
cargo run --release --no-default-features --features wgpu --example ms2_experiment -- ... --load <ckpt> --diagnose --out <json>
# Python reference and data tools (RDKit; `uv run --with rdkit --with numpy --with pyarrow`, PYTHONPATH=tools/ms2)
python tools/ms2/export_msgym.py --name <name> --train-molecules N --validation-molecules M [--scaffold-holdout] [--instrument orbitrap|qtof]
python tools/ms2/export_msgym.py --name <name> --split msgym-split-v1 --fit-molecules N --rank-molecules N --calibration-molecules N --report-molecules N
python tools/ms2/formula_table_msgym.py --out <report.json> --table-out <table.json>
python tools/ms2/formula_table_db.py --out <report.json> --table-out <table.json>     # train + ChEBI 3-star formulas
python tools/ms2/export_casmi.py --data <casmi data> --name <name> --train-molecules N --validation-molecules M   # M1 only
python tools/ms2/label_specificity.py --export <export.json> --out <json>
python tools/ms2/export_fingerprints.py --export <export.json>                          # whole-parent Morgan sidecar
# Python bindings (CPU wheel): in bindings/python, a virtualenv with maturin, numpy and pytest
maturin develop --release && pytest tests/test_ms2.py -q
```

Not yet present: a GPU wheel of the Python binding and a CUDA/HIP run (the binding itself and
`bindings/python/tests/test_ms2.py` exist, P6.6). The CASMI
exports are CC BY-NC and stay in the CASMI data directory; MassSpecGym exports stay in `data/ms2/` (ignored), and its
`test` fold is never read; reports in `bench/results/ms2/` hold aggregates only. Mark unsupported dtype/backend pairs
as explicit capability cases; do not treat skipped required GPU execution as a passing GPU test.

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
| P2.6 | 2026-10-04 | `examples/profile_ms2_substructure.rs`: stage boundaries come from hooks inside the production `generate` (shared stage functions: preprocess, encoder, formula search, decoder initialisation, every decode step, validation, readout) and the production training step (forward, backward, optimizer); cold first call, warm p50/p95 over repeats, warm-up count; launches, reads, bytes and allocations per stage; exact launch budget `L_call = L_preprocess + L_encoder + L_search + L_init + (T − 1) L_step + L_finalize` (difference 0, `tests/ms2_launch_budget.rs`, pinned per backend); memory estimate, refusals and sampled reserved bytes; `--stability`; JSON output. Clocks are named: `sync_wall_ms` is `SynchronizedHostWallClock`; `--profile-mode device` runs the same production stages inside `backend::profile_session` (session state built and dropped on the device runner thread, `Send`-only callbacks, no `unsafe`, panic-safe, session identities) and reports `client.profile` spans with the runtime's timing method. **Limitation, measured and kept in the output**: with the pinned cubecl-wgpu 0.10 a `client.profile` span covers one timestamped compute pass (at most 32 tasks), so on wgpu only single-pass stages have a device duration ([JSON](../bench/results/ms2/profile_device_spans_wgpu_radeon860m.json): search 0.11 ms for 32 launches, validation 0.05 ms, optimizer 1.2 ms) and every longer stage is `"unavailable"` with the reason; on the CPU runtime the method is `SystemTime` and every stage has a span. `tests/ms2_profile.rs` (17 tests) on CPU and wgpu. Three codex reviews, the third one's findings fixed |
| P3.6 | 2026-10-04 | `profile_ms2_substructure --mode both --n 64,128,256,512 --b 1,8,32` on the Radeon 860M through wgpu ([JSON](../bench/results/ms2/profile_grid_wgpu_radeon860m.json), 10 repeats, 200 stability calls) and on the CPU runtime of the same machine, a Ryzen AI 7 350 ([JSON](../bench/results/ms2/profile_grid_cpu_ryzen.json), 3 repeats): 24 configurations each, generation and training, none refused under the 4 GiB limit (largest estimate 527 MiB, training at B = 32). Per configuration: cold and warm p50/p95 wall time, stage boundaries taken inside the production call (preprocess, encoder, formula search, decoder initialisation, decode loop and per step, validation, readout; forward, backward, optimizer), launches, reads, bytes, allocations, the memory estimate and sampled reserved bytes. Warm `generate`, wgpu, K = 8: 20 to 33 ms at B = 1, 68 to 73 ms at B = 8, 237 to 351 ms at B = 32, independent of N within noise; the decode loop is 92 to 95% of the summed stage times; 3,436 launches and 1 read per call. Warm training step, wgpu: 41 to 45 ms (B = 1), 54 to 62 ms (B = 8), 104 to 123 ms (B = 32), about 4,900 to 5,000 launches. The machine was not quiet (CPU builds ran alongside), so times are indicative and counts exact |
| V0.1 | 2026-10-03 | `ModelConfig::v0()` (N=128, d=128, s=32, two encoder and two decoder blocks, F=4, K=8, A=16, R_max=4, T=22 from trace coverage in contracts §9), FP32, direct composed cross-attention; per-layer K/V computed once per call and held in `DecoderState` |
| V0.2 | 2026-10-03 | `formula_window` → `FormulaHead::score` → `formula_top` on device with work counters, absent-gold slot (`u32::MAX`) and exhaustion status; `top_count` counts written entries only; `oracle_formula` is `Error::Unsupported` in V0; `tests/ms2_formula.rs`, CPU and wgpu |
| V0.3 | 2026-10-03 | `sample_step` (shared `#[cube]` legality helpers, inverse-CDF draws keyed per architecture §3.6), `init_trajectories` (budgets from the integer counts), `validate_trajectories`, `generate.rs` with carry freeze; `tests/ms2_generation.rs` (twin bit-equality, exhaustive 33-trace frequencies on sequential ids within 4σ, stream independence, batch independence, carry freeze, teacher/stepped log-prob agreement), CPU and wgpu |
| V0.4 | 2026-10-03 | `decoder.rs` teacher forcing + `graph_loss` (q-weighted, divisor B), replay masks equal to `TraceState` (P1.9 bitsets) and shared with the sampler, creation-state atom memory, stepped parity within 1e-4, finite-difference gradients on exercised rows; `tests/ms2_decoder.rs`, `tests/ms2_decoder_footprint.rs` (0 reads, constant launches per training step), CPU and wgpu |
| V0.6 | 2026-10-04 | `CandidateBatch` carries `evidence_status`, `identity_resolution` and `attachment_partition` (V0 constants) with the candidate and request status bits; `tests/ms2_generation.rs` pins them: every generated record has the three constants; a forced same-trace, same-formula pair through the device `ms2_validate` flags only the later record and keeps all records' action words, lengths, formula rows and `finished` bits; two different legal traces of one labeled graph are not flagged (unresolved graph duplicates stay visible); `CandidateBatch::distinct_traces` is the host helper for callers that drop exact-trace duplicates. One device read per warmed `generate` (footprint binaries). CPU and wgpu (Radeon 860M); codex-reviewed |
| V0.7 | 2026-10-04 | End-to-end profiles with launch, read, byte and allocation counts: the P3.6 grids (CPU and Radeon 860M) plus generation at K ∈ {1, 8, 32} × B ∈ {1, 8, 32} on the Radeon (`bench/results/ms2/profile_generate_k*_wgpu_radeon860m.json`); training-step profiles in the same grids. Targets written down before any production optimisation and before any held-out result with the enumeration source: [Baselines and targets](#baselines-and-targets-v07) |
| V0.5 | 2026-10-03 | Overfit fixture, 3,524-spectrum pilot and 36,974-spectrum scale run with real, shuffled (molecule-aware donors), metadata-only and structure-prior models on wgpu/M1, bootstrap intervals over molecules; diagnostics and label-specificity check. At scale, real spectra beat every control on held-out NLL with non-overlapping intervals. [V0 results](#v0-results), `bench/results/ms2/v0_*.json` |
| P5.1 | 2026-10-04 | The grammar state of V0 (`state [.., 3A + 16]`: atom types, residual valences, parents, counters, used composition against the formula budget; charges and hydrogens are fixed by the atom type in the V0 domain; started/finished/failed flags live on the device) with capacities taken from `ModelConfig` (`A` up to 32, `R_max` up to 8, `T` up to 64, up to 4 decoder blocks, decoder `d_inner` apart from `d_model`); no V0 capacity is compiled into a kernel or head. Shape-pinning tests run at `(A, R_max, T) = (16, 4, 22)` and `(32, 8, 42)` (`tests/ms2_decoder.rs`, `tests/ms2_generation.rs`, `tests/ms2_footprint.rs`), CPU and wgpu; codex review part C: conformant |
| P5.2 | 2026-10-04 | `decoder.rs`: causal Mamba-3 stack, direct cross-attention with per-layer keys and values computed once per spectrum, creation-state atom memory, factorised heads in the frozen order; teacher-forced logits equal stepped logits with every carry and the atom memory compared, at both shapes above (4 decoder blocks at the larger one), within 1e-4; no graph network (ablation only, not built). CPU and wgpu |
| P5.3 | 2026-10-04 | Legality masks from one set of `#[cube]` functions shared by replay (training), sampling and validation; replay masks equal `TraceState::masks` for every fixture trace and for synthetic traces up to 32 atoms and 8 closures; conditional log-probabilities normalised over the legal set, identical in training and inference; a state with no legal action sets `no_valid_action` and emits no token (no NaN, no fabricated probability). CPU and wgpu |
| P5.8 | 2026-10-04 | `T = 2 + A + R_max` holds every supported target trace (longest 22 at the V0 caps, P1.9; synthetic traces at 32/8/42); explicit statuses tested through `generate`: an unsatisfiable formula fails every trajectory with `no_valid_action` and no token, a malformed request is `request_failed`; the tiny-domain test enumerates every absorbing outcome, finished and failed, with probabilities summing to 1 (the failure leaves go through the same evaluator), fixed-seed frequencies within 4 standard errors, batch independence; truncation is tested with a deliberately short kernel horizon. `tests/ms2_generation.rs`, CPU and wgpu |
| P6.1 | 2026-10-04 | `ms2_validate` replays every trace with the shared grammar functions and applies the final-validity rules; it gained the missing check "a record claimed finished must end in STOP" in kernel and twin. One negative test per rule on device and twin, each paired with its minimal legal variant: pointer to a missing atom, a second bond between two atoms, a bond beyond residual valence, an atom type outside the formula budget, a closure beyond `R_max`, a closure to the parent, finished without STOP (a legal truncated history stays `truncated`), no atom. Explicit bond orders and exact labelled-graph identity only; aromatic equivalence is not applied. CPU and wgpu; codex review part C: conformant |
| P8.1 | 2026-10-04 | The composed baseline before any tuning is the P3.6 grid and the K-curve of [Baselines and targets](#baselines-and-targets-v07), with the formula search as its own stage (32 launches, about 1 ms warm on wgpu for the table source). There is no beam-state stage because beam mode is not built |

## Partially done and open items

State on 2026-10-04, 13:55. Nothing in this list is a checked box. "CPU" and "GPU" say on which backend the
supervisor has re-run the tests of that state; "CPU only" means the GPU run of the latest change is still owed.

- **P2.3** (shape-bucketed workspace): `GenerationWorkspace` and the trainer keep per-bucket buffers with bounded
  retention (at most 4 generation buckets, keyed by batch, trajectories, steps, raw capacity, formulas, formula
  window and enumeration lanes). Measured on the Radeon (`--stability 200`): reserved bytes identical before and
  after 200 fixed-shape calls and 200 alternating `B = 1 / 8` calls, 2 buckets, the same number of allocation calls
  on every call. Not checked: the decode loop still allocates its outputs every step (173 allocation calls per step
  on wgpu), so "reusable outputs" holds for the workspace buffers only, and no serving measurement exists.
- **P4.1 / P4.3 / P4.9** (formula candidates). Done and verified on CPU and GPU: the candidate-composition stage
  (schema version 2, window capacity `M`, device gold slot, conditioning on the parent composition); the
  enumerating source end to end — host reference with train-fitted pruning, four device kernels equal to their
  host twins element for element (`ms2_enum_count`, `ms2_enum_offsets`, `ms2_enum_fill`, `ms2_cand_pad`),
  generation and training with `--formula-source enumerate`, artifacts fitted on train data only (enforced: a
  validation export or one sharing a molecule key is refused) and stored in the checkpoint, one read per
  `generate`; the fixes for the third review of the kernels and for the review of the integration (one effective
  scored cap for offsets/fill/pad, DBE endpoint validation, the rare-table boundary, fill early exits, lane and
  address checks, source-specific counter validation, duplicate detection by formula identity, truthful statuses
  for searches that did not complete, lane refusal before any upload). Measured comparison of the sources:
  [V1 §1.5](MS2_V1_ARCHITECTURE.md).
  **GPU reset, found and fixed.** With this source and a formula window of 2,048 on the scale export the driver
  reset the GPU (`amdgpu: ring gfx_0.0.0 timeout`, then "Parent device is lost"). Bounding the enumeration
  dispatches (976 lanes and 4,096 visits per lane per launch, one submission per launch) did not help; bisection
  on the GPU showed the same data training at window 512 and window 2,048 running on a small set, which pointed
  at the top-F selection kernel: O(F·M²) inside a single lane per spectrum, about 17 million guarded loads at
  M = 2,048. Rewritten with its twin as F argmax passes (O(F·M); a score outside the validated domain is never
  selected, the only behavioural change), the 2,048 window now trains on the scale export without a reset
  (150 steps, 0.51 s per step at p50, 0.81 s at p95, 5,264 launches). The bounded dispatch and per-launch
  submission stay as bounds on the worst case.
  **Open, blocking these boxes:** the shuffled-spectrum control shows that the formula ranking with this source
  is almost entirely precursor mass and prior (recall 0.448 shuffled against 0.482 real), so "neural ranking" of
  P4.1 is not demonstrated to use the spectrum; the fixes after the last two reviews and the top-F rewrite are
  under review; the search has not been profiled as its own stage at production shapes (P4.9 asks for measured
  latency of the device path; the host reference's is in V1 §1.5). Held-out results at windows 512 and 2,048
  are in the results section.
- **P4.2** (ion assignment): host reference (`models/ms2/ion.rs`, `ms2_ion_report`); kernels equal to their twins
  (`ms2_ion_assign`, `ms2_ion_label_mask`, `ms2_ion_evidence`; CPU and GPU, reviewed twice, findings fixed); the
  assignment head and loss (`models/ms2/assign.rs`: distribution over kept hypotheses plus an explicit unassigned
  class; gradients reach the formula head's row network; CPU and GPU; reviewed: accept-with-fixes, two findings
  open — the "partial" count is a proxy that does not measure dropped labels, and the peak projection has a bias
  the specification does not). Integrated since 2026-10-04 (CPU only, not reviewed): `ModelConfig::assignment`,
  `TrainConfig::lambda_assign` with `L = L_graph + 0.2 L_formula + lambda_a L_assign` and no extra device read,
  `GenerationConfig::evidence` filling `evidence_status` (0, 1, 2, bit 7) and up to four evidence records per
  candidate (original peak id, hypothesis, shift, residual, assignment log-probability) chosen by assignment
  probability, and the driver's `--assign` / `--evidence`. A CPU smoke run on 16 spectra drives the assignment
  loss from 1.13 to 0.10 while the graph loss still falls (labelled hypothesis ranked first for 97% of anchored
  peaks, a pseudo-label metric under the true parent formula). Measured under the true parent formula on the
  pilot exports: 75.5% (train) and 86.6% (validation) of kept peaks have at least one hypothesis, a median of 1
  and a 95th percentile of 3 to 5 hypotheses per peak, 99.9% and 99.2% of anchored peaks keep their whole label
  set at `J = 4`. GPU: the integration's suites pass and a full held-out run exists (results section: assignment NLL 0.135,
  no gain in formula recall or candidate quality). A follow-up (CPU so far) added the packed and resident
  evidence records, kernel-versus-twin tests for the two new kernels (which found and fixed a twin bug: the
  first evidence record was never inserted), the true partial-label state, the bias-free projection and the
  label-overflow count. Not done: GPU run and review of that follow-up; a rerun with the corrected head.
- **P4.4–P4.8** (sparse relations, slots, online softmax): not started. P4.7's requirement that keys, values and
  memory are shared across candidates and never replicated K times is how V0 already works, but the box also
  covers slots and evidence, which do not exist.
- **P5.4 / P5.6 / P5.10** (beam search): not built; optional by the plan, and `GenerationMode::Beam` is
  `Error::Unsupported`. Independent seeded sampling is the only mode.
- **P5.5** (allocation and candidate score): `ms2_allocate` (round robin and proportional, complete per-trajectory
  formula records) is called by `generate`; `init_trajectories` reads the allocation; the candidate score
  `formula_log_prob + trace_log_prob` orders the packed output. CPU only: the GPU run of this integration and the
  confirmation of the wgpu launch pins (search stage 32 → 33) are owed; not yet reviewed.
- **P5.7**: fixed-step dispatch, absorbing finished trajectories and zero per-step reads are V0 properties kept by
  the footprint tests; `CandidateBatch::work()` reports active invocations per step from the one read (CPU and
  GPU). Not done: the comparison with compaction of finished trajectories (O9).
- **P5.9** (in-place recurrent step): not built; the estimate budgets two cache banks.
- **P6.2 / P6.4 / P6.5** (identity, ranking, packed output, readout modes): kernels with host twins
  (`ms2_graph_hash`, `ms2_graph_identity`, `ms2_rank`, `ms2_record_pack`, `ms2_pack`; CPU and GPU, reviewed, fixes
  applied) and their integration: `GenerationConfig::identity = Graph` flags `duplicate_graph` (bit 7) and
  `identity_unresolved` (bit 8) and fills `identity_resolution`; `generate_packed` returns the top-R
  `PackedCandidateBatch` with one read; `generate_resident` performs no read and owns its buffers until `read()`.
  CPU only for the integration; not yet reviewed.
- **P6.3 / P7.9** (reranker, calibration): standalone components — the 8-feature kernel `ms2_rerank_features`
  with its twin, `Reranker` and `RerankTrainer` (weighted BCE, no read between reports), Platt scaling,
  expected calibration error, Brier score and reliability tables per size stratum, a versioned calibration
  artifact (CPU and GPU). Reviewed: rejected with five findings (a cross-entropy whose gradient was wrong at
  exactly zero logits, an undamped Platt fit that could diverge, zero-weight batches still moving parameters
  through Adam's momentum, two evidence features off the specification, overflow in standardisation); all five
  are fixed in a working copy with tests (CPU), not yet merged, GPU-run or re-reviewed. The split it needs exists
  (`msgym-split-v1`: fit 33,667 / rank 8,277 / calibration 2,854 / report 3,002 spectra) and a generator trained
  on `fit` only exists (held-out NLL per token 1.220 [1.191, 1.247] on `report`). Not done: no reranker has been
  trained, no calibration fitted, nothing is wired into the drivers.
- **P6.6** (Python): `mamba3_ms2` exposes the configs, spectrum batches, the formula table, `generate`, candidate
  arrays, the experiment set and the trainer (`step`, `teacher_eval`, save/load); 16 tests pass on the CPU wheel,
  including equality of training losses with the Rust driver for the same data and seed. `generate_packed`,
  `generate_resident`, the assignment and evidence fields and the new config fields were added with the
  integrations above and compile (`cargo check`); their pytest run is owed. Reviewed: rejected — resident
  inputs, `encode`, loading a trained model for inference and the enumerating source's artifacts are not
  exposed, and the error-parity test covers only some error variants. No GPU wheel has been built.
- **P6.7**: the property holds by construction (one device read per `generate`, no chemistry toolkit linked into
  the crate, containment and canonical traces used only in label preparation, metrics and tests) and the read
  count is tested, but the audit has not been written up as its own test and statement.
- **P6.8**: the supported mapping is the mass relation `(g, s)` of contracts §4.3; `ion.rs` implements it
  (`mapping_is_supported`, `embedding_ion`) and tests it against the label recipe for every fixture molecule,
  both adducts and shifts −2 to 2. Not done: no atom-level mapping exists (the contract says the relation names no
  atoms), and candidates carry no evidence yet.
- **P7.1 / P7.2 / P7.4 / P7.5 / P7.8**: not started beyond V0 (targets, losses and checkpoints are V0's, extended
  by the formula artifacts in the checkpoint). The new kernels are integer and non-differentiable; the new
  differentiable paths (conditioning on the gold composition, the assignment head, the reranker) have
  finite-difference gradient tests.
- **P7.3** (fingerprint supervision): standalone — `export_fingerprints.py` (Morgan radius 2, 1,024 bits of the
  whole parent), `models/ms2/fingerprint.rs` (head, loss, metrics), tests on CPU. Reviewed: rejected (the same
  zero-logit gradient defect as the reranker's loss, an inconsistent weight normalisation, a sidecar check by key
  only); fixed in a working copy with tests (CPU), not yet merged or re-reviewed. Not wired (`lambda_fp` does not
  exist in the trainer yet); no GPU run.
- **P7.6**: clipping and the optimizer run on the device and a warmed training step performs no read between
  reports (tested on both backends since V0). Not checked: no on-device gradient-finiteness check or loss scaling
  exists.
- **P7.7** (precision): measured matrix (`tests/ms2_dtype.rs`, `bench/results/ms2/dtype_matrix_cpu.json`). CPU
  runtime: f32 and bf16 train and generate correctly (bf16 final loss within 1% of f32, exact-mass integer outputs
  identical to f32's); pure f16 gives a NaN loss from step 0. wgpu: f32 works; bf16 is refused with a clear
  error; f16 fails to compile an MS2 kernel (`3e38` is not representable) and surfaced as a panic. A guard that
  refuses every unvalidated dtype before any launch was added with the integration above (CPU only). No held-out
  quality with bf16 has been measured.
- **P8** (optimisation): P8.1, and the launch-count half of O4 (2026-10-04, see
  [Progress against the targets](#progress-against-the-targets-o4)): the sampler step is fused
  (`Ms2Decoder::step_packed`, architecture §3.9), the teacher pass scores every position in one pass, and the
  single-step SSM update no longer copies the state's transpose. Not done: O1 to O3 and O5 to O12; the decode
  loop still allocates (81 calls per step at the time of that measurement, down from 173; not re-counted since);
  the training target at the real shape is not reached. A second pass
  ([Device-bound kernels on the Radeon](#device-bound-kernels-on-the-radeon-2026-10-04-second-pass)) fused the
  mixer step for devices with planes and stored the pointer scores, which meets the B = 8 generation target.
- **P9**: first measurements only — molecule-disjoint, scaffold-held-out and instrument-held-out (Orbitrap to
  QTOF) evaluations of the table-source model, the split audit, generation latency and throughput over B and K
  (see the results sections). Baseline encoders for P9.3 exist as standalone stacks (DeepSets-style set encoder,
  Transformer, forward-only Mamba; parameter counts matched within 1.3%; CPU and GPU; reviewed:
  accept-with-fixes, both findings fixed in a working copy) but are not selectable in the model. The split tool
  was reviewed (rejected: fold conflicts were detected by SMILES, not by identity block) and fixed; the existing
  exports are unchanged by the fix (same molecule arrays by hash). No FPNet comparison. No release evaluation.
- **Capability gaps**: `--features vulkan` (SPIR-V) crashes in the Radeon driver's SPIR-V front end on the first
  MS2 kernel; whole-stage device time is unavailable on wgpu (P2.6); no CUDA or HIP run; f16 unsupported.
- **Reviews**: V0-C (sampler, generation) and V0-D to V0-G (evaluation, trainer, diagnostics) have still not had a
  codex review of their own; the V1 changes to those files were reviewed as diffs. Codex's usage limit was
  reached once during this work (02:24 to 04:09), so reviews were batched; see [Review history](#review-history)
  for what has and has not been re-reviewed.

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

## MassSpecGym results (Linux, Radeon 860M)

Second dataset and second GPU for V0.5, produced on 2026-10-03/04. The CASMI data is not on this machine; the runs use
[MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) 1.5 (`data/pinned/`, 231,104 spectra), exported by
`tools/ms2/export_msgym.py` with the same schema and the same `uniform-in-domain-v2` sampling as the CASMI exporter:
training molecules from the `train` fold, validation molecules from the `val` fold, the `test` fold never read.
MassSpecGym's folds are disjoint by structure (not only by molecule), all its in-domain spectra are `[M+H]+`, and
the exports are derived data kept out of the repository (`data/ms2/`, ignored). The export tools were reviewed by
codex ([review](reviews/MS2_MSGYM_EXPORT_CODEX_REVIEW.md)); two findings were fixed and the exports are unchanged
by the fix (same molecule arrays by hash).

Device: AMD Radeon 860M (integrated, shared memory) through `--features wgpu` (WGSL → naga → Vulkan/RADV, Mesa
26.2.3). The process holds `/dev/dri/renderD128` and the GPU reads 98% busy during training. Protocol as for CASMI:
V0 model config, AdamW lr 1e-3, batch 16, seed 1, the same step budget for every model of a comparison, 95% bootstrap
intervals over molecules, metrics conditional on in-domain molecules. Formula table: the 12,806 distinct formulas of
the 24,171 in-domain train-fold structures. Labels (`ms2_label_report`,
[train](../bench/results/ms2/labels_msgym_pilot_train.json),
[validation](../bench/results/ms2/labels_msgym_pilot_validation.json)): 88.8% of 3,480 pilot-train and 85.6% of 714
pilot-validation spectra are labeled, 7.8 and 6.6 targets per spectrum, no replay failure, longest trace 22.

**A structural difference from the CASMI runs.** With a table of train formulas, the gold formula of a
structure-disjoint validation molecule is usually missing: 542 of the 714 validation spectra (76%) have it outside
the table, formula recall at F is 0.23 for every model, and V0 then conditions teacher forcing on the zero vector.
About 27% of validation spectra abstain (no table row in the precursor window), and validity of the sampled
candidates is about 0.50 against 0.87 on the overfit fixture; the cause of the lower validity was not isolated.
The NLL comparisons below are therefore between models that share this handicap; P4
removes it (see [MS2_V1_ARCHITECTURE.md](MS2_V1_ARCHITECTURE.md) §1).

| Run | Model | Validation NLL/token, final step | at steps 1,500 / 3,000 / 4,500 | Coverage at K=8 | Precision |
|---|---|---|---|---|---|
| Overfit, 128 spectra, 3,000 steps | Real spectra | 4.598 → 0.138 [0.121, 0.156] (train = eval) | — | 0.701 [0.656, 0.745] | 0.599 [0.547, 0.649] |
| | Shuffled | 4.567 → 0.144 [0.125, 0.164] | — | 0.707 [0.662, 0.750] | 0.555 [0.500, 0.607] |
| Pilot, 3,089 labeled train spectra, 6,000 steps | Real spectra | 2.951 [2.834, 3.072] | 2.572 / 2.901 / 2.910 | 0.008 [0.001, 0.018] | 0.011 [0.007, 0.017] |
| | Shuffled | 3.250 [3.141, 3.364] | 2.580 / 3.275 / 3.259 | 0.004 [0.000, 0.011] | 0.009 [0.004, 0.018] |
| | Metadata only | 3.386 [3.243, 3.530] | 2.325 / 2.884 / 3.236 | 0.003 [0.000, 0.010] | 0.005 [0.002, 0.009] |
| | Structure prior | 3.291 [3.153, 3.426] | 2.287 / 2.814 / 3.247 | 0.001 [0.000, 0.003] | 0.009 [0.006, 0.011] |
| Scale, 37,259 labeled train spectra of 23,235 molecules, 6,000 steps | Real spectra | **1.442** [1.384, 1.498] | 1.709 / 1.568 / 1.465 | **0.035** [0.021, 0.051] | 0.022 [0.015, 0.030] |
| | Shuffled | 1.576 [1.522, 1.628] | 1.731 / 1.643 / 1.585 | 0.001 [0.000, 0.001] | 0.006 [0.004, 0.009] |
| | Metadata only | 1.590 [1.531, 1.643] | 1.759 / 1.673 / 1.632 | 0.000 [0.000, 0.001] | 0.006 [0.004, 0.008] |
| | Structure prior | 1.577 [1.520, 1.630] | 1.728 / 1.639 / 1.569 | 0.001 [0.000, 0.002] | 0.009 [0.006, 0.012] |

Reports: `bench/results/ms2/v0_msgym_*.json`. Reading, with the declared primary comparison (held-out NLL per token
at the final step):

- **Overfit**: the criterion "below 10% of the initial NLL" is met on the GPU (3.0%); coverage above the shuffled
  control is not (0.701 against 0.707), for the reason already found on CASMI — with one spectrum per molecule the
  formula identifies the molecule.
- **Pilot**: every model overfits; each is best at step 1,500, where the structure prior and the metadata-only
  model are ahead (2.29, 2.33) of real and shuffled spectra (2.57, 2.58). At the declared final step real spectra
  are below every control with intervals that do not overlap the shuffled control's (3.072 against 3.141), but that
  is an ordering among overfitted models and is not a result in favour of the spectrum.
- **Scale**: the curves fall throughout, and real spectra are below each control at every evaluation and at the
  final step with non-overlapping intervals (1.442 [1.384, 1.498] against 1.576, 1.590 and 1.577). Coverage is 0.035
  against at most 0.001. This reproduces the CASMI scale finding on a second dataset, a structure-disjoint split and
  a different GPU and driver stack. Absolute quality is still far from useful, and three quarters of the validation
  spectra are evaluated without their formula.

Diagnostics of the saved checkpoints (`--diagnose`, `bench/results/ms2/v0_msgym_*_diagnose.json`; train NLL on
the first 714 labeled train spectra; donor − own is the paired change in validation NLL when a spectrum is given
another molecule's peaks; the two independent NLL computations agree to 1e-6 in every run):

| Run | Model | Train NLL/token | Validation NLL/token | Donor − own peaks, validation |
|---|---|---:|---:|---|
| Pilot | Real spectra | 0.298 | 2.951 | +0.106 [0.067, 0.146] |
| | Shuffled | 0.497 | 3.251 | −0.001 [−0.026, 0.023] |
| | Metadata only | 0.439 | 3.386 | 0 (exact) |
| | Structure prior | 0.565 | 3.291 | 0 (exact) |
| Scale | Real spectra | 0.933 | 1.442 | **+0.275** [0.250, 0.302] |
| | Shuffled | 1.078 | 1.575 | +0.001 [0.000, 0.001] |
| | Metadata only | 1.080 | 1.590 | 0 (exact) |
| | Structure prior | 1.064 | 1.577 | 0 (exact) |

The pilot's train/validation gap (0.30 against 2.95) is the overfitting; at scale the gap is 0.93 against 1.44 and
only the real model reacts to donor peaks (+0.275, of which atom type +0.242 and STOP +0.209), the same picture as
on CASMI (+0.269).

**Conditioning on the parent composition (V1 §1.2, first measurement).** The same scale protocol with
`--gold-conditioning composition` (teacher forcing conditions on the head embedding of the true parent composition
instead of the zero vector when the table lacks the formula), table source, `M = 32`
(`bench/results/ms2/v1_msgym_scale_*_goldcomp.json`; built from the working tree before the review fixes of that
change; the real-spectra run repeated on the reviewed code gives 1.2215 [1.1716, 1.2697], the same result,
`v1_msgym_scale_none_goldcomp_final.json`):

| Model | Validation NLL/token at steps 1,500 / 3,000 / 4,500 / 6,000 | Final, with interval | Coverage at K=8 | Formula recall at F |
|---|---|---|---|---|
| Real spectra | 1.470 / 1.319 / 1.288 / 1.221 | **1.221** [1.172, 1.270] | 0.029 [0.017, 0.044] | 0.230 |
| Shuffled | 1.503 / 1.404 / 1.371 / 1.336 | 1.336 [1.288, 1.381] | 0.003 [0.000, 0.010] | 0.230 |

The same two models trained for 20,000 steps (about 8.6 epochs; `v1_msgym_scale_*_goldcomp_20k.json`):

| Model | Validation NLL/token at steps 4,000 / 8,000 / 12,000 / 16,000 / 20,000 | Final, with interval | Coverage at K=8 | Precision |
|---|---|---|---|---|
| Real spectra | 1.276 / 1.196 / 1.188 / 1.146 / 1.175 | 1.175 [1.125, 1.227] | 0.050 [0.033, 0.069] | 0.037 [0.026, 0.048] |
| Shuffled | 1.367 / 1.309 / 1.290 / 1.281 / 1.304 | 1.304 [1.254, 1.353] | 0.002 [0.000, 0.004] | 0.007 [0.004, 0.010] |

Both curves flatten after about 16,000 steps and turn up slightly at 20,000; the gap to the shuffled control stays
(non-overlapping intervals at the final step), and coverage and precision of the real model rise from 0.029 and
0.019 at 6,000 steps to 0.050 and 0.037. More steps of this model on this data are not the lever; the formula
source is.

Both models improve on their V0 counterparts (1.442 and 1.576), and real spectra stay below the shuffled control
with non-overlapping intervals. This is an **oracle-conditioned** number: it measures the decoder given the right
formula. Generation still draws its formulas from the train-formula table (recall 0.23), so coverage does not
move; that is what the enumeration source of V1 §1.4 is for.

**Enumerating formula source, held-out results (V1 §1.4; `v1_msgym_scale_none_enum512.json`,
`v1_msgym_scale_none_enum2048.json`, `v1_msgym_scale_shuffled_enum2048.json`).** The scale protocol (37,259 labeled train spectra, 6,000 steps, batch 16,
composition conditioning, real spectra) with `--formula-source enumerate`, domain and ratio bounds fitted on the
train export only (7,993 rare-element lanes, 4,096 visits per lane), evaluated on the 714 structure-disjoint
validation spectra:

| Formula source | Formula recall at F = 4 | Gold formula in the scored support | Spectra over the window | Validation NLL/token | Coverage at K=8 | Precision | Validity | Abstention | Training step, p50 |
|---|---|---|---|---|---|---|---|---|---|
| Train-formula table, M = 32 | 0.230 [0.188, 0.272] | 0.24 | — | 1.221 [1.172, 1.270] | 0.029 [0.017, 0.044] | 0.019 [0.013, 0.026] | 0.51 | 0.27 | 0.17 s |
| Enumeration, M = 512 | 0.471 [0.422, 0.521] | 0.902 | 72.3% | 1.239 [1.188, 1.290] | 0.039 [0.025, 0.057] | 0.015 [0.010, 0.020] | 0.77 | 0.00 | 0.47 s |
| Enumeration, M = 2,048 | **0.482** [0.435, 0.530] | 0.965 | 20.9% | 1.230 [1.181, 1.279] | 0.048 [0.032, 0.066] | 0.022 [0.016, 0.030] | 0.78 | 0.00 | 0.56 s |
| Enumeration, M = 2,048, **shuffled spectra** (control) | 0.448 [0.396, 0.497] | 0.965 | 20.9% | 1.337 [1.290, 1.381] | 0.001 [0.000, 0.001] | 0.008 [0.006, 0.011] | 0.77 | 0.00 | 0.55 s |

Formula recall at F over the evaluations (steps 1,500 / 3,000 / 4,500 / 6,000): 0.417 / 0.417 / 0.428 / 0.471 at
M = 512 and 0.418 / 0.414 / 0.436 / 0.482 at M = 2,048, still rising at the end. "Spectra over the window" are
`formula_search_exhausted`: more joined candidates than the window, so their probabilities are over a prefix of
the support. The gold-in-support figure at M = 2,048 equals the host reference's prediction (V1 §1.5: 0.965).

Reading: with the enumerating source the head puts the true formula among its top 4 for about half of held-out
spectra, twice the ceiling a table of known formulas has on this split, and no spectrum abstains for lack of a
formula; validity rises from 0.51 to 0.78. Quadrupling the window brings the gold formula into the support for
96.5% of spectra instead of 90.2% but moves formula recall by one point: ranking, not the support, is now the
limit.

**The formula ranking does not yet use the peaks.** With each spectrum's peaks replaced by another spectrum's
(precursor mass and adduct kept), formula recall at F = 4 is 0.448 [0.396, 0.497] against 0.482 [0.435, 0.530]
with the real peaks: the intervals overlap almost entirely, and over the four evaluations the control is ahead
once and behind three times. Nearly all of the ranking comes from what the control keeps — the precursor mass
residual and the composition prior learned from training formulas. The graph decoder does use the peaks: the
same control costs 0.107 nats per token (1.337 against 1.230, intervals disjoint) and takes coverage from 0.048
to 0.001 and precision from 0.022 to 0.008. Consequence for the plan: the formula head needs peak-derived
evidence in its candidate features (the ion-assignment head of P4.2 is the designed route: how many peaks a
candidate composition can explain) before a larger window or longer training can be expected to help.

**What could rank the formulas (host experiment, `formula_evidence_msgym_scale.json`; 714 validation spectra,
M = 2,048, linear softmax rankers fitted on 3,000 train spectra).** Recall of the true formula, all spectra in
the denominator:

| Ranker | Recall@1 | Recall@4 | Recall@16 | Recall@4, shuffled peaks |
|---|---|---|---|---|
| Precursor mass residual alone (rule) | 0.626 | 0.707 [0.675, 0.741] | 0.755 | 0.707 (does not use peaks) |
| Composition prior alone (linear) | 0.048 | 0.226 | 0.595 | 0.226 |
| Explained peak intensity alone (rule) | 0.305 | 0.342 | 0.380 | 0.179 |
| Trained neural head (GPU run above) | — | 0.482 | — | 0.448 |

Three findings. (1) The neural head does not see the candidate's mass residual at all (its features are
`ln(1 + count)` and the pooled spectrum vector), and the residual alone beats it by 22 points. (2) Much of the
residual's strength is a property of this dataset, not of mass spectrometry: the stored precursor m/z is within
0.1 ppm of the theoretical ion mass of the true formula for 46% of the validation fold's spectra and 37% of the train fold's
(median error 0.14 and 0.23 ppm; all [M+H]+ and [M+Na]+ rows of the source table, train and validation folds only) — for many
library records it was computed from the formula. A measured precursor is off by 1 to 5 ppm. A ranker that uses
the residual must therefore also be evaluated with a realistic precursor error added, or its recall here
overstates what it would do on measured data. (3) Peak explanation alone is unspecific: the true formula never explains uniquely the most intensity, and
when it is at the maximum it shares it with a median of 457 other candidates; what it adds to a prior is
measured in the next table. (The first run's combined linear ranker is left out: its optimiser had not
converged.)

**With a realistic precursor error added (`formula_evidence_msgym_scale_jitter.json`).** Each precursor m/z is
multiplied by `1 + e·10⁻⁶`, `e` normal with standard deviation σ ppm, truncated at 3σ, seeded; the window keeps
the true formula for 96.5% of spectra at every σ. Recall@4, all 714 validation spectra in the denominator:

| Ranker | σ = 0 (as stored) | σ = 1 ppm | σ = 2 ppm | σ = 5 ppm |
|---|---|---|---|---|
| Precursor residual alone (rule) | 0.707 | 0.116 | 0.050 | 0.027 |
| Composition prior alone (linear) | 0.226 | 0.233 | 0.231 | 0.220 |
| Prior + explained peaks, no residual (linear) | 0.496 | 0.482 | 0.478 | 0.499 |
| — the same on shuffled peaks | 0.224 | 0.237 | 0.233 | 0.228 |

The residual rule collapses from 0.707 to 0.116 with one ppm of error: its strength on the stored data is the
dataset property described above and would not carry over to measured precursors. The explained-peak features
are worth 25 points of recall@4 over the prior, at every σ, and vanish with shuffled peaks (0.48 to 0.50 against
0.23): that is genuine spectral information, and a linear model with them already equals the trained neural head
(0.482), which has no such features. The rows that combine the residual with other features come from an
optimiser that did not converge (a model with more features ended with a worse training objective than its
sub-model; unstandardised features), so their values and their shuffled controls are not reported here; the
experiment is being rerun with standardised features and an enforced convergence check, together with a
non-linear peak-free prior as an extra control.

Consequences, fixed before the next training runs: (a) the formula head gets per-candidate explained-peak
features (an exact device kernel with a host twin) and residual features; (b) every formula-ranking number is
reported at the stored precursor and with σ = 2 ppm added, and the residual features are only trained with the
error added; (c) the V0.7 target for formula recall is judged on the σ = 2 ppm number.

Coverage and precision of the generated candidates with real spectra stay where they were, within their
intervals (the best point values so far, 0.048 and 0.022, are far below the V0.7 targets of 0.10). The V0.7
target for formula recall (0.50) is inside the interval and not met by the point estimate. A training step costs
about 0.55 s against 0.17 s with the table.

**Assignment loss and evidence, first held-out run (V1 §2; `v1_msgym_scale_none_enum2048_assign.json`).** The
scale protocol with `Enumerate`, `M = 2,048`, `--assign --evidence` (`lambda_a = 0.1`, `J = 4`), on the GPU, with
the head as it was before the two review fixes (a bias in the peak projection, the "partial" count a proxy):

| Quantity (714 validation spectra) | Value |
|---|---|
| Assignment NLL under the true parent formula (pseudo-label) | 0.135 (0.149 / 0.141 / 0.136 / 0.135 over the four evaluations) |
| Labelled hypothesis ranked first (pseudo-label) | 0.946 of 4,559 eligible anchored peaks; 35 peaks dropped at `J = 4` |
| Graph NLL per token | 1.229 [1.177, 1.277] (1.230 without the assignment loss) |
| Formula recall at F = 4 | 0.438 [0.388, 0.488] (0.482 [0.435, 0.530] without) |
| Coverage / precision at K = 8 | 0.039 [0.025, 0.056] / 0.017 [0.012, 0.022] (0.048 / 0.022 without) |
| Validity | 0.794 [0.770, 0.817] |
| Generated candidates with evidence status 0 / 1 / 2 | 0.884 / 0.003 / 0.113; mean evidence records per candidate 0.15 |
| Training step, p50 / p95 | 0.58 s / 1.92 s; 5,372 launches; one read per generate call |

Reading: the assignment head learns its pseudo-labels (held-out NLL 0.135, labelled hypothesis first for 95% of
anchored peaks under the true formula) without hurting the graph loss. It does not improve formula ranking or
candidate quality — every difference from the run without it is inside the intervals, with the point values
slightly lower. 88% of generated candidates have no supporting peak at all, which is the low candidate quality
seen from another side. The assignment metrics are pseudo-label metrics under the oracle formula; nothing here
says a hypothesis is the experimentally correct fragment.

**Holdouts (first measurements for P9.1; table source, composition conditioning, teacher-forced NLL is
oracle-conditioned; `bench/results/ms2/v1_msgym_orbitrap_*.json`, `v1_msgym_scale_goldcomp_scaffold.json`).**

| Trained on | Evaluated on | Spectra / molecules | Validation NLL/token | Coverage at K=8 | Precision | Abstention |
|---|---|---|---|---|---|---|
| Orbitrap train spectra (31,294 spectra, 17,134 molecules), 6,000 steps | Orbitrap validation | 619 / 327 | 1.253 [1.197, 1.311] | 0.039 [0.023, 0.056] | 0.033 [0.022, 0.047] | 0.25 |
| the same model | QTOF validation (instrument never seen) | 897 / 659 | 1.331 [1.290, 1.368] | 0.020 [0.012, 0.028] | 0.019 [0.014, 0.025] | 0.39 |
| Scale train, 20,000 steps | Scaffold-held-out validation (Bemis–Murcko scaffold absent from every train molecule) | 683 / 367 | 1.165 [1.111, 1.222] | 0.051 [0.034, 0.072] | 0.036 [0.025, 0.047] | 0.26 |
| the same model | Molecule-disjoint validation (for reference) | 714 / 384 | 1.175 [1.125, 1.227] | 0.050 [0.033, 0.069] | 0.037 [0.026, 0.048] | 0.27 |

The scaffold holdout is not harder than the plain validation set here: MassSpecGym's split already separates
structures, and 367 of its 384 sampled validation molecules have a scaffold no train molecule has. The instrument
holdout costs about 0.08 NLL per token and halves coverage; QTOF requests also abstain more (0.39 against 0.25),
because fewer of them have a table formula in their precursor window. Split audit of `msgym-split-v1`
([JSON](../bench/results/ms2/msgym_split_v1_audit.json)): parts are disjoint by InChIKey block; the fraction of
molecules whose formula also occurs among the `fit` formulas is 0.56 for `rank` (same fold) and 0.20 for
`calibration` and `report`. No preprocessing statistic is fitted on data (the energy scale and clip are constants).
These are measurements of the current table-source model, not the release evaluation.

Timing on this GPU (not a quiet machine: CPU builds and tests ran alongside): a training step 0.17 to 0.19 s at
p50 for batch 16 (4,979 launches), a `generate` call about 0.06 s for 8 spectra (3,435 launches, 1 read); a
6,000-step run with its four evaluations takes 20 to 24 minutes.

## Baselines and targets (V0.7)

Declared on 2026-10-04, before any P8 optimisation and before any held-out result with the enumerating formula
source. Baselines are measured (V0 model config, N = 128, wgpu on the Radeon 860M, the machine not quiet: times
indicative, counts exact); targets are what P8 and P9 are judged against, not results.

Generation, warm, one call, `T = 22` (21 decode steps):

| K | B | p50 / p95 wall time | Spectra per second | Candidates per second | Launches | Reads | Memory estimate |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1 | 25.3 / 28.3 ms | 40 | 40 | 3,289 | 1 | 6.2 MiB |
| 1 | 8 | 38.9 / 39.9 ms | 206 | 206 | 3,352 | 1 | 19.3 MiB |
| 1 | 32 | 58.6 / 78.6 ms | 547 | 547 | 3,352 | 1 | 64.2 MiB |
| 8 | 1 | 32.7 / 37.6 ms | 31 | 245 | 3,436 | 1 | 9.8 MiB |
| 8 | 8 | 60.4 / 64.6 ms | 132 | 1,059 | 3,436 | 1 | 48.4 MiB |
| 8 | 32 | 206.8 / 214.3 ms | 155 | 1,238 | 3,436 | 1 | 180.9 MiB |
| 32 | 1 | 38.7 / 44.8 ms | 26 | 827 | 3,436 | 1 | 22.3 MiB |
| 32 | 8 | 200.1 / 208.8 ms | 40 | 1,279 | 3,436 | 1 | 148.5 MiB |
| 32 | 32 | 666.4 / 672.6 ms | 48 | 1,537 | 3,436 | 1 | 581.2 MiB |

The call is launch-bound at small B·K (about 25 ms for 3,300 launches whatever the work) and work-bound from about
B·K = 64 trajectories upward; the decode loop is 92 to 95% of it (155 launches and 173 allocation calls per step).
A training step at batch 16 on real data takes 0.17 to 0.19 s (4,979 launches, 0 reads between reports).

| Quantity | Baseline | Target | Judged in |
|---|---|---|---|
| `generate`, B = 8, K = 8, warm p50 | 60 ms | at most 30 ms | P8.2–P8.4 |
| Launches per decode step | 155 | at most 80 | P8.3 (O4) |
| Allocation calls per decode step | 173 | 0 in the warmed loop | P8.2 (O2), the open P2 acceptance item |
| Device reads per `generate` / per non-report training step | 1 / 0 | 1 / 0 (kept) | P8.7 |
| Reserved bytes over 200 fixed-shape and alternating calls | flat | flat (kept) | P8.7 |
| Reserved bytes against the estimate, generation | ratio 1.0 to 1.7 by shape | within [0.67, 1.5] | P8.2 |
| Training step, batch 16, real data | 0.17–0.19 s | at most 0.10 s | P8.3–P8.4 |
| Held-out formula recall at F = 4, structure-disjoint data | 0.23 (train-formula table) | at least 0.50 with the enumerating source | P4.3, P9.2 |
| Held-out target coverage at K = 8 (conditional) | 0.050 [0.033, 0.069] | at least 0.10 | P9.2 |
| Held-out containment precision of finished distinct candidates | 0.037 [0.026, 0.048] | at least 0.10 | P9.2 |
| Real spectra against the shuffled control, held-out NLL per token | lower, non-overlapping intervals | kept at every scale-up | P7.9, P9.3 |

The quality targets are deliberately modest: they are the first numbers at which candidate lists would start to be
usable for review, not a claim that the model reaches them. A result is reported whichever way it falls.

### Progress against the targets (O4)

Measured on 2026-10-04 after the fused sampler step, the position-batched teacher pass and the transpose-free
SSM step (architecture §3.9), same machine and model config as the baselines. Counts are exact. Times are of two
kinds: the profile driver's warm p50 ([JSON](../bench/results/ms2/profile_p8_fused_wgpu_radeon860m.json)), which
was taken in a different window from the baseline and is indicative, and a paired comparison
([JSON](../bench/results/ms2/p8_fused_step_ab_wgpu_radeon860m.json)) in which the old and the new form run
interleaved in the same minutes (`examples/ms2_launch_tally.rs --time`), which is what the ratios rest on.

| Quantity | Baseline | Now | Target | Met |
|---|---|---|---|---|
| Launches per decode step (wgpu) | 155 | 65 | at most 80 | yes |
| Launches per `generate` call, K = 8 | 3,436 | 1,553 | — | — |
| Allocation calls per decode step | 173 | 81 | 0 | no |
| `generate`, B = 8, K = 8, warm p50 | 60 ms | 31 ms | at most 30 ms | no (1 ms short) |
| `generate`, B = 1 / 32, K = 8, warm p50 | 33 / 207 ms | 12 / 121 ms | — | — |
| Launches per training step (B = 8, 2 slots) | 4,946 | 920 | — | — |
| Device reads per `generate` / per non-report training step | 1 / 0 | 1 / 0 | 1 / 0 | yes |

Paired ratios on wgpu, old form over new form: generation 1.8 to 2.1 times faster across B = 1, 8, 32 and
K = 1, 8, 32 (composed reference step against the fused step, one process); training step 2.7 times faster at
B = 8 with 2 slots, 1.4 times at B = 32 with 2 slots, and 1.4 times at the real shape, B = 16 with 16 slots
(about 215 ms to about 155 ms on synthetic spectra: that shape is bound by work, not launches, so the training
target of 0.10 s is not reached). On the CPU runtime the same changes take a generation call from 3,636 to 1,689
launches (paired, 1.8 times faster at B = 1 and at B = 8, K = 8: 286 ms to 159 ms) and a training step from
5,808 to 1,563 launches.

Where the remaining time is, from the timed tally at the real training shape: the general matrix product at
small shapes, the broadcast adjoints (`sum_dim`), the lookup adjoint (22 ms, down from 67 ms), the scan's
backward pass and peak selection (12 ms per step, whose lanes still wait on one guarded load per comparison).
In generation the Mamba-3 mixer step is about 44 of the 65 launches of a decode step and was left as it is.

### Device-bound kernels on the Radeon (2026-10-04, second pass)

With the launch counts down, a warmed decode step on wgpu is no longer priced by its launches: a pipelined launch
costs about 10 µs there, and removing launches by folding a norm, a bias or a residual into a hand-written
projection made every shape slower (a unit of such a kernel walks its input one memory round trip at a time; the
tuned block product does not), so that fold was measured and dropped. What a step costs is a handful of kernels
whose lanes are long or uncoalesced. Changed:

- **Mixer step on a device with planes** (`tensor::ops::mixer_step`): the fused incremental Mamba-3 step, as first
  written, gave each state row and each `(batch, head)` one unit. On the Radeon that made generation 1.7 times
  *slower* than the composed step (B = 8, K = 8: 58 ms against 33 ms) although it launches a third as much. Both
  kernels now have a plane form — a row's segment of a plane takes its vectors side by side and reduces with a
  butterfly, as the plane RMS norm does — and the unit form stays for the CPU runtime. One block step at the
  decoder's shape (64 rows, 4 heads of 64, state 32): 1,000 µs before, 208 µs after, composed 368 µs.
- **Pointer scores** (`step_logits_pack`): a pointer score is `query · (key + E_residual[r])`. For the 23
  conditioning queries both halves are known before the step that reads them, so the head row now carries
  `key · E_cond` (stored with the key when an atom is added) and `query · E_residual`, and `E_cond · E_residualᵀ` is
  a `[23, 8]` table built once per call. Only the row's own query is a dot product per step: 16 per row instead of
  384. The 27 head biases are added by the same kernel instead of a launch of their own.
- **Carry freeze**: two carries to a launch whatever their shapes (3 launches per step for two layers, from 4).
- **Lookup adjoint** (`lookup_backward`): on a device with planes the row scan is split into groups of 64 rows
  with one reduction over the groups; a lane no longer scans every row of the batch (21 ms to 6 ms per training
  step, queue drained between launches).
- **`ms2_launch_tally --timed`** now drains with a read: on wgpu a bare synchronisation returns before the queue
  has run, and the per-site times above it were charged to the wrong sites.

Paired on wgpu (Radeon 860M), the committed tree against this one, interleaved in the same minutes, three pairs
each, medians of 5 rounds of 5 calls:

| `generate`, warm | Before | After | Ratio |
|---|---|---|---|
| B = 1, K = 8 | 19.7 ms | 13.5 ms | 1.46 |
| B = 8, K = 8 | 37.0 ms | 23.1 ms | 1.60 |
| B = 8, K = 32 | 149 ms | 104 ms | 1.43 |
| B = 32, K = 8 | 177 ms | 130 ms | 1.1 to 1.4 (pairs disagree) |

Launches per `generate` call (B = 8, K = 8): 1,553 to 975 on wgpu, 956 to 675 at the pinned CPU shape (22 per
decode step there, from 36; decoder init 26, from 13: the per-call pointer products). The B = 8 target of 30 ms is
met. A training step at the real shape (B = 16, 16 slots, synthetic spectra) moved from about 159 ms to about
145 ms, with one of three pairs showing no gain: within the noise of this machine, so not claimed (the next
section is the training work). On the CPU runtime `generate` at
B = 8, K = 8 is 89 ms (159 ms before the mixer step was fused).

Still open, by drained time at the real training shape: the transposed products of the backward pass (70
launches, about 0.5 ms each), five products that run on the row-tiled plan at 3 ms each, the scan's backward pass
(2.5 ms a launch), and in generation the state update of the mixer step, which moves 8 MB a layer at B·K = 64
because `last_u` is stored at full size although it is an outer product.

### Training on the Radeon: the teacher pass on occupied slots (2026-10-04, third pass)

A training step on wgpu is bound by arithmetic, not launches: about 950 launches at roughly 15 µs each against
120 to 160 ms a step, and the matrix products of the backward pass alone are a third of the drained time, running
within a factor of two of what the tuned block kernels reach on this device. So the lever is the amount of
arithmetic. On the real exports about half of the `B * 16` target slots are empty (`pilot_train`: 49.0% occupied,
median 127 of 256 rows per batch of 16 labeled spectra; `orbitrap_train`: 50.4%), and the padded teacher pass ran
every decoder layer, head and adjoint over them.

- **Compact teacher pass** (`TargetBatch::compact`, `Ms2Decoder::teacher_grouped`, `Ms2Trainer::forward_state`):
  the occupied slots of each spectrum are regrouped on the host into *virtual spectra* of 4 slots, padded to a
  multiple of 8 virtual spectra (row counts in steps of 32, so the kernels and the tuned products meet few
  shapes). The decoder scores those rows with each virtual spectrum taking its spectrum's memory, mask and
  conditioning embedding: keys and values are projected once per spectrum and gathered, and the gather's adjoint
  sums a spectrum's gradient over its virtual spectra. The graph loss keeps `B` as its divisor. A row's result
  depends on its tokens and its spectrum only, so the losses and gradients are those of the padded pass up to
  rounding (`tests/ms2_experiment.rs`: the repacking keeps every occupied slot under its spectrum; four reported
  steps agree, the first to 1e-5). The padded pass remains for the evaluation paths, which read a result per
  slot, for a batch that would not shrink, and behind `MAMBA3_MS2_COMPACT_TEACHER=0` /
  `train::set_compact_teacher`. Groups of 2 and 4 slots measured alike, 8 slower.
- **Memory**: `Ms2MemoryEstimate::training` is unchanged and now bounds the compact pass from above (it is for a
  batch with every slot occupied); `tests/ms2_footprint.rs` reconciles it with the padded pass.
- **Benchmark**: `ms2_launch_tally --mode train --data <export> --table <table>` times labeled batches of a
  real export in turn. The synthetic spectra of the default mode are unlabeled — every slot empty — and would
  overstate the gain.

Paired on wgpu, the committed tree against this one, `pilot_train` at B = 16 with 16 slots, five interleaved
pairs of 7 rounds of 8 steps, medians: 159 to 169 ms before, 119 to 123 ms after — **1.34 times faster**. All ten
runs were taken with the package power-capped (about 21 W, GPU at 100%, CPU near 1.8 GHz), which is the state a
long training run is in; in the short boosted state after idling the same step takes about 66 ms, so only pairs
taken in one state compare. Warmed CPU pin: 1,564 launches a step at the pinned shape (1,554 padded).

Tried and left: projecting keys and values per virtual spectrum instead of gathering them (no measurable
difference; the gathered form is kept because it is the smaller amount of arithmetic and needs no stand-in
encoder output). Still open, by drained time above the launch floor on a real batch: the transposed products of
the backward pass (72 launches, about 34 ms), 79 same-shape additions of 0.5 to 2 MB tensors (residuals and
gradient accumulation, about 16 ms), the scan's backward pass (10 ms), the gradient concatenation of the
projection's bands (7 ms). Trace lengths fill 28% of rows times `T`: the next section packs them.

### Training on the Radeon: ragged sequences along the time axis (2026-10-05)

The slots-only pass above still ran every occupied row over the full horizon of 22 positions, and a trace is 12.6
tokens long on average: on `pilot_train` a batch of 16 spectra holds about 141 traces and 1,790 tokens, against
5,632 positions padded and about 3,550 with the empty slots left out.

- **Packing** (`TargetBatch::pack`, `PackedTargets`): the traces of a spectrum are laid end to end, longest first,
  first fit, in rows of two horizons (44 positions), a row holding one spectrum's traces only because a row
  attends one spectrum's memory. Packed rows come in multiples of 8 and trace rows in multiples of 32. Fill is
  about 75% (2,400 positions a batch); what is left over is the unfilled end of each spectrum's last row, and row
  lengths of 22, 33, 66 and 88 fill no better.
- **Decoder** (`Ms2Decoder::teacher_packed`): the layers — nearly all of the pass's arithmetic — run over the
  packed rows through `Mamba3Block::apply_with_state_masked`, with a reset flag where a trace begins after another
  in its row, so the state, and with it everything a position can see, is cut at the trace's start. A cell's step
  embedding is its position inside its own trace. Keys and values are projected once per spectrum and gathered
  per row. The layers' output is then taken back to one trace per row (`Var::ms2_take_rows`: the map is injective,
  so its adjoint is a gather through the inverse map rather than an accumulation) and the heads, the grammar
  replay and the loss run as before on those rows.
- **Trainer**: `train::TeacherPass` is `Ragged` by default; `MAMBA3_MS2_COMPACT_TEACHER=slots` selects the
  slots-only pass and `=0` the padded one (`train::set_teacher_pass`). Evaluation keeps the padded layout.
- **Checks** (`tests/ms2_experiment.rs`, CPU and wgpu): every trace is packed whole, once, in a row of its own
  spectrum, and the two maps invert each other; a trainer run under each of the three layouts reports the same
  losses over four steps, the first to 1e-5 (the fixture packs two traces into one row, so the reset is
  exercised on a rotational decoder).

Paired on wgpu, `pilot_train` at B = 16 with 16 slots, six interleaved triples of 7 rounds of 8 steps, the
package at 24 W throughout, medians: committed tree 146 to 155 ms, slots-only 108 to 122 ms, ragged 85 to 90 ms —
**1.75 times faster than the committed tree**, 1.3 times faster than slots-only. Warmed CPU pin: 1,561 launches
a step at the pinned shape.

Left for the next section: rows that mix spectra, and the encoder, by then about half of the step's positions.
The heads still run over `rows * (T - 1)` positions (cheap: a few narrow products).

### Training on the Radeon: packed encoder scans and mixed scan rows (2026-10-05, second pass)

- **Encoder** (`Ms2Encoder::encode`, `models::ms2::ragged`): a spectrum keeps a third of its 128 peak slots on
  the real exports (`pilot_train`: 42 of 128 on average, median 30; a batch's longest spectrum is 128 in most
  batches, so truncating the batch would gain nothing), and the encoder's four scans per pass ran every slot. The
  encoder is two one-directional blocks per layer, the backward one fed the peaks reversed within each
  spectrum's length, so both see a spectrum's peaks first and its padding last: each scan now runs over the kept
  prefixes laid end to end in rows of one peak capacity, any spectrum in any row, with a reset where a spectrum
  begins (`RowPacking`, rows in multiples of 2). The host's peak counts bound what the device keeps, so the
  layout needs no read; `DeviceSpectra` carries them. Everything per-position — features, embedding, the
  reversal gathers, the residual, norm, memory and pool — stays in the padded layout. Applies to generation as
  well. `MAMBA3_MS2_PACK_PEAKS=0` / `encoder::set_pack_peaks` keep the padded scans.
- **Decoder scan rows that mix spectra** (`TargetBatch::pack`): the ragged teacher pass kept a row to one
  spectrum because a row attends one spectrum's memory, which left a quarter of the positions unfilled. Each
  layer now works in two layouts of the same cells: the block scans rows of any spectrum's traces (nearly full,
  rows in multiples of 4), and cross-attention regroups the cells into rows of one spectrum each (the earlier
  layout), two gathers a layer.
- **Row gather** (`ms2::lookup`): moves a vector at a time and loads unconditionally (an out-of-range id reads
  row 0 and stores zero), instead of one element per lane behind a branch. It carries every pack and unpack.
- **Checks** (CPU and wgpu): `tests/ms2_encoder.rs` — the packed scans give the padded memory, mask and pool, and
  the same parameter gradients, on a batch with an empty spectrum, spectra sharing a row, ineligible peaks inside
  a count and a spectrum over capacity; `RowPacking` places every segment whole and once. `tests/ms2_experiment.rs`
  — both decoder layouts hold every trace once, name each other's cells, and the three teacher layouts report
  the same losses with two spectra in one scan row.

Paired on wgpu, `pilot_train` at B = 16 with 16 slots, interleaved triples of 7 rounds of 8 steps at 20 W
(five of six; the sixth caught the boosted state and is left out), medians: committed tree 154 to 160 ms, the
first ragged pass 92 to 99 ms, now 68 to 77 ms — **about 2.1 times faster than the committed tree**. In
separate pairs the encoder scans alone took a step from about 89 ms to about 75 ms, and the mixed scan rows from
about 77 ms to about 72 ms; the gather kernel's change is inside the noise. Warmed CPU pin: 1,604 launches a
step at the pinned shape. `generate` on the synthetic spectra (128 peaks each, nothing to pack) is unchanged;
its gain on real spectra was not measured.

Still open, by drained time above the launch floor: the transposed products of the backward pass (71 launches,
about 23 ms), the scan's backward pass (8 ms), and about 200 elementwise launches over the padded encoder layout
and the heads.

### Training on the Radeon: the matrix products of the backward pass (2026-10-05, third pass)

The tuner's log (`MAMBA3_TUNE_LOG=1 MAMBA3_TUNE_CACHE=0`) on a real batch separates the products into three
kinds. The wide ones — the mixers' 840-column projection and its two adjoints — run at the kernels' peak in
every orientation (about 1,000 GFLOP/s in the boosted state) and were left alone. Two kinds ran far below it:

- **Weight gradients with a small output** (`matmul::split_k_factor`): `Xᵀ G` for a `d x d` or `2d x d`
  projection is a `128 x 128` or `256 x 128` output over all the rows of the pass as `k` — two to four block
  tiles walking 1,000 to 4,000 rows, at 220 to 550 GFLOP/s with the rest of the device idle. Split-K existed for
  `k >= 4096` only, which the compact passes no longer reach. It now also takes an output of at most `256 x
  128` cells from `k = 1024`, in slices of about 256 rows (the largest count that divides `k` with slices of at
  least 128), summed in one reduction; slices of 128 and 512 measured no better. Devices with planes only, as
  before; `MAMBA3_SPLIT_K=0` turns all of it off, `MAMBA3_SPLIT_K_SLICE` sets the slice for measurement.
- **Attention products over the memory** (`Ms2Decoder::packed_hidden`): the memory is `1 + N = 129` slots, not
  a multiple of 4, so the batched products with the memory as `n` or `k` — scores, context and their four
  adjoints per layer — could not use the vectorised block kernels at all; the scores ran on the row-tiled plan
  at 73 GFLOP/s. The packed teacher pass now appends three masked slots of zeros to the memory and its mask (a
  masked slot takes no attention weight). The same products then run at 180 to 370 GFLOP/s, and the elementwise
  passes over `[.., 132]` vectorise too.

Drained time of one training step on a real batch (queue drained between launches, same minutes): all matrix
products 53 ms before, 42 ms with the split, 31 to 36 ms with the padded memory as well; every launch of the
step 283 ms, 264 ms and about 229 ms. Paired end to end on wgpu, `pilot_train` at B = 16 with 16 slots, six
interleaved triples at 20 W: about 70 ms a step before, about 62 ms with the split, about 62 ms with both (the
padded memory's gain does not show end to end within the ±4 ms of these runs; it is kept as the smaller amount
of device work). Against the committed tree, in the capped state: about 157 ms to about 62 ms, **2.5 times
faster**. `tests/matmul_paths.rs` checks the new split regime against a host product (including a prime `k`,
which takes the direct kernel); the teacher-layout equivalence test covers the padded memory. Warmed CPU pin:
1,609 launches a step (the split is not taken on the CPU runtime; the padding adds five launches).

Still open among the products: the batched attention products remain a third of the kernels' peak (44 x 32
tiles are too small for the block shapes), and the scan's backward pass and the elementwise launches are now
the larger share.

### Training on the Radeon: the batched attention products (2026-10-05, fourth pass)

A step issues twelve of these (two decoder layers, scores and context with their four adjoints), about 70 MFLOP
each at 192 matrices of `44 x 132 x 32`. Three changes, in order of what they turned out to be worth:

- **The weights between the two products** (`fused::masked_softmax`, `Var::masked_softmax`): scale, key mask
  and softmax were eight launches forward (`mul_scalar`, an `expand` of the `[b, m]` mask to the scores' shape,
  `mask_logits`, and a softmax of five) and their adjoints back, each a pass over `[b, heads, 44, 132]`. They
  are now one launch each way, a plane per row with the maximum and the sum as plane reductions, reading the
  `[b, m]` mask in place; the adjoint needs only the weights (`dx = scale * w * (g - sum(g w))`, zero at a
  masked slot as the composed rule has it). 34 launches fewer a step (wgpu 1,085 to 1,051; CPU pin 1,609 to
  1,575).
- **A block fitted to the matrix** (`matmul::Plan::BlockFit`, `fit_shapes`): the fixed block shapes cover
  `44 x 132` with three `64 x 64` cubes and `44 x 32` with a third of one. A fitted block is the output rounded
  up to the register tile, one cube per matrix, on the transposed block kernel with its staging walked by flat
  element and guarded (a fitted tile need not divide across its units). Offered to the tuner for a batch whose
  best fixed cover wastes a quarter or more. It wins four of the six attention shapes at 400 to 470 GFLOP/s
  against 310 to 360 for the padded cubes; the plain scores product stays on the vectorised `64 x 64` block.
  These products are at a few dozen FLOPs per byte moved, so the kernels' 1,000 GFLOP/s is not on offer.
  `MAMBA3_FIT_BLOCK=0` withdraws the candidates.
- **Pipelined tuner probes** (`matmul::probe_reps`): a probe was one launch and a synchronisation, and a
  synchronisation costs more than a small product. Every candidate for the attention shapes measured 200 to 350
  GFLOP/s whatever it was, and the cached winner was chance (the row-tiled plan at a third of the best, some
  runs). A probe now issues enough launches to put about a gigaflop between two synchronisations.
- The head split of keys and values is taken per spectrum, before the gather per attention row
  (`Ms2Decoder::split_heads`, `attend_heads`): the copy that swaps the contiguous axis moves a third of the
  bytes. Not resolved end to end.

Paired end to end on wgpu, `pilot_train` at B = 16 with 16 slots, three binaries interleaved, 3 x 20 steps a
run: committed tree 60 to 63 ms a step in the capped state, fitted blocks and probes alone about 61, with the
fused weights 45 to 51; in the boosted state 45 to 47 against 38 to 39. **About 1.2 times faster**, nearly all
of it the fused weights. `tests/autograd.rs` checks the fused weights against the composed chain (values,
gradients, a row with nothing legal, rows shorter than a plane) and against central differences;
`tests/matmul_paths.rs` runs the six attention shapes in every stored layout against a host product, and under
`MAMBA3_TUNE_CHECK=1` on wgpu every fitted candidate is compared with the simple kernel before it can win.

**The training memory estimate, re-derived** (`Ms2MemoryEstimate::training`). The fused weights lowered the
step's real peak (214 to 180 MB at B = 4 on wgpu) and took the measured peak / estimate ratio of
`tests/ms2_footprint.rs` to 0.63, under its band of [0.67, 1.5]; it had already drifted from 0.95 to 0.69 to
0.75 over the earlier passes. The retained terms were re-derived from what a step holds rather than rescaled.
Live bytes probed at the boundaries of one step show the peak is the end of the backward pass and is exactly
three parts: the fixed state (13.7 MB), the forward values the tape keeps (69.8 MB at B = 4) and one gradient
buffer per node of the tape (96.3 MB; `Var::backward_retain`). The old items counted every forward tensor
twice ("forward plus gradient"), which is wrong in both directions: a value nothing captures is freed as soon
as its consumer has run (the eleven embedding buffers are one), and gradients exist for nodes whose values are
not kept. Stale shapes went too: the largest item still held a `[positions, A, d]` pointer-key tensor the head
no longer builds (66 of its 78 MB), and `head_scratch` a `[rows, 19, A]` table. The forward items are now the
kept values by region, each checked against its probe (decoder mixer 3,462 floats a cell and layer against
3,462 measured; attention 1,761,152 floats a layer, exact), and `activation_gradients` is the list of retained
gradients by shape (89.6 MB against 91.8 MB measured without the parameters'). Encoder scans are sized over
`min(n_raw, N)` kept peaks, as the packed layout runs them. Ratio now: **0.99, 0.90, 0.90** at B = 4, 8, 16
on wgpu and 1.37, 1.25, 1.26 on the CPU runtime (which holds about 70 MB more live at B = 4 than wgpu does),
band unchanged. Not modelled: the scan's working set for more than one chunk (the V0 horizon is one), and the
unpacked encoder layout (`MAMBA3_PACK_PEAKS=0`), which runs `N` cells a spectrum whatever was kept.

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

Reviews of 2026-10-03/04 (all by `codex exec`, read-only; implementation by opencode
`opencode-go/muse-spark-1.3-contributor`; every review is stored under `docs/reviews/`):

| Subject | Review file | Verdicts in order | State now |
|---|---|---|---|
| MassSpecGym export tools | [MS2_MSGYM_EXPORT_CODEX_REVIEW.md](reviews/MS2_MSGYM_EXPORT_CODEX_REVIEW.md) | 3 findings | 2 fixed, 1 declined with reason; fix not re-reviewed |
| Profile driver and harness (P2.6), V0.6 tests, first enumeration reference | [MS2_P2_P4_HOST_CODEX_REVIEW.md](reviews/MS2_P2_P4_HOST_CODEX_REVIEW.md), part A of [MS2_V1_INTEGRATION_CODEX_REVIEW.md](reviews/MS2_V1_INTEGRATION_CODEX_REVIEW.md) | reject, reject, reject (3 findings) | third review's findings fixed and verified on CPU and GPU; not re-reviewed |
| V1 architecture, sections 1–2 and 3–4 | [MS2_V1_ARCHITECTURE_CODEX_REVIEW.md](reviews/MS2_V1_ARCHITECTURE_CODEX_REVIEW.md) | not ready ×3 | text revised after each; remaining points settled in the code reviews |
| Candidate compositions, schema version 2 | [MS2_P4_CANDIDATES_CODEX_REVIEW.md](reviews/MS2_P4_CANDIDATES_CODEX_REVIEW.md), part B of the integration review | reject, reject (4 partly), **accept** | closed |
| Formula enumeration: host reference, pruning, kernel twins, kernels | [MS2_P4_ENUM_PRUNING_CODEX_REVIEW.md](reviews/MS2_P4_ENUM_PRUNING_CODEX_REVIEW.md) | accept-with-fixes, reject, reject, reject | last review's 7 findings fixed and verified on CPU and GPU; not re-reviewed |
| Ion assignment: host reference, then kernels | [MS2_P4_ION_HOST_CODEX_REVIEW.md](reviews/MS2_P4_ION_HOST_CODEX_REVIEW.md), last sections of [MS2_P5_P6_IDENTITY_HOST_CODEX_REVIEW.md](reviews/MS2_P5_P6_IDENTITY_HOST_CODEX_REVIEW.md) | accept-with-fixes, accept-with-fixes, reject (kernels) | kernel findings fixed and verified on CPU and GPU; not re-reviewed |
| Graph identity and allocation: host twins, then kernels | [MS2_P5_P6_IDENTITY_HOST_CODEX_REVIEW.md](reviews/MS2_P5_P6_IDENTITY_HOST_CODEX_REVIEW.md) | reject, identity accept-with-fixes / allocation reject, identity reject / allocation accept-with-fixes | findings fixed and verified on CPU and GPU; not re-reviewed |
| Decoder capacities, validation, work report (P5, P6.1) | part C of the integration review | accept-with-fixes | the one finding fixed |
| Enumeration integrated into generation and training | part D of the integration review | reject (9 findings) | fixed, verified on CPU and GPU; not re-reviewed |
| Ranking and packing kernels | part E of the integration review | reject (5 findings) | fixed, verified on CPU and GPU; not re-reviewed |
| Top-F selection, bounded enumeration dispatch, allocation/identity/packed integration, dtype guard | [MS2_V1_INTEGRATION2_CODEX_REVIEW.md](reviews/MS2_V1_INTEGRATION2_CODEX_REVIEW.md) | accept-with-fixes, accept-with-fixes, reject (8 findings: packed validation rejects legal enumeration counters, bf16 allocation and packed scoring not in f32, packed evidence missing, a finished graph without formula accepted), reject (dtype gate checks the configuration, not the element type) | fix task written, starts after the assignment follow-up |
| Re-review of fix rounds: graph identity, ion assignment, enumeration, profile harness | [MS2_V1_REREVIEW_CODEX_REVIEW.md](reviews/MS2_V1_REREVIEW_CODEX_REVIEW.md) | accept-with-fixes, accept-with-fixes, reject (the lane does not enforce the visit bound the chunk sizing assumes; packed counter rule), accept-with-fixes | same fix task |
| Standalone modules: assignment head, reranker and calibration, fingerprint head, baseline encoders, split and table tools, Python bindings | [MS2_V1_MODULES_CODEX_REVIEW.md](reviews/MS2_V1_MODULES_CODEX_REVIEW.md) | accept-with-fixes, reject, reject, accept-with-fixes, reject, reject | reranker, calibration, fingerprint, baselines and tools fixed (CPU), not re-reviewed; the two assignment-head findings and the bindings' missing API are open |

Not reviewed at all yet: the integration of the assignment head and evidence, and the host formula-evidence
experiment.

