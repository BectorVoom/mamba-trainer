# Original independent review — Claude Code Opus 5.5

This is the reviewer response to the pre-revision documents. It is advisory, not an implementation specification. Some suggestions were rejected or qualified; see [disposition](../MS2_SUBSTRUCTURE_REVIEW.md). Claims about uninspected project notes and alternative cache layouts must not be treated as verified repository facts.

# Review: MS2→substructure Mamba-3 design and task plan

Scope: read-only review of `docs/MS2_SUBSTRUCTURE_DESIGN.md` and `docs/MS2_SUBSTRUCTURE_TASKS.md`. No code or files were changed. All numbers below are arithmetic from stated assumptions. None of them are measurements.

## 0. Verdict

The plan is careful about residency, precision and honesty. Its main weakness is that the learning problem is underspecified. Nothing defines which subgraph is the target for a given spectrum, which traversal order is used, or what attachment ports and hydrogens mean. As written, the generation likelihood, the ranking, and the calibration are not well defined.

The second problem is complexity in the critical path. Several pieces sit before the first trainable slice even though they are optimizations or later experiments:

- formula tables
- two-limb arithmetic
- boundary-ambiguity retries
- sparse relations
- slots
- beam ancestry
- incremental GNN
- typed arenas
- concurrent leases

Most of this can be removed from v1 by fixing explicit bounded domains. The on-device inference requirement stays intact.

The largest latency risk on this repository's wgpu path is per-step launch submission in the decoder, not memory. Project notes record that wgpu rounds are dominated by host submission. The plan never budgets launches per decode step, and T=128 inflates that cost.

---

## 1. Priority table

Class key: **B** = correctness/feasibility blocker, **I** = important simplification or risk reduction, **O** = optional optimization.

| # | Issue | Class | Change | Where |
|---|---|---|---|---|
| 1 | No target distribution for spectrum-conditioned generation | B | Define a versioned target recipe q(g \| spectrum) built from parent fragmentation and peak matching | §8, new P0.7, new P1.8 |
| 2 | Graph likelihood depends on traversal order | B | Canonical traversal for training. Rank with a separate reranker, not raw likelihood | §3.5, §3.6, P0.8, P7.1 |
| 3 | Ports, H and valence are redundant or ambiguous; `MARK_ATTACHMENT` adds an unidentifiable action | B | Atom token includes (element, charge, H, valence state). Ports are *derived* as open valence at STOP, which equals the cut bonds in the target | §1, §3.5, P0.3 |
| 4 | "Captured probability" of top-F is undefined without a full normalizer | B | Normalize over **all** enumerated in-domain formulas plus an explicit `OTHER` logit | §3.2, P4.1 |
| 5 | Fixed-point section is self-contradictory (bound direction, widen vs. ambiguous, "higher-resolution" fallback) | B | Make the integer rule *the specification*: u32 µDa, host-computed integer windows, documented slack. The oracle replicates it bit-exactly | §4.2, P1.3, P2.5 |
| 6 | Calibration target unnamed; likelihood ranking contradicts "presence probabilities need not sum to 1" | B | Name the target (candidate ⊆ parent under a defined match), calibrate the reranker output per size bin, refit after any model or search change | §3.6, §8, P7.9 |
| 7 | Teacher forcing without the inference legality masks gives a different normalization | B | Apply identical masks in training using offline per-step bitsets. Require a device-mask parity test | §3.5, §8, P5.3, P7.2 |
| 8 | Decoder scan (training) vs. recurrent step (inference) parity not required | B | Add a teacher-forced vs. stepped logits parity test covering all carries | P3.5 → also P5.2 |
| 9 | T=128 inconsistent with A=32; incomplete-candidate path exists only because T is arbitrary | I | Derive T = 2 + A + R_max and enforce R_max by mask. Truncation becomes impossible by construction | §3.5, §8 caps, P5.8 |
| 10 | Launch count per decode step not budgeted | I | Make launches/step and launches/call first-class metrics. Reduce T. Amortize over B·K | §4.1, §7, P2.7, P8 |
| 11 | Beam vs. sampling undecided; beam forces double banks and ancestry gather of all state | I | v1 uses K independent counter-RNG trajectories (no ancestry, single bank). Beam becomes a later experiment | §3.5, §6, K08/K09, P5.4/P5.6 |
| 12 | Formula tables / meet-in-the-middle in the critical path | I | Closed-form-hydrogen enumeration over a bounded heteroatom grid: exhaustive within domain, no tables | §3.2, K03, P1.6, P4.1 |
| 13 | Two-u32 limb arithmetic | I | Not needed with a u32 µDa domain cap. Defer | §4.2, P2.5, P2.4 |
| 14 | Slots and tiled online softmax at N≤512 | I | v1 cross-attends to all encoder outputs, with per-layer K/V cached once per spectrum | §3.4, K06, P4.6/P4.7 |
| 15 | Incremental GNN prevents parallel teacher forcing | I | Atom memory = decoder hidden state at the atom's creation step (gather in training, store in inference) | §3.5, K07, P5.2 |
| 16 | Segmented valid-length reversal kernel | I | Pads become identity transitions (Δ=0, inputs selected to 0), then a plain flip | §3.1, K02, P3.1 |
| 17 | Sparse relations on the critical path; float atomics not portable | O/I | Move to post-MVP. Build destination and source edge lists (CSR + CSC) so backward needs no atomics | §3.3, K05, P4.4/P4.5 |
| 18 | MIMO rank 4 as an experiment | O | Gate O7 on a native MIMO kernel. The current reference is an R² SISO decomposition | §3.1, O7 |
| 19 | Typed arenas, concurrent leases, sub-buffer offsets in P2 | I | v1 uses preallocated tensors reused in-loop plus existing counters. Single stream. Defer arenas and leases | §6, P2.3, P2.8 |
| 20 | Contrastive loss needs a structure encoder; `L_local_evidence` undefined | I | Drop both from MVP. Define them or delete them | §8, P7.2 |
| 21 | No Kekulé-variant identity policy; on-device aromaticity perception | I | Use canonical Kekulé targets. No on-device aromatic perception in v1. Measure residual duplicates offline | §3.6, P6.1/P6.2 |
| 22 | Existing binding hazards (per project notes: Fortran-ordered inputs scrambled; GIL held during device waits) | I | MS2 bindings enforce C-contiguity and release the GIL. Parity tests include non-contiguous arrays | P6.6 |
| 23 | Silent dropped launches observed on wgpu Metal for some candidates (project notes) | I | Poison all new-kernel outputs with NaN or sentinels in tests. Validate autotune winners | §7, P2.8, P8.5 |

---

## 2. Scientific feasibility, supervision and identifiability (blockers)

### 2.1 Define the target distribution (new §8.1, P0.7, P1.8)

"Pretrain on connected parent subgraphs" and "confidence-weighted fragment labels" do not define what the spectrum-conditioned decoder should put probability on.

Proposed recipe, versioned and built offline on the host as part of data preparation:

1. **Parent → fragment candidates.** Enumerate connected subgraphs by bond breaking (MAGMa-style) up to depth D and size A. Restrict to heavy-atom count ≥ A_min so trivial fragments are excluded.
2. **Fragment → ion hypotheses.** Each fragment yields neutral formula F_sub plus `ports` (open valence units). Ion formulas are F_sub + H·(ports + δ) ± charge carrier, with δ ∈ [−δ_max, δ_max] set by the domain.
3. **Peak matching.** Match ion hypotheses to peaks with the integer rule from §4.2.
4. **Weights.** Use w(g) = Σ_{peaks p matched by g} I_p / |{g' matching p}|. This splits ambiguous peaks.
5. **Target.** Normalize the weights into q(g \| spectrum). Training minimizes E_q[−log p(g \| spectrum)], which is forward KL to q. Store the matched peak IDs with each target.

This makes "generation approximates q" a concrete, testable statement. It also makes the evidence and supporting-peak outputs consistent with supervision.

Record in P0.5/P0.6 that q is an assumption-laden proxy. It contains no rearrangement or hydrogen-shift chemistry beyond δ_max. Metrics against q (fragment-assignment accuracy) must be kept separate from containment metrics against the true parent, as §8 already requires.

### 2.2 Canonical traversal (§3.5, new P0.8)

The log-likelihood of one linearization is not the likelihood of a graph. "Multiple valid traversals" (P7.1) and "rank by size-normalized likelihood" (§3.6) are incompatible.

- v1 should use an offline canonical DFS. Root and atom order come from canonical ranks computed with RDKit. Each ring closure is emitted immediately after the atom that closes it, so `u` is always the most recently added atom and only one pointer (`v`) is needed.
- Ranking and calibration use a separate reranker (§2.5). Likelihood remains one input feature.
- Random-order augmentation becomes an experiment, valid only alongside the reranker.

### 2.3 Atom token and derived ports (§1, §3.5, P0.3)

The current grammar predicts H state, ports through `MARK_ATTACHMENT`, and bonds. These are linked by valence, so the masks must prevent inconsistent combinations, and port positions are weakly identifiable from a spectrum.

Proposed grammar v1:

```text
START
ADD_ATOM(atom_type, bond_order_to_parent, parent_ptr)   # first atom: root, no bond
CLOSE_RING(v_ptr, bond_order)                           # u = last added atom
STOP
atom_type ∈ versioned vocabulary of (element, formal_charge, H_count, valence_state)
ports(atom) := valence_state − H − Σ bond orders   # derived at STOP, in valence units
```

Properties:

- **Ports equal cut bonds in the target.** If valence_state is taken from the parent atom, open valence after the subgraph is exactly the cut-bond valence. No attachment supervision is needed beyond the atom token.
- **STOP is always legal** once size ≥ A_min, because the mask enforces "used valence ≤ valence_state" at every step. This removes all-invalid-action states, whose handling the plan otherwise needs (P5.3, P5.8), and removes completion-feasibility lookahead.
- **One normalized head replaces several.** A joint atom-type vocabulary is exactly normalized and replaces separate element/charge/H heads. Out-of-vocabulary atoms are out of domain.
- **Port bond types are reported as valence units.** Whether a 2-unit port is one double bond or two single bonds is not identifiable. State that in §1.

Factor order within a step is action kind → atom type and bond order → parent pointer. Pointer masks depend on the sampled bond order, so log p = Σ log masked_softmax(factor). Training uses the same order and masks. All three factors fit in one fused per-candidate sampling kernel.

### 2.4 Formula supervision and confidence (§3.2, P4.1–P4.3)

- Formula cross-entropy is computed over all enumerated in-domain candidates (§5.2 makes this cheap) plus an `OTHER` class. `OTHER` is the target when the true formula is outside the domain or beyond the candidate cap.
- "Captured probability of top-F" then means the in-domain mass of the top-F set. The truncation flag is exactly `count > F_enum_cap`.
- Formula probabilities exposed in outputs need their own calibration (currently missing from §3.6).
- Add a **formula-oracle mode**: an optional known precursor formula input. It is the MVP training and inference path and the upper-bound ablation. Some public datasets ship molecular formulas, but verify per dataset.
- Assignment loss uses the marginal likelihood over plausible assignments, −log Σ_{j∈plausible} p_j, with an explicit noise class.
- **Score comparability across hypotheses.** §3.5 makes K a total budget across formulas. Scores for candidates conditioned on different formulas are only comparable as log p(f) + log p(g \| f).
- **Train the decoder with the true formula only.** A wrong formula can make the target violate the composition masks. Exposure to wrong formulas at inference is handled by the reranker. Document this as a known train/inference gap.

### 2.5 Ranking and calibration (§3.6, §8, P6.3, P7.9)

Define one reranker: a small on-device MLP that scores each validated candidate from these features:

- normalized log p
- formula log p
- best integer mass residuals of ion hypotheses
- matched-peak intensity mass
- size
- port count

Train it on decoder samples from the frozen decoder on the validation split.

Name the calibration target: P(candidate ⊆ true parent). Matching is defined offline with RDKit substructure match. Atom tokens match exactly, ports may be matched by any parent bond, and H counts use the parent convention. Report ECE and reliability per size bin, because trivial fragments are almost always present.

Applying calibration (temperature or isotonic lookup) is a tiny device op. Fitting happens offline. Any change to model, search, peak retention or precision invalidates calibration (§7 step 8 should say so).

### 2.6 Identifiability caveats to document (§1, §8)

- Same-formula isomers and ring-vs-chain alternatives are often not separable from MS2 alone.
- Attachment positions are weakly identified.
- Report port accuracy separately and consider an "unknown port" ablation.

These caveats are reasons for the size-stratified and isomer hard-negative metrics already planned, not reasons to change architecture.

### 2.7 Optional: peak-anchored decoding (P0 decision)

An alternative is `START → SELECT_PEAK(ptr over N) → atoms…` with the composition budget taken from that peak's assigned fragment formula. It gives tighter masks, evidence by construction, and diversity across peaks, but it adds a dependency on the assignment head.

Recommendation: keep parent-formula conditioning for MVP. Store matched peak IDs in targets (§2.1) so peak anchoring is the first follow-up experiment without rebuilding data.

---

## 3. Internal contradictions and resolutions

| Topic | Contradiction | Resolution |
|---|---|---|
| Fixed-point bound | §4.2: quantization error "at least a/(2S)". It is an **upper** bound | Use per-element precomputed tables `mass_int[e][n] = round(n·m_e·S)`. Error is ≤ 0.5/S per element present, independent of atom count |
| Fixed-point boundary | The window is widened by the error bound (a superset, so no false rejects), *and* near-boundary cases return `mass_boundary_ambiguous` with a "higher-resolution device representation", which contradicts the portable no-FP64 stance | Define the integer rule as the spec: accept iff \|m_ion − \|z\|·mz\| ≤ tol_int + ε_q, with tol_int computed per peak on host (FP64/decimal, allowed by §4.1) and ε_q a documented slack. The oracle implements the same integer rule (bit-exact parity) plus a decimal proof that it is a superset of the real-valued rule. Delete the ambiguous status. Use a soft residual feature for scoring, since instrument error is far larger than quantization error |
| Formula confidence | A normalized top-F within a truncated search vs. a reported captured mass | §2.4: normalize over the full in-domain enumeration plus `OTHER` |
| Formula enumeration | Offline tables, bounded tiles, a running top-F *without materializing*, and "capacities must be measured", while exhaustion must also be visible | Two phases. (1) Exhaustive closed-form enumeration within domain caps, compacted to ≤ F_enum_cap (exhaustion = count overflow only). (2) Neural scoring and top-F. Do not fuse the MLP into enumeration |
| Beam/history | §3.5 offers "stochastic beam search *or* sampled continuations" (different memory). §6 mandates two cache banks but the graph state is not double-buffered. History "or parent pointers" | v1 uses independent sampling: one bank, no gather, history written per trajectory. With beam later: double-buffer *everything per candidate* (SSM state, trapezoidal factors, rotation angle, conv state if present, atom memory, adjacency, valence/composition counters, score, RNG lineage). History is backpointers with one device backtrack at the end |
| Graph target | "Parent subgraphs" vs. "observed fragments"; multiple traversals vs. likelihood ranking; ports vs. H | §2.1–2.3 |
| Calibration | Calibrated presence (non-exclusive) vs. likelihood ranking (sums to 1 over sequences) | §2.5 reranker |
| Default caps | T=128 vs. A=32. h=8, p=32 implies d_inner=256 (expand 1) while encoder d=256 with the usual expand 2 needs p=64 (doubles state). F fixed before recall@F is measured. r=32 slots at N=256 saves little. Decoder width, layers and s are unspecified separately from the encoder | T = 2 + A + R_max (e.g. A=32, R_max=8 gives T=42, round up to 48). State the invariant h·p = d_inner. Pick F from measured recall@F. Specify decoder (d, s) separately and treat s ∈ {16, 32} as a measured choice |
| Padding | Segmented valid reversal, pooling masks, dead candidates, unused atom slots, enumerated-buffer tails each handled ad hoc | One rule: masking by **select**, never multiply (NaN-safe). Pads are identity scan steps. Dead candidates use Δ=0 (frozen, no drift). Pointer softmax masks atom slots ≥ count. Candidate buffers carry an explicit count |
| Training gradients | "New fused ops need VJPs" (K11) vs. the inference kernels being mostly discrete | MVP needs **no new VJPs**: the training graph is composed (features, masked scan, attention, heads, select-masks). Fused forward kernels for training come later, each with a VJP or recompute |
| Loss normalization | Microbatch accumulation must equal full batch, but per-action vs. per-target normalization is unspecified | Define L_graph = Σ_actions / Σ_targets (or per-action), fixed globally, so accumulation is exact (P7.5 test) |
| Incomplete candidates | §3.5 marks budget-exhausted candidates, which can only occur because T is arbitrary | With derived T and masked R_max, termination is guaranteed. Assert no truncation |

---

## 4. Precision contract rewrite (§4.2, P1.3, P2.4, P2.5)

- **Representation.** u32 micro-dalton (S = 10⁶ per Da). The domain cap is \|z\|·m/z ≤ 4294.967 Da, which is well above typical small-molecule MS2.
- **Why not wider integers.** WGSL has no native 64-bit integers. This choice removes limb arithmetic and the wide-int capability check from v1.
- **Overflow is proven statically, not checked at runtime.** At domain-load time, the host proves Σ_e mass_int[e][cap_e] + max tol_int + ε_q < 2³². Comparisons use the subtraction-free form `m + t ≥ o && o + t ≥ m`.
- **Quantization error.** With per-element tables and ≤ 10 elements, the error is ≤ 5 µDa. That is about 0.05 ppm at 100 Da and 0.005 ppm at 1000 Da, against tolerances of several ppm. If a dataset needs absolute tolerance floors below roughly 50 µDa, revisit the scale (record in P0.4).
- **Host conversion.** Per-peak m/z → u32 and per-peak integer windows are computed once on host at upload (allowed by §4.1). The electron mass and adduct deltas are table entries.
- **v1 charge domain.** \|z\| = 1. Multiply charged ions are a domain extension.

---

## 5. On-device CubeCL plan changes

### 5.1 Kernel table: MVP vs. later (§5)

| ID | MVP form | Later |
|---|---|---|
| K01 | Fused pointwise features from u32 mass and f32 intensity. Select-based masks | — |
| K02 | One cube per spectrum, shared-memory bitonic sort of (mz_int, original_id), N_in ≤ 1024 bucket. No segmented reversal (§3 padding) | Larger buckets |
| K03 | Formula-oracle path in MVP. Then closed-form enumeration (§5.2) with deterministic compaction | Fused scorer |
| K04 | Reuse existing blocks/scan. Add Δ/x pad masking in the parameter epilogue | Fused epilogues |
| K05 | — (post-MVP). Build CSR + CSC so backward is a gather, with no float atomics | — |
| K06 | Replaced by cross-attention over all N encoder outputs. K/V projected **once per spectrum per layer** and shared across K and T | Slots if profiling demands |
| K07 | Recurrent decoder step (reuse the existing Mamba-3 step path if the RL rollout step generalizes; verify). Atom memory = stored hidden state | Incremental GNN as ablation |
| K08 | Per-candidate fused masked-softmax + Gumbel-max over factorized heads. No cross-candidate top-k per step | Two-stage top-k for beam |
| K09 | Not needed (independent trajectories) | Double-buffered all-state gather |
| K10 | Validator (independent code, shared domain tables), evidence (§5.3), reranker, per-spectrum sort and compaction | WL + bounded backtracking dedup |
| K11 | Composed autograd only | VJPs for fused ops |

### 5.2 Formula enumeration without tables

- **Grid.** One thread per (spectrum, heteroatom combination), decoded mixed-radix from `ABSOLUTE_POS`. The cube count comes from `calculate_cube_count_elemwise`, with a bounds check, because the grid can exceed 65 535 cubes on wgpu.
- **Inner loop.** Each thread loops over carbon count. Hydrogen count is solved in closed form: n_H = nearest to (target − heavy)/m_H, checked against the window at n_H ± 1 using the H table. Then domain rules (RDBE, valence parity) are applied.
- **Compaction.** Either per-cube count → exclusive scan → write (deterministic), or a u32 atomic counter followed by a deterministic sort of ≤ F_enum_cap records. u32 atomics are portable; f32 atomics are not assumed.
- **Fragment sub-formulas.** Same scheme per (peak, parent formula): loop over heavy sub-compositions of the parent and solve H.
- **Work estimate (op counts, not timings).** Example caps N≤10, O≤15, S≤3, P≤2, F≤6, Cl≤3, Br≤2, I≤1 give ≈3.5×10⁵ heteroatom combinations. Times a carbon range of tens, that is ~10⁷ cheap checks per spectrum before mass pruning. A C30N5O8S parent gives about 3.3×10³ heavy sub-compositions per peak; times N=256 and F=8, about 7×10⁶ checks per spectrum. Both are bounded, exhaustive within domain, and need no resident tables. Profile separately as §3.2 already asks.
- **Top-F over a large candidate set.** Up to F_enum_cap = 4096 records at 8 B is 32 KiB, which can hit threadgroup-memory limits. Use per-cube partial top-F then merge, with a capability check.

### 5.3 Evidence without decoder coupling

After STOP, each candidate computes F_sub, ports, and ion formulas for δ ∈ [−δ_max, δ_max]. It binary-searches the sorted peak mz_int with the host-supplied windows. Outputs are matched peak IDs, residuals, and matched intensity. The work is candidate-local and integer-only.

### 5.4 Termination and launches (§4.1, §7)

- With derived T, the fixed-step loop is short. Kernels read a device `all_done` flag and early-exit, so GPU work drops even though submissions remain.
- Budget metric: launches/call ≈ encoder + formula + T·L_step + validation. For scale, project notes record about 63 dispatches for one RL rollout step. A 4-layer decoder step with cross-attention, heads, masks and sampling is plausibly of that order. This is an estimate to measure, not a result.
- At fixed L_step, T=128 vs. T=48 is 2.7× more submissions independent of B.
- Under a host-bound regime, per-call latency is roughly flat in B·K until the GPU saturates. Throughput is improved by raising B·K per launch (bounded by §6 memory), not by kernel micro-tuning.
- Each host read is a near-fixed wait on Metal per project notes, so one final read per call is a floor. Batching more spectra per call amortizes it.
- An optional chunked termination check (one read every m steps) is a measured experiment that violates the zero-intermediate-read gate. Keep it out of the default.

### 5.5 CubeCL hygiene to put in §5

Several items are already in the document (per-axis builtins, scalar counts in `from_raw_parts`); keep them. Add:

- one-copy uploads via `client.create(Bytes::from_bytes_vec(..))`, not `create_from_slice`
- shared-memory and barrier kernels (bitonic, top-F) run on CPU for correctness only, with small CPU test sizes
- the plane-size fallback for any reduction

---

## 6. Memory and compute estimates

Assumptions (stated, not measured): B=8, K=32, N=256, encoder d=256 with d_inner=512, decoder Ld=4 with h=8 and p=64 (so h·p = 512), A=32, T=48, FP32.

| Buffer (inference) | Formula | Size |
|---|---|---|
| Decoder SSM state, one bank | B·K·Ld·h·p·s·4 B | s=64: **128 MiB**; s=32: 64 MiB; s=16: 32 MiB |
| Same with beam double bank | ×2 | 256 / 128 / 64 MiB |
| Trapezoidal + rotation carries, **factorized** (x_{t−1}, B_{t−1}, cumulative angle) | B·K·Ld·h·(p+s+s/2)·4 B | ≈ 5 MiB at s=64 (vs. 128 MiB if stored as an outer product; do not) |
| Cross-attention K/V cache (once per spectrum) | Ld·2·B·N·d·4 B | 16 MiB |
| Atom memory | B·K·A·d·4 B | 8 MiB |
| Adjacency (u8) + counters | B·K·A²·1 B | 256 KiB |
| Per-step transients | ≈ B·K·d_inner·10·4 B | ≈ 5 MiB |
| Encoder inference activations | few × B·N·d_inner·4 B | ≈ 4 MiB each |
| Formula candidates + assignments | B·F_cap·16 B + B·N·F·J·16 B (F=8, J=4) | < 2 MiB |
| Weights | ≈10M params (rough) | ≈ 40 MB |

Inference total is on the order of 150–250 MiB excluding pool rounding. Per project notes, pool reservation jumps after large allocations, so compare the estimator with logical bytes and treat reserved bytes as within documented rounding. The decoder state dominates, which makes s and beam-vs-sampling the main memory levers.

Training, rough, with B=16 spectra, m=8 targets per spectrum, T=48 and ~10 saved tensors per block:

- encoder: ≈ 1 GB (12 block-directions)
- decoder: ≈ 0.5 GB
- cross-attention probabilities: ≈ 0.2 GB
- optimizer: ~0.2 GB

That is **order 2 GB**, so checkpointing (P7.5) is conditional rather than mandatory.

**Key win.** Encode once per spectrum and pack the m targets into the query-length dimension of a standard batched cross-attention ([B, m·T, d] × [B, N, d]ᵀ). This avoids m× encoder recompute and needs no replication. It is the training analogue of the existing "no K-fold replication" rule.

Short-sequence scan-backward choice: per project notes, the chunked backward was slower than recurrent at small d_state on M1. Re-measure for N=256 and T=48 rather than assume.

---

## 7. Streamlined critical path and minimum trainable vertical slice

```text
S0 Domain v0 + grammar v1 + target recipe + data adapter  (P0.1–P0.8)
 ├─ S1 Offline target builder: canonical actions, per-step mask bitsets, formula/peak labels  (P1.8)
 └─ S2 CPU oracle: integer mass rule, enumeration, grammar replay/validator  (P1.1–P1.7, independent code)
S3 Encoder reuse + identity-pad masking + scan/step parity  (P3, minus K02 reversal)
S4 Decoder teacher-forced training, formula-oracle mode, offline masks; tiny overfit  (P7.1/7.2/7.4-lite/7.9-overfit)
S5 Device generation: recurrent step, mask kernel (parity vs. bitsets), counter-RNG sampling,
   validator, evidence, single final read, Rust + Python, CPU + GPU  (P5 subset, P6.1/6.3/6.4/6.5/6.6)
════ MVP GATE (§9) ════
S6 Formula enumeration + scorer + OTHER; switch to predicted top-F; recall@F → choose F  (P4.1–4.3)
S7 Reranker + calibration; WL + bounded-backtracking dedup  (P6.2, P7.9)
S8 Profile-driven: launches/step fusion, B·K amortization, s choice, K/V cache verification  (P8 O1/O2/O4/O6)
S9 Experiments: beam (K09), sparse relations (P4.4/4.5), slots (P4.6), peak anchoring,
   incremental GNN, MIMO (only with native kernel), reduced precision  (P8 O5/O7/O8/O9)
```

Dependency changes to the task table:

- P5 no longer depends on P4. It depends on P1, P3 and the S1 targets.
- P2 shrinks to counters, an estimator, and preallocated in-loop tensors; it is not a prerequisite workspace framework.
- P7 starts at S4, not after P6.

**MVP domain v0:**

- elements C, H, N, O, S, P, F, Cl, Br, I with a small charged-atom set
- single, double and triple bonds, Kekulé form
- adducts [M+H]⁺, [M+Na]⁺, [M+NH₄]⁺, [M−H]⁻, with \|z\| = 1
- A=32, R_max=8, T=48, A_min=3
- formula oracle
- K=32 independent samples, no dedup (status `dedup=not_run`), no beam

---

## 8. Exact document changes

### Design

- **§1**: Add "Candidate semantics v1". Specify the parent-convention H, derived ports in valence units, Kekulé bond orders, no on-device aromaticity, that port bond type is not identifiable, and that output may contain fewer than R candidates.
- **§2**: Add the integer sidecar (u32 µDa per peak, per-peak integer windows computed on host) and the domain mass and charge caps.
- **§3.1**: Replace "reverse only valid positions" with identity-transition padding (select-masked Δ and x, then a full flip). State that chunked-scan results under different pad offsets match within tolerance, not bitwise.
- **§3.2**: Replace tables and meet-in-the-middle with §5.2. Normalize over all enumerated formulas plus `OTHER`. Add formula-oracle mode, formula calibration, and score = log p(f) + log p(g \| f).
- **§3.3**: Mark as post-MVP. Require CSR + CSC and no float atomics on the portable path.
- **§3.4**: v1 uses full-memory cross-attention with per-spectrum K/V cache. Slots move to an experiment. Remove the tiled online softmax.
- **§3.5**: Grammar v1 (§2.3), canonical DFS, derived T, `u = last atom`, atom memory = hidden state, independent sampling in v1, fixed factor order, identical training and inference masks, no incomplete candidates.
- **§3.6**: Share domain tables between masks and validator but keep the code independent. Add the reranker, the named calibration target with size bins, and the Kekulé-variant policy.
- **§4.1**: Add launches/step and launches/call budgets and the device early-exit flag.
- **§4.2**: Rewrite per §4. Fix the bound direction and delete `mass_boundary_ambiguous`.
- **§5**: Mark MVP vs. later per §5.1. Add the hygiene items from §5.5.
- **§6**: Single bank in v1. Factorized carries. State h·p = d_inner. Add the cross-attention K/V and atom-memory rows. Defer typed arenas and leases.
- **§7**: Add a NaN/sentinel poisoning requirement for kernel and autotune validation, B·K amortization as the first throughput lever, and calibration refit after algorithmic changes.
- **§8**: Add the target recipe §2.1. Define or remove `L_local_evidence`. Move contrastive training post-MVP. Train with the true formula. Add a reranker training stage. Specify loss normalization. Revise caps (T derived, F from recall, decoder s explicit).

### Tasks

- **New P0.7**: target recipe q(g \| spectrum) and its version.
- **New P0.8**: grammar v1, atom-type vocabulary, canonical ordering.
- **P0.3**: replace "attachment ports" with derived ports.
- **P1.3**: rewrite as the integer-spec rule plus decimal superset proof.
- **P1.6**: closed-form enumeration oracle with brute-force cross-check.
- **New P1.8**: offline target builder emitting canonical actions, per-step legality bitsets, peak IDs, and formula and assignment labels.
- **P2.3**: reduce to preallocated tensors reused in-loop.
- **P2.4**: drop wide-int.
- **P2.5**: defer.
- **P2.8**: drop concurrent leases from v1; add poisoning tests.
- **P3.1**: remove valid-length reversal.
- **P3.5**: extend to decoder teacher-forced vs. stepped parity.
- **P4.4–P4.7**: move to an "Experiments" phase.
- **P5.3**: add mask parity against offline bitsets.
- **P5.4**: v1 sampling; exhaustive-enumeration comparison on tiny graphs.
- **P5.5**: add the formula log-prior to scores.
- **P5.6**: move to experiments, specifying all-state double buffering.
- **P5.8**: replace exhaustion handling with a "termination guaranteed by grammar bound" test.
- **P6.2**: after MVP.
- **P6.6**: add contiguity enforcement, GIL release, and non-contiguous parity inputs.
- **P7.2**: remove contrastive loss from MVP; define loss normalization.
- **P7.5**: checkpointing conditional on the estimator.
- **P7.9**: split into an overfit fixture (MVP) and reranker calibration (S7).
- **O7**: gate on a native MIMO kernel.
- **O9**: rename to "beam search and compaction".

---

## 9. Ideas to reject or defer

**Reject for v1:**

- offline meet-in-the-middle formula tables
- two-u32 limb mass arithmetic
- `mass_boundary_ambiguous` with a higher-resolution retry
- T=128
- `MARK_ATTACHMENT` as a separate action
- on-device aromaticity perception
- typed arenas, sub-buffer offsets and concurrent leases
- tiled online softmax for N×r ≤ 512×32
- float atomics on the portable path
- full-model persistent kernel (already rejected; agree)

**Defer until measured or justified:**

- beam search and its ancestry gather
- slots
- sparse relations
- incremental GNN
- contrastive structure encoder
- MIMO rank 4 (an R² SISO decomposition implies roughly R²× SISO state and compute at rank R; verify against the code before any experiment)
- device beam compaction or indirect dispatch
- persisted compile caches (O10)
- reduced-precision training

---

## 10. Most important acceptance gates

1. **Integer mass spec.** The GPU accept/reject decision is bit-identical to the CPU oracle on random and boundary fixtures. A decimal check proves the integer rule is a superset of the real-valued tolerance rule.
2. **Enumeration completeness.** The GPU formula set equals the brute-force CPU set exactly within domain caps. The truncation flag is set iff count > cap.
3. **Target legality.** 100% of training targets replay legally under grammar v1. Domain coverage and dropped-target rates are reported per dataset split.
4. **Mask parity.** The device mask kernel equals the offline bitsets for every step of every validation target, as exact bitset equality.
5. **Train/inference consistency.** Teacher-forced decoder logits match stepped recurrent logits within a documented tolerance, covering SSM state, trapezoidal and rotation carries, and conv state if present.
6. **Search correctness.** On a tiny vocabulary with A ≤ 3, log-probs of all complete sequences match a CPU enumeration and sum to 1. Seeded sampling frequencies pass a fixed-seed goodness-of-fit test. The candidate set for a spectrum is invariant to batch position and composition on the same backend.
7. **Padding.** Valid outputs are unchanged within tolerance across shape buckets, and with pads poisoned by large values and NaN (select-based masking).
8. **Residency.** Between upload and final output, the read counter shows exactly one batched read per call. Launches per decode step and per call are reported. There are no allocation calls inside the warmed decoder loop. Reserved bytes are stable over repeated fixed-shape and alternating-bucket calls.
9. **Dropped-launch detection.** Every new kernel and autotune winner overwrites a NaN/sentinel-poisoned output on CPU and on a real GPU.
10. **Learnability.** The overfit fixture reproduces its target subgraphs within top-K. The MVP trains end-to-end on a small real split with no accuracy claim.
11. **Parity.** Rust and Python produce bit-identical candidates for the same seed and backend, with identical error and status codes, including for non-contiguous inputs. CPU and real-GPU suites both run; a skipped GPU run counts as not passed.
12. **Quality release gate (post-MVP).** On held-out molecule- and scaffold-disjoint splits, report the following against baselines:
    - formula recall@F
    - size-stratified containment precision and coverage at K
    - reliability of the named calibration target
    - duplicate and abstention rates

    Thresholds are set only after baselines exist.
