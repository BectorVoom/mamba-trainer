# Codex design review: MS2 V1 architecture, revision 1 (sections 1 and 2)

Reviewer: codex exec (read-only), 2026-10-03. Verdict: not ready. Revision 2 of the specification answers these findings section by section.

The specification is **not ready for implementation**. The hydrogen bookkeeping itself is consistent with contracts §4.3 for the two V0 adducts; the blockers concern search bounds, support construction, arithmetic, and unstated interface changes.

1. **§1.4 — blocker: the visit limit is checked after the work has already happened.**  
   `ms2_enum_count` enumerates every lane, then `ms2_enum_offsets` discovers that visits exceeded `formula_rows_visited_max`. A request with a limit of 100 can therefore perform millions of checks. Likewise, discovering excess lanes after launching them does not bound work. Tile fills repeat enumeration without a stated aggregate limit. This contradicts contracts §9’s stop-before-exceeding rule and P4.1/P4.9.

   **Proposed wording:** “Before dispatch, bound submitted lanes and assign deterministic work budgets. Enumeration stops before the next operation would exceed its budget. Count and fill use identical cutoffs. Specify separate logical-candidate and physical-reenumeration counters, and bound total work across all passes. Any cutoff sets `formula_search_exhausted` and clears support completeness.”

2. **§1.4 — blocker: ‘all arithmetic is u32’ lacks the required proof and signed representation.**  
   The existing host enumerator uses `u64` heavy masses and signed arithmetic; [`dbe_twice`](src/models/ms2/formula_enum.rs:173) includes negative halogen terms. Computing `v_max − 2` in unsigned arithmetic underflows for F/Cl/Br/I. Mass pruning must happen **before** overflowing multiplication or addition: an artifact allowing 400 carbons produces `4,800,000,000` µDa. Ratio coefficients are unbounded; `65,535 × 100,000` exceeds u32. Prefix sums, lane products, radix products, and counters also need bounds.

   **Proposed wording:** “Validate artifacts before upload with checked wide host arithmetic. Prove bounds for every device product, sum, index, offset and counter. Represent DBE as separate positive and negative totals and compare before subtraction. Guard mass multiplication using division-based bounds. Reject artifacts requiring unsupported arithmetic; never wrap. Define counter overflow behavior and reserve `u32::MAX` exclusively for sentinels where applicable.”

3. **§1.4 and §2.1 — blocker: the closed-form hydrogen interval is incomplete as a device algorithm.**  
   The mathematical interval is correct, but unsigned “ordered subtraction” alone does not define it. If `lo < m <= hi`, the lower hydrogen bound is zero, rather than an unsigned negative quotient. If `m > hi`, the interval is empty. Domain hydrogen bounds must also intersect it. The existing [`check_hydrogen`](src/models/ms2/formula_enum.rs:588) explicitly handles these cases using signed wide arithmetic.

   For ions, the interval must be constructed around neutral atomic mass: `mass(u) = mz + z_a*m_e`. Searching around observed m/z without this transformation shifts boundaries by 549 µDa. The superset must include electron residual error and a conservative hydrogen-dependent arithmetic bound.

   **Proposed wording:** “Define `[lo,hi]` explicitly, including observation, adduct/electron and domain-wide composition error bounds. Return empty when heavy mass exceeds `hi`; otherwise use `h_lo = max(hydrogen_min, ceil(max(lo−m,0)/m_H))` and `h_hi = min(hydrogen_max, floor((hi−m)/m_H))`. Use overflow-safe quotient/remainder ceiling division. Ion endpoints use checked signed electron correction; every candidate then receives its own exact verdict.”

4. **§1.2–§2.4 — blocker: the required schema migration is unstated.**  
   Contracts §3 and existing code use schema version 1. `GenerationConfig` has no `formula_source`, `formula_tiles`, `enum_lanes_max`, `ion_work_max`, or assignment capacity `J`. `CandidateBatch::formula_row` means a resident-table row; an enumeration rank cannot silently replace it. There are no output fields for formula compositions, ion distributions, assignment completeness, evidence peak IDs or residuals. `ModelConfig` requires a formula-table artifact and does not define enumeration artifacts or assignment-head parameters.

   **Example:** an enumerated formula with source ID 17 can be serialized as `formula_row=17`, misleading a V0 consumer into retrieving an unrelated table composition.

   **Proposed wording:** “Define a versioned schema migration before implementation. Add validated configuration fields and defaults; composition-bearing formula records with explicit source kind and source ID; assignment/evidence records and statuses; and hashed/versioned `EnumDomain`, `RatioBounds`, rare-table and assignment-head checkpoint metadata. Specify V0 loading, unsupported-version rejection and oracle behavior.”

5. **§1.2–§1.3 — blocker: the unchanged table kernel cannot supply later tiles.**  
   [`formula_window`](src/tensor/ops/ms2.rs:3379) accepts no tile offset or continuation state and stores only the first `min(M, rows_scored_max)` joined rows. Repeating it for tile 1 repeats tile 0. With 50 joined rows and `M=32`, the remaining 18 rows cannot be scored by the specified “as before” path. The existing head also returns normalized tile log-probabilities, whereas streaming needs raw scores.

   **Proposed wording:** “Specify a tiled table-search interface that produces each joined support rank exactly once in `(mass, composition)` order. Define range discovery, continuation state, counters and visit accounting across tiles. Add a raw-score formula-head interface; per-tile normalized log-probabilities must not enter the global partition.”

6. **§1.3–§2.1 — blocker: the six-array binding limit is not demonstrated.**  
   Straightforward interfaces using the separately described buffers exceed it. Counts below include inputs and outputs, excluding scalar arguments and CubeCL metadata.

   | Kernel/pass | Arrays needed with separate buffers | Count |
   |---|---|---:|
   | Existing formula window | table, request metadata, window, counters | 4 |
   | Formula gather | window, table masses, table counts, candidate counts, candidate metadata | 5 |
   | Enumeration count | request metadata, rare table, domain, ratio bounds, chemistry constants, lane statistics | 6 |
   | Enumeration offsets | lane statistics, offsets, counters | 3 |
   | Enumeration fill | request metadata, rare table, domain, ratio bounds, chemistry constants, offsets/counts, candidate counts, candidate metadata | 8 |
   | Streaming partition fold | scores, candidate validity/metadata, log-partition, support flag | 4 |
   | Running top-F merge | tile scores, tile metadata, tile counts, running IDs/ranks, running scores, running counts, top count | 7 |
   | Final probability/retained-mass pass | running scores, running IDs/validity, log-partition, support flag, counters, output log-probabilities, retained mass | 7 |
   | Ion enumeration | parent counts, parent-slot validity, kept peaks, request metadata/uncertainty, chemistry constants, ion counts, ion metadata | 7 |

   These are avoidable through packing or separate launches, but the implementer must currently invent that layout.

   **Proposed wording:** “Publish every kernel signature and packed-buffer layout. Combine domain/ratio artifacts, pass frozen chemistry constants as scalars where suitable, pack top validity/count into integer state, and split finalization where necessary. Every forward and backward kernel must have at most six arrays.”

7. **§2.1 — major: the ion equations are correct, but the claimed enumeration equivalence is false.**  
   Substituting `h = H(g)+h_a+s` into `mass(u)−z_a*m_e` reproduces contracts §4.3 for `|z|=1`. There is no double counting of hydrogen.

   However, the parent-wide hydrogen interval is only a composition superset. For a `C3H8O` parent under `[M+H]+`, it allows the full heavy composition with 11 hydrogens. The whole-parent graph has `c(g)=0`, so its supported mapping permits only 9. Composition alone cannot establish connectivity, parent-relative hydrogen counts or boundary bonds. Moreover, charge and adduct are **fixed by the request**, not enumerated alternatives. Sodium or another charge cannot be represented by varying hydrogen.

   **Proposed wording:** “This phase supports only the two V0 adducts and singly charged ions. Enumerate a relaxed composition superset under the fixed request adduct; do not claim every composition corresponds to a supported graph-to-ion mapping. Broader adduct/charge alternatives require explicit hypothesis IDs, composition deltas, charge-scaled mass comparisons and separate domain rules.”

8. **§2.4 — major: open-valence units are substituted for boundary-bond count.**  
   Contracts §4.5 explicitly distinguishes them. A fragment cut across one double bond has two open-valence units but `c(g)=1`. Section 2.4 permits `s=±2`, whereas the contract permits only `±1`. The current labels use actual embedding boundary counts in [`Candidates::label`](src/models/ms2/targets.rs:431).

   **Proposed wording:** “Use known boundary-bond count when available. When attachment partition is unknown, report possible mapping separately from certified anchoring; define whether certification requires validity for every compatible partition. Open-valence sum must not be substituted for boundary-bond count.”

9. **§1.3–§1.4 — major: deterministic support order is not fully defined.**  
   Lexicographic ordering by “rare combination index” depends on how the resident rare table is built. Its ordering, uniqueness and indexing are unspecified. Lane order must be independent of dispatch order. The existing host enumerator traverses heavy elements in a different order and mass-sorts its retained buffer; it is not presently the twin of this proposed device order.

   **Example:** reversing two rare-table rows changes the capacity-limited prefix and resolves equal neural scores differently.

   **Proposed wording:** “Serialize rare combinations once in a specified lexicographic element order, without duplicates. Define lane index `carbon*P+rare_index`, local `(N,O,H)` order, and joined rank as lane offset plus local joined index. Count and fill must share identical predicates and cutoffs. Add a host reference for this exact order, including capacity and work exhaustion.”

10. **§1.3 — major: running top-F state, padding and numerical determinism are underspecified.**  
    The tie rule is sensible, but `top` retains an undefined V0-style layout, and no validity/count state is specified. A tile slot becomes meaningless after its buffer is reused. Also, padding log-probabilities are zero in V0: summing all `F` entries would give retained mass 4 for one real candidate with `F=4`.

    “One tile reproduces V0 bit for bit” is too broad: GPU `ln(1+counts)` can differ from the host-uploaded features, and streaming normalization can round differently from the existing log-softmax.

    **Proposed wording:** “Define persistent top state as composition, source ID, support rank, raw score and validity/count. Compare `(descending finite score, ascending support rank)`. Finalize only valid entries; padding is never included in retained mass. Define nonfinite-score handling and reduction order. Require exact integer/search parity and numerical tolerances for recomputed floating outputs; retain the old numerical path if bitwise parity is required.”

11. **§1.2–§1.4 and §2.1 — major: counter and status semantics conflict or remain missing.**  
    Actual V0 counters are [`u32 [B,5]`](src/tensor/ops/ms2.rs:1501), despite V0 architecture §3.3’s stale `[B,4]` wording. The unchanged wrapper rejects `[B,8]`; an adapter or new interface is necessary.

    Section 1.2 also calls missing **gold** `formula_absent`, while contracts §8–§9 reserve that fatal request bit for a completed search joining nothing. Gold may be absent while other formulas are present. `complete` needs search-completion semantics beyond `joined==scored`. New ion status bits have no namespace, propagation rule or completeness field. Unknown precision currently produces indistinguishable all-zero ion rows.

    **Proposed wording:** “Keep training metric `gold_not_scored` separate from fatal request `formula_absent`. Define counter-layout migration explicitly. Support is complete only when all search work finished and every joined candidate was scored. Specify exhausted-empty, unavailable, oracle and failed-request behavior. Assign ion status bits and expose unavailable, incomplete and complete-empty assignment states.”

12. **§2.1 — major: ion work is not bounded at request level.**  
    Heavy sub-composition cardinality is  
    `∏heavy e (c_f[e]+1)`, before excluding the empty vector or applying other constraints. The specification gives neither this bound nor a radix convention. “Visits” could mean heavy vectors, hydrogen checks or accepted hypotheses.

    There are `B×F×N` lanes. With `B=8`, `F=8`, `N=256` and `ion_work_max=100,000` **per lane**, the bound is 1.6384 billion visits, before accounting for multiple hydrogen checks.

    **Proposed wording:** “Define radix order, exclusion of empty heavy composition, visit unit, safe iteration without overflowing the full radix product, and per-lane plus aggregate request limits. Bound hydrogen verdicts as well as heavy-vector visits. Continue after filling `J` only within the work budget; otherwise accepted totals are lower bounds and completeness is false.”

13. **§1.4 and §2 — major: the workspace and activation estimates omit dominant terms.**  
    Enumeration lanes total `B×(C_cap+1)×P`; both lane statistics and offsets require storage, and the serial offsets kernel performs that much reduction work. The rare table alone costs `24P` bytes.

    Assignment counts cost `40BFNJ` bytes. Materialized embeddings cost `4BFNJd` bytes in FP32: at `B=8,F=8,N=256,J=16,d=128`, embeddings alone require **128 MiB**, plus 10 MiB of counts. Autograd retains additional projection/SiLU intermediates. `Tt×B×M×d` is an element count and does not cover all formula-training activations.

    **Proposed wording:** “Extend the checked memory estimator with every resident artifact, lane-statistics/offset buffer, tile buffer, retained formula record, assignment count/meta/logit/probability tensor, neural activation and backward buffer. Define ownership and reuse by full shape/configuration bucket. Refuse the request before allocation when the estimate exceeds its budget.”

14. **§1.3 — major: streaming differentiation is asserted without a safe execution contract.**  
    Mutable device `log_z` state suitable for inference does not automatically remain connected to autograd. Reusing tile storage can overwrite activations required by earlier tiles. An empty tile also has an undefined log-sum-exp unless separately guarded; `has_support` for the accumulated partition does not by itself solve that.

    **Example:** gold lies in tile 0 and negatives in tile 1; detaching the fold loses tile 1’s contribution to the formula gradient.

    **Proposed wording:** “Search and top-F selection are nondifferentiable. Raw-score computation and partition folding remain differentiable through all valid tiles, using functional autograd operations or a specified VJP/recomputation scheme. Empty tiles leave partition state unchanged. Verify gradients against concatenated-support cross-entropy, including empty tiles, later-tile gold and extreme scores.”

15. **§1.2–§1.3 — major: missing-gold loss handling lacks exact support matching and normalization.**  
    The existing head identifies gold by a table/window slot and normalizes over spectra whose gold is scored. Enumeration needs exact ten-count composition equality and a global support rank. The unconditional §1.3 loss expression cannot be evaluated when gold is absent.

    **Example:** gold passes chemical filters but lies just beyond `Tt*M`; computing its independent embedding score against the truncated partition would silently change the specified objective.

    **Proposed wording:** “Match gold by exact composition within the scored prefix. Compute formula loss only when that match exists, using the same global partition. Distinguish domain exclusion, pruning exclusion, work exhaustion and score-cap exclusion. Preserve normalization over eligible spectra and specify accumulation using total numerator and eligible count. Independent gold embedding for graph conditioning does not add gold to formula-loss support.”

16. **§1.2 and §2.3 — major: oracle-conditioned training needs an explicit inference boundary.**  
    Using the true composition embedding is valid conditional training and is an intentional change from [`train.rs`](src/models/ms2/train.rs:641). It does not demonstrate performance with predicted formulas. Training assignment under gold `F=1` likewise produces an oracle-conditioned metric.

    **Example:** gold is outside every scored tile; graph and assignment losses may improve while ordinary generation can never select that parent formula.

    **Proposed wording:** “Gold compositions and anchors are training/evaluation-only inputs. Ordinary generation conditions exclusively on selected `top_counts`; assignment and candidate evidence use the trajectory’s selected formula slot. Gold replacement requires explicit oracle mode and marked outputs. Report oracle-conditioned and predicted-formula metrics separately, including full-dataset search misses. Add an inference test demonstrating independence from gold/anchor payloads.”

17. **§2.3 — major: label-set construction and dropped-label normalization are ambiguous.**  
    A peak’s label set must use only anchors for that specific original peak ID and must deduplicate compositions. Two different target graphs can yield the same ion formula; summing it twice can exceed probability one.

    Partial overlap is also unspecified: if labels are `{u1,u2}` and only `u1` is retained, is the peak trained or dropped? “Mean over anchored peaks” conflicts with “contributes nothing” unless the denominator is defined.

    **Proposed wording:** “Define `L_bp = unique{ion_composition(g,s) | g retained, (peak_id_p,s) in anchors(g)}` after mapping original IDs to kept peaks. Define `A_bp = L_bp ∩ valid_retained_hypotheses`. State the partial-overlap policy explicitly. Skip empty intersections before logarithms; divide by the specified eligible-peak count with a zero-count guard. Define microbatch accumulation from numerator/count and report fully and partially dropped labels separately.”

18. **§1.4–§1.5 — major: fitted pruning artifacts and stage-recall gates are not implementable yet.**  
    `RatioBounds` has no schema or fitting rule. Carbon/size buckets, inclusive boundaries, rational coefficients, carbon-zero behavior, rare totals/distinct counts and fitted DBE intervals are unspecified. The eight-word counter layout provides one aggregate exact-filter reject count, despite requiring separate counts for hydrogen ceiling, parity and DBE, and omits an explicit mass-reject counter.

    **Example:** a carbon-free composition either bypasses a carbon ratio, fails it, or triggers division-by-zero depending on the implementer’s interpretation. Rare combinations removed during host table construction cannot be audited from device reject counts alone.

    **Proposed wording:** “Freeze artifact schemas, fitting procedure, train provenance, bucket boundaries, zero-denominator rules and filter order. Define `h_max` and twice-DBE explicitly; these are necessary filters, not feasibility certificates. Provide stage-specific reference runs/counters including prefiltered rare combinations, and compare unlimited tiny-domain enumeration against brute force before measuring bounded-prefix recall.”

19. **§2.2–§2.4 — major: probability and evidence outputs lack support semantics.**  
    The distribution is normalized over retained `J` hypotheses plus unassigned, even when additional accepted hypotheses were dropped. A retained probability of 0.9 is therefore conditional on truncated support. Unassigned has no direct supervision under the stated recipe; it represents probability outside labeled assignments, rather than demonstrated experimental noise.

    Evidence serialization is also missing: signed versus absolute residual, units, multiple matching hypotheses, peak-ID mapping, and selected-formula association.

    **Proposed wording:** “Expose assignment support completeness and truncation with every distribution/evidence record. Name probabilities as conditional on retained support when incomplete. Describe unassigned as a modeled unmatched class under partial pseudo-label supervision. Define evidence records with original peak ID, formula/hypothesis identity, signed integer residual and aggregation policy.”

20. **§1.1, §1.5 and launch accounting — minor: measurement and acceptance references are not reproducible in this workspace.**  
    No `bench/results/ms2/formula_sources_*.json` files are present, so the reported recall/occupancy numbers cannot be checked here. Section 1.5 remains a placeholder, leaving production capacities/defaults undecided. “Three launches plus tile fills” also conflicts with the listed count → offsets → fill sequence, which implies two setup launches plus one fill per tile.

    **Proposed wording:** “Name exact report files, hashes, commands, artifact versions and recall denominators. Keep the production-source decision gated on measured recall, bytes, work and latency. State the exact launch formula and require CPU/GPU host-twin parity, poisoned-output coverage, boundary arithmetic, capacity-prefix determinism, tiled gradient parity and warmed one-read verification.”

No files were modified and no cargo commands were run.

**Verdict: not ready.**

---

# Second review (revision 2)

Reviewer: codex exec (read-only), 2026-10-03. Verdict: not ready; revision 3 answers the new problems listed here.

Revision 2 fixes much of revision 1, but still leaves implementation-critical ambiguities. Review performed read-only through CodeGraph and the documents; no files modified and no cargo commands run.

The following numbers correspond to the [20 previous findings](docs/reviews/MS2_V1_ARCHITECTURE_CODEX_REVIEW.md).

1. **Partly resolved.** Pre-dispatch lane limits and per-lane cutoffs now exist, but logical visits versus count/fill re-enumeration work remain undefined.
2. **Partly resolved.** Guarded mass products and separate DBE totals fix the principal unsigned hazards, but window-width arithmetic, saturation semantics and packed signed DBE intervals still lack complete rules.
3. **Partly resolved.** Hydrogen endpoints and electron correction are correct, but overflow-safe checks for the four/three-hydrogen limits and construction of `E_ion` remain unspecified.
4. **Partly resolved.** Formula provenance is corrected, but legacy output migration, configuration fields, assignment serialization and checkpoint loading remain incomplete.
5. **Resolved.** Removing tiles removes the continuation and cross-tile normalization problems.
6. **Partly resolved.** Most signatures fit six bindings, but label masking needs an additional original-peak-ID buffer unless it is explicitly packed.
7. **Resolved.** Assignment now explicitly enumerates a composition superset under the request’s fixed V0 adduct and charge.
8. **Partly resolved.** Boundary-bond bounds replace the incorrect equality with open valence, but “possibly anchored” and universal anchoring lack parent-compatible partition existence checks.
9. **Partly resolved.** Lane and local order are explicit, but the stated `enumerate_device_order` reference is absent from the current code.
10. **Partly resolved.** Persistent window slots and retained-count handling fix padding, but unconditional V0 parity conflicts with the new nonfinite-score selection rule.
11. **Partly resolved.** Five-word counters and gold-miss terminology are corrected, but enumeration failure propagation and externally serialized ion statuses remain incomplete.
12. **Partly resolved.** Aggregate heavy-vector work is bounded, but radix-product cutoff has an off-by-one ambiguity and hydrogen-work validation is incomplete.
13. **Partly resolved.** Dominant lane and embedding allocations are recognized, but assignment features, logits, probabilities, labels and backward intermediates are not comprehensively included.
14. **Resolved.** A single ordinary masked log-softmax removes mutable streaming-autograd and reused-tile hazards.
15. **Resolved.** Exact composition matching, eligible-spectrum normalization and microbatch numerator/count accumulation are specified.
16. **Resolved.** Gold-conditioned training and predicted-formula generation now have an explicit boundary and separately named metrics.
17. **Partly resolved.** Deduplication and partial-overlap policy are clear, but label-cap ordering and safe exclusion of empty intersections before logarithms are unspecified.
18. **Partly resolved.** Existing versioned `RatioBounds` supplies fitting semantics, but device packing, unseen-bucket handling and equivalence of the new enumeration order remain unverified.
19. **Partly resolved.** Conditional probabilities and pseudo-label evidence are described, but output schema, hypothesis identity and evidence aggregation remain incomplete.
20. **Partly resolved.** The three reports now exist and defaults are explicit, but measurements describe the existing host DFS rather than the proposed lane-budget implementation, and launch accounting omits feature generation.

New problems and counterexamples in [revision 2](docs/MS2_V1_ARCHITECTURE.md):

- **Schema-v1 loading cannot use the stated missing-field defaults.** A valid v1 `CandidateBatch` containing `formula_row=17` has neither counts nor window rank; its serialized output cannot reconstruct either without the original table/search context, and the table gives those fields default “—”. Define legacy absence explicitly or require reconstruction inputs. Likewise, a v1 checkpoint lacks assignment parameters: loading them must specify disabled assignment or an explicit initialization/migration policy.

- **“V1 default” changes legacy training behavior.** A legacy training configuration missing `gold_formula_conditioning` would receive `Composition`, whereas V0 used `ScoredRowOrZero`; when gold is outside the window, decoder conditioning changes from zero to a learned composition embedding. Version-dependent defaults are required.

- **The hydrogen-limit calculation can overflow even though its mathematical bound is correct.** With `half=2^31`, unsigned `2*half` wraps to zero, producing a bound of one instead of 4,262; `tol + bound + E_domain` can overflow earlier too. Validate with checked wide arithmetic or compare against thresholds without multiplication. The limits of four and three are explicit scope restrictions, not consequences of the chemistry; a wide window may contain many hydrogen counts up to the fitted maximum of 96.

- **Equal lane budgets can discard the entire useful budget.** With `P=7,993` from the scale-fit caps/ranges and `formula_rows_visited_max=1,000`, every lane receives zero visits, so the search returns no formulas despite having a positive request budget. With `P=2`, budget three and lane workloads three/zero, only one useful visit is permitted. Define this deliberate underutilization, or distribute the remainder and unused work deterministically; report recall for that exact policy.

- **Saturation loses counter meaning.** For example, 8,000 lanes each reporting 1,000,000 visits yield 8 billion visits but serialize `4,294,967,294`; “exhausted” does not tell consumers that this counter is a lower bound. Define saturated counters as lower bounds, and explicitly use cap-clamped offsets for filling or prove saturated offsets cannot affect scored ranks.

- **The fill/pad single-writer claim needs an inner cutoff.** With `M=128`, scored cap 32 and lane zero joining 40 candidates, the lane passes the stated first-rank guard; if it writes all 40, slots 32–39 are also written by `ms2_cand_pad`. Specify that fill writes only ranks `< rows_scored`, and stops before any later slot; the lane-entry guard alone is insufficient.

- **The radix-product guard uses the wrong comparison threshold for completeness.** For parent heavy counts `C=1,O=1` and `ion_work_max=3`, the radix product is four but there are exactly three nonempty vectors, so support is complete. A product guard that only records “exceeds three” cannot distinguish this from a genuinely truncated product of six. Determine whether the product exceeds `ion_work_max+1`, using overflow-safe multiplication and explicit terminal-state handling.

- **Label masking has seven bindings with the existing peak representation.** The six listed arrays omit `peak_id`, needed to map the kept raw index to the original ID. For kept raw slot five with original ID 900, comparing label ID 900 to raw index five drops a correct label. Pack original IDs into `kept`, or add a separate gather pass.

- **Boundary-count inequalities do not establish possible anchoring.** In acetylene `C2H2`, fragment `CH` has residual valence three, giving `c_min=1,c_max=3`; the only remaining heavy atom permits one boundary bond, a triple bond, so `c(g)=1` and shift two is impossible although §2.4 calls it possible. Worse, a generated single-bond `CH2–CH2` under parent `C2H4` has two residual units but no remaining parent atoms, so no compatible boundary partition exists at all; universal anchoring requires an existence condition.

- **The 64-label cap can alter supervision without a deterministic rule.** With 65 distinct labels and only label 65 matching a retained hypothesis, retaining the first 64 drops the peak while another ordering trains it. Freeze ordering, deduplication-before-cap and overflow/partial metrics. Also skip state-zero/two peaks before evaluating `log(0)`; multiplying their infinite losses by zero is unsafe.

- **The cap validations narrow previously accepted inputs without a migration rule.** `C256H514` has integer mass `3,589,???,???` µDa, below `u32::MAX`, yet its carbon cap exceeds 255 and its heavy bucket requires row 64 beyond a 64-row table. Separately, the existing table loader accepts a mass-valid `H1024` row, which the new log table rejects. These may be intentional device restrictions, but “v1 still means exactly what it meant” cannot remain unconditional.

- **Launch accounting is incomplete.** Enumeration requires count, offsets, fill, pad **and** `ms2_count_features` before the head: five launches, not four. Teacher forcing and assignment additionally need feature interfaces for `[B,10]` gold counts and 12-word ion records; the declared 13-word `cand` interface cannot consume those unchanged.

Several suggested concerns are **not defects as written**:

- **DBE signs are correct:** coefficients are C `+2`, H `−1`, N `+1`, O `0`, F `−1`, P `+3`, S `+4`, Cl/Br/I `−1`; these match `dbe_twice` and the maximum-valence ceiling.
- **Electron signs are correct:** for atomic mass 100,000,000, positive m/z is 99,999,451 and adds 549; negative m/z is 100,000,549 and subtracts 549.
- **Excluding radix index zero is correct:** hydrogen-only ions have no supported nonempty heavy graph in this domain.
- **The log-table feature claim is correct within its validated range:** V0 upload already computes `(1.0 + f32::from(count)).ln()`, identical to the proposed expression for counts ≤1023; the broader parity claim fails for nonfinite scores because V0 top selection can retain them and revision 2 excludes them.
- **H94 and perfluoro counts do not violate these caps:** the actual scale fit has `hydrogen_max=96` and `F_cap=34`; `C16F34` fits, and even `C73F148` fits the element and integer-mass limits. These examples do not justify lowering either cap.

Before implementation:

- **B1 — §§1.2–1.3 table source:** complete version-dependent schema/checkpoint migration; qualify parity; specify gold feature input, empty-support handling and full allocation/launch accounting.
- **B2 — §1.4 enumeration:** freeze checked arithmetic, packed bounds, budget/saturation semantics and fill ownership; implement the exact host twin and measure its bounded-prefix recall.
- **B3 — §2 ion assignment:** fix radix completeness, hydrogen-bound validation, label bindings/cap policy, safe loss masking, parent-compatible evidence semantics and serialized assignment outputs.

**Verdict: not ready.**

---

# Third review (sections 3 and 4, first version)

Reviewer: codex exec (read-only), 2026-10-03. Verdict: not ready; the section texts were revised to answer the list below.

1. **§4.2 — blocker: “exact equality” lacks essential conditions.** The DFS must require equal atom counts, an injective complete assignment, and induced bond-order equality. Preserving existing bonds alone can accept a graph with extra bonds when hashes/refined labels collide. Equal bond counts plus a bijection preserving every bond is also sufficient. **Fix:** “Reject unequal atom/bond counts; maintain an injective mapping; for every assigned pair compare adjacency orders, including order 0 for non-bonds. Return equal only after a complete bijection.”

2. **§§3.2, 4.2, 4.4 — blocker: binding layouts are not implementable as specified.** `ms2_init_trajectories_kernel` already has six arrays ([ops/ms2.rs:6693](src/tensor/ops/ms2.rs:6693)); adding `traj_formula` makes seven unless an existing input is consolidated. Replay/sample/validate currently use **6/6/4** arrays, unchanged at A=32. Identity scratch and wide-output packing have no defined layouts or binding budgets. **Fix:** “Specify every kernel signature and packed-buffer offset. Consolidate initialization inputs to ≤6 arrays; pack candidate integer/float fields before `ms2_pack`, allowing rank + two source buffers + two destination buffers + returned_count = six.”

3. **§4.3 — major: the split protocol conflicts with frozen contracts and leaves leakage paths open.** Contracts §1 already defines separate validation, ranking and calibration identity groups; §4.3 replaces that scheme. It also never explicitly says the frozen generator was trained **only on fit**. Formula tables, enumeration bounds, assignment artifacts or structural pretraining fitted on rank structures can expose their labels despite excluding rank spectra from generator training. Generating candidates for unseen rank molecules is otherwise entirely well defined. **Fix:** “Use the frozen contract splits, or version their replacement explicitly. Exclude ranking/calibration/report identity groups from generator training and fitted structural artifacts; freeze the generator before candidate generation and reranker fitting.”

4. **§4.4 — major: compact output contradicts `CandidateBatch` semantics.** Contracts §3.5 requires B×K trajectory-ordered records, including failed requests. B×R ranked records need a separate schema. An empty record of all zeros falsely gives formula row/rank 0; fatal requests also require `request_failed`. Default returned=10 exceeds the existing default K=8. **Fix:** “Define `PackedCandidateBatch` with capacity R, returned_count, original trajectory IDs and explicit padding sentinels. Validate `1 ≤ R ≤ K`, with a compatible default. Never select `u32::MAX` ranks; returned_count=min(R, eligible_count). Preserve request fields and define failed-request padding.”

5. **§4.4 — major: resident results lack ownership guarantees.** Existing generation reuses workspace buffers. A resident result followed by another generation could silently expose the second request through the first result’s `read()`. **Fix:** “Resident results own or lease their buffers until completion/read/release; workspace reuse cannot overwrite outstanding results.” The intended read counts—generate **1**, packed **1**, resident **0**, deferred read **1**—are feasible with the existing batched read approach, provided NumPy field conversion introduces no additional reads.

6. **§§3.3–3.4 — major: lengths miss the failure-detection invocation, and the proposed failure fixture contradicts initialization.** Initialization writes START and length=1 ([ops/ms2.rs:6769](src/tensor/ops/ms2.rs:6769)); root failure sets `no_valid_action` without increasing length ([ops/ms2.rs:6042](src/tensor/ops/ms2.rs:6042)). At sampling step t=1, `length > t` reports inactive despite an active failure-detection invocation. Never-started length=0 and truncated length=T are otherwise handled correctly. Under the legal derived cap, STOP prevents truncation; truncation fixtures require a deliberately shorter low-level horizon. **Fix:** “For t=1..T−1, count length>t plus no_valid_action trajectories with length=t. Root failure retains START, length=1, and emits no sampled token. Enumerate failure leaves without fabricated token probability; test truncation separately with a shortened kernel horizon.”

7. **§4.2 — major: hash invariance needs synchronous rounds and an explicit identity scope.** Commutative sums are valid isomorphism invariants with deterministic wrapping arithmetic and synchronous refinement. In-place atom-by-atom updates would make hashes depend on atom order, allowing a wrong ‘different’ decision. Refinement also cannot distinguish uniformly labeled triangular-prism and K₃,₃ graphs: both have six degree-3 vertices and nine identical bonds. Exact checking handles that blind spot. Formula rank makes the hash an invariant of **graph plus conditioning formula**, not graph alone. **Fix:** “Use two label banks; specify wrapping-u32 hash arithmetic; compare formulas by composition, never enumeration’s shared `formula_row=u32::MAX`. State whether deduplication is per formula or across formulas, and omit formula rank for global graph identity.”

8. **§4.2 — major: resource bounds and duplicate/unresolved precedence are incomplete.** A pair budget of 4,096 still permits 2,016 comparisons per spectrum at K=64. At A=32, bond scratch needs up to **39 bonds**; label banks, mapping, used-target flags and DFS cursors also need explicit sizing. “Both candidates are kept” conflicts with ranking exclusions if either already has a proven duplicate flag. Bits **7/8 are free** after contracts’ bits 0–6, but the contracts and output validators need updating. **Fix:** “Specify lane-owned scratch widths, checked memory estimates and total comparison bounds. Read immutable validation flags and write identity results separately. Preserve unresolved pairs unless an independent exact duplicate proof exists; define whether unresolved status propagates to both endpoints. Define the relationship between duplicate_trace, duplicate_graph and identity_resolution.”

9. **§4.3 — major: feature and training-label definitions remain incomplete.** Missing-evidence defaults are supplied, but assignment-disabled support flags and zero/unknown fragment-tolerance denominators are not. Containment `WorkLimit` is unresolved, not a justified negative training label; existing metrics deliberately count it as a miss. **Fix:** “Specify finite feature values for disabled/empty/incomplete evidence and a positive, versioned residual denominator. Train on validated eligible candidates with resolved induced-containment labels; exclude and count unresolved labels. Freeze candidate-generation, deduplication and sampling policies with the reranker artifact.”

10. **§4.5 — major: Python integration and parity are underspecified.** The extension is `_mamba3_rl` ([lib.rs:269](bindings/python/src/lib.rs:269)); packaging currently includes `mamba3_graph` ([pyproject.toml:31](bindings/python/pyproject.toml:31)). A new `mamba3_ms2` import needs registration, packaging and stubs. Existing error mapping is variant-dependent—ValueError/OSError/NotImplementedError/RuntimeError ([err.rs:24](bindings/python/src/err.rs:24))—rather than one exception class. P6.6 also requires resident **inputs** and encoding methods, absent here. **Fix:** “Register MS2 classes in the existing extension and package a `mamba3_ms2` facade; reuse the documented error mapping with identical messages; specify resident-input and encoding APIs, FP32/backend restrictions, and array ownership.” Existing strided-array copying supports the proposed layout policy; it is not a demonstrated defect.

11. **§§3.1–3.2, 4.1 — minor: tighten claims and boundary wording.** No A=16/T=22 capacity constants require replacement in replay/sample/validate. At A=32: state=112 words, record=204 words, logits=795; pointer indices 0–31 fit one u32 mask and u8 host pointers; replay’s four legality words remain sufficient. Compiled vocabulary widths remain 5/18/4 ([decoder.rs:226](src/models/ms2/decoder.rs:226)), while STEP_ROWS=64 is already sufficient ([decoder.rs:41](src/models/ms2/decoder.rs:41)). K<top_count and equal allocation probabilities are already defined; top_count=0 is not. Fixed f32 order does not guarantee cross-backend allocation equality. RNG keying remains batch-independent. Finally, validation currently does not explicitly require STOP ([ops/ms2.rs:6931](src/tensor/ops/ms2.rs:6931)). **Fix:** “Say no V0 *capacity* is compiled in; retain frozen vocabulary widths. Define zero-count allocation as all sentinels, require exactly K assignments, and limit repeatability guarantees to the same backend. Validate STOP for claimed finished records while preserving legal truncated histories.”

not ready