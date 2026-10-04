# Codex review: host twins for graph identity and trajectory allocation (H6)

Reviewer: codex exec (read-only), 2026-10-04. Verdict: reject; the five findings are fixed in the follow-up.

1. **Major — [src/models/ms2/identity.rs:244](src/models/ms2/identity.rs:244): refinement reads the wrong bank and hashes the wrong round.**  
   Initially, bank 0 holds `hash(type, degree)` and bank 1 holds degrees. Round 0 reads bank 1 and overwrites bank 0, discarding atom types. Round 3 writes bank 1, but the final hash and equality pruning read bank 0. They therefore use **three refinements starting from raw degrees**, rather than four starting from typed labels.

   **Failing input:** legal singleton traces `START; ADD_ATOM(type=1,bond=0,pointer=0); STOP` and the equivalent type-4 trace. Both currently hash to `0x688990c0`. The specified algorithm, using the existing mixer and full mask, produces `0xe96f5975` and `0x97e8f25e`, respectively.

   **Fix:** read bank 0/write bank 1 on even rounds, reversing on odd rounds. Add independently computed expected hashes and final-bank labels. Existing invariance and canonical-equality tests cannot detect this defect because the weaker hash remains isomorphism invariant.

   **Soundness qualification:** this does **not** invalidate “hash mismatch means different” for legal traces with matching capacities. The implemented hash still uses synchronous, commutative wrapping sums, incident-bond degree, bond orders and atom/bond counts, without atom-index labels. Exact equality separately checks atom types.

2. **Major — [src/models/ms2/allocate.rs:113](src/models/ms2/allocate.rs:113), [identity.rs:172](src/models/ms2/identity.rs:172), [identity.rs:438](src/models/ms2/identity.rs:438): the lanes are not literal kernel-portable twins as documented.**  
   **Concrete trigger:** proportional allocation with two formulas and `K=5` allocates three dynamic `Vec`s: floors, fractions and picked flags. Unlike the fixed arrays in `ion_lane_visit`, these require a structural rewrite for a kernel.

   The hash lane performs `usize` arithmetic for token addressing: saturating multiplication/addition and wrapping additions. Equality derives capacity using `usize` division and constructs two dynamic suffix slices. These exceed the stated “`usize` only for indexing/bounds” convention.

   **Array accounting:**
   - Hash kernel: actions, hash output, graph scratch = **3** bindings.
   - Exact-pair lane: two scratch views, two type views, stack = **5** array arguments; `candidate_fits` additionally receives two label views, making **7** views. Those label views alias scratch, so seven physical bindings are **not inherently necessary**.
   - Architecture identity kernel = **5** bindings, but the host wrapper’s separately decoded type arrays have no counterpart in that layout. They must be reconstructed locally or explicitly accommodated.
   - Architecture allocation kernel = **5** bindings; externalizing its three temporary arrays would make **8**, exceeding six.

   **Fix:** use fixed local arrays under the documented caps, pass scalar capacities and offsets, address existing buffers with `u32`, and decode types locally from actions. There is no lane recursion, explicit `u64/i64` arithmetic, or early return inside nested loops.

3. **Minor — [src/models/ms2/identity.rs:369](src/models/ms2/identity.rs:369): capacity inference accepts incompatible scratch layouts and can report false “different.”**  
   **Failing input:** hash a seven-atom, type-3 path using `A=32,R=8`, producing 181-word scratch rows. Compare its endpoint-root BFS trace against its centre-root BFS trace, but supply an 87-word stack (`A=29`). All layout guards pass: the lane infers bond capacity 41 and label offset 123 instead of 117. The shifted labels force incompatible root mappings, and a sufficiently funded comparison reports different although the graphs are isomorphic.

   **Fix:** pass the actual `A` and bond capacity explicitly and validate both buffers against them. Return unresolved for inconsistent layouts.

   This is outside the correctly sized wrapper path. For normal bounded capacities, I found no out-of-bounds access from deriving `A`, and the guarded accesses do not themselves establish a memory-safety defect.

4. **Minor — [src/models/ms2/allocate.rs:130](src/models/ms2/allocate.rs:130): floating-point overshoot wraps the remainder and can eliminate the mandatory allocation.**  
   **Failing input:** finite log probabilities `[0.0,-100.0]`, `top_count=2`, `K=16_777_221`, full output buffer. `K'=16_777_219` rounds to `16_777_220.0f32`; floors become `[16_777_220,0]`. The remainder wraps to `u32::MAX`. Both slots receive an extra pick, but bounded slot-order output assigns every trajectory to slot 0, leaving slot 1 with none.

   **Fix:** explicitly enforce the production `K<=64` bound, or reconcile floor overshoot before unsigned subtraction while reserving every baseline assignment. Add an overshoot regression if the public lane promises arbitrary `u32 K`.

   Production configuration restricts `K` to 64, so this is a defensive/API-domain defect. Within that bound, the at-least-one ordering, `K<top_count`, zero-count sentinel, smaller-slot ties and documented buffer clamps follow the specification.

5. **Minor — [tests/ms2_identity.rs:560](tests/ms2_identity.rs:560): the advertised induced-equality regression exits before checking adjacency.**  
   **Failing test scenario:** the path has three bonds and the cycle has four. `graph_equal_lane` rejects the unequal counts before inspecting the overwritten labels or any mapped pair. Removing the adjacency checks would still pass this particular test.

   **Fix:** compare same-atom-count, same-bond-count non-isomorphic graphs with equalized labels, and add a same-count bond-order counterexample. The prism/K3,3 test already supplies useful independent topology coverage; this test’s claimed purpose is nevertheless unverified.

For correctly sized, legal inputs, I found no additional DFS soundness defect: mapping is injective and complete, pair checks include bond orders and non-bonds, cursors resume correctly, and backtracking clears used flags. Work consumption is deterministic and monotone. At `work_max=0`, a viable nonempty assignment yields unresolved; an immediately disproved pair can yield different, and two empty decoded graphs yield equal.

`identity_batch` compares eligible `j<k` records within each spectrum, including earlier duplicates. Equality sets bit 7 even when `j` is already duplicate; any unresolved comparison independently sets bit 8 and resolution 2, including when another comparison proves equality. Ineligible records receive resolution 1. It leaves `duplicate_trace` untouched. Eligibility and malformed-trace handling rely on validated input: decoding correctly follows legal root, parent-pointer and newest-atom closure semantics, but permissively skips malformed/truncated words rather than validating them.

The prism and K3,3 constructions are legal: type 1 supports their degree-three single bonds, and each needs exactly four closures. The test uses `R=8`, so it does not explicitly assert legality at `R=4`. A read-only reproduction of the seeded random generator found **37 equal singleton pairs**; equality coverage is therefore not wholly vacuous, but nontrivial isomorphic pairs are not guaranteed. Add generated permutations/retraces and assert resolved nontrivial equality counts.

Allocation’s operation order is explicit, but host/device `exp` differences can cross **floor boundaries**, not merely fractional-part ties. NaN/infinity follow deterministic host fallback behavior but are outside the documented domain and depend on Rust’s saturating float casts; they need an explicit kernel policy. Signed zero behaves normally.

No files were modified and no cargo commands were run. Counterexamples were checked with read-only Python reproductions.

**Verdict: reject.**

---

# Second review of the ion, identity and allocation host twins after their fixes

Reviewer: codex exec (read-only), 2026-10-04. Eight of ten earlier findings resolved, two partly. The remaining items (full-buffer addressing with offsets, complete allocation records, the inactive-slot addition, the helper bound, the `embedding_ion` test) are carried into the device-kernel tasks, where the twin and the kernel are brought to the same signature.

The arithmetic and DFS fixes are largely sound. **Eight findings are resolved; two are partly resolved.** The remaining concerns mainly affect the promised literal GPU copies and their regression coverage.

No files were modified and cargo was not run. Independent Python checks compared 50,000 radix-budget cases and 50,000 verdict cases against wide-integer references; all matched.

The ten earlier findings:

1. **Ion #1 — resolved.** The wrapper is explicitly excluded from the kernel copy at `src/models/ms2/ion.rs:364–370`. Digit extraction, guarded heavy-mass accumulation and verdict comparisons use u32 at `:269–342`. See remaining problem 4 for a narrower arithmetic caveat introduced by the rewrite.

2. **Ion #2 — resolved.** `examples/ms2_ion_report.rs:183–207` counts inspected, domain-excluded, unknown-precision, unlabeled and labeled spectra. `:362–377` emits the full denominator, coverage, inspected-limit semantics and oracle-conditioned naming.

3. **Ion #3 — resolved.** `examples/ms2_ion_report.rs:225–260` constructs uncapped and uploaded label sets and counts upload losses. `:329–338` evaluates uncapped labels against retained hypotheses; `:378–397` separately reports uncapped retention, uploaded-label retention and overflow counts.

4. **Ion #4 — partly resolved.** `tests/ms2_ion.rs:623–659` independently derives supported compositions and masses, and `:689–700` requires coverage for every adduct/shift category. However, the sweep no longer calls `embedding_ion` at all. Making that helper return `None` for every negative-adduct case still passes this sweep. See remaining problem 1.

5. **Ion #5 — resolved.** `tests/ms2_ion.rs:938–943` adds an explicitly absent anchor and compares the complete output with the baseline. `:948–985` adds adduct-2, nonzero-shift and negative-hydrogen cases.

6. **Identity/allocate #1 — resolved.** Typed initial labels enter bank 0 at `src/models/ms2/identity.rs:259–261`; even rounds read bank 0 and odd rounds read bank 1 at `:265–294`. Four rounds finish in bank 0, which is summed at `:299–306`. `tests/ms2_identity.rs:489–525` independently checks hashes and final labels.

7. **Identity/allocate #2 — partly resolved.** Allocation now uses fixed local arrays at `src/models/ms2/allocate.rs:216–218`; equality decodes types locally into `[u32;32]` arrays at `src/models/ms2/identity.rs:465–469`; dynamic label suffixes are gone. But the literal-copy claim still lacks full-buffer addressing and allocation-record emission. Concrete failures appear in remaining problems 2 and 3.

8. **Identity/allocate #3 — resolved.** `src/models/ms2/identity.rs:447–463` validates explicit atom/bond capacities and exact scratch/stack lengths. Layout failure returns unresolved at `:479–480`. The original seven-atom path/87-word-stack regression is covered at `tests/ms2_identity.rs:898–960`.

9. **Identity/allocate #4 — resolved for the allocation lane.** `src/models/ms2/allocate.rs:141–148` enforces `K<=64`, `F<=8` and `top_count<=F`; `allocate_checked` rejects invalid ranges at `:285–299`. Reconciliation precedes remainder allocation at `:230`. The original `K=16_777_221` input is now rejected. See remaining problem 5 for the independently public helper.

10. **Identity/allocate #5 — resolved.** `tests/ms2_identity.rs:809–856` compares equal-count nonisomorphic graphs with both labels and hashes forced equal. `:858–893` adds an equal-count bond-order counterexample. These reach the adjacency checks.

Remaining or new problems:

1. **Minor — the rewritten superset test leaves `embedding_ion` untested.**  
   **Evidence:** `tests/ms2_ion.rs:629–684`; production helper at `src/models/ms2/ion.rs:840–864`.

   **Failing implementation:** return `Ok(None)` whenever `adduct_id == 2`, or whenever `shift != 0`. The rewritten sweep uses independently calculated compositions directly in `ion_assign`, so neither mutation affects it.

   **Fix:** keep the independent oracle, then also compare `embedding_ion` against its expected `Some`/`None`, composition and mass before checking assignment coverage.

2. **Major — record-local slices still prevent literal copying onto the stated full-buffer bindings.**  
   **Evidence:** `src/models/ms2/identity.rs:24–26` claims explicit offsets, but `graph_hash_lane` at `:174–180` and `graph_equal_lane` at `:432–444` accept no record-base offsets. The host supplies sliced rows at `:698–702` and `:724–744`. Allocation similarly infers F from the entire supplied slice at `src/models/ms2/allocate.rs:129`.

   **Concrete input:** bind a valid allocation batch with `B=2,F=8,K=4`. The bound `top_log_prob` has 16 words. A literal copy interprets this as `F=16`, fails the cap guard at `allocate.rs:144–145`, and writes sentinels. For identity, two 181-word scratch rows presented as the full binding are zeroed together by `identity.rs:203–205`; equality rejects the 362-word binding against the expected 181-word row at `:459–463`.

   **Fix:** pass scalar F, record/spectrum indices, strides and base offsets; address the original bound arrays through them. Check available ranges against each record’s extent. Preserve record-local slice adapters in the host wrapper if useful.

   The **physical binding counts themselves are compliant**: ion 5, graph hash 3, identity 5, allocation 5. Local arrays and aliased row views do not add bindings.

3. **Major — allocation’s claimed kernel twin does not emit the specified trajectory records.**  
   **Evidence:** `src/models/ms2/allocate.rs:78–89` explicitly implements only the slot-word projection. `top` and `top_counts` are length-checked at `:150–154` but never read for emission; output stores at `:255` write only formula slots. Architecture §3.2 requires `[B,K,12]` records containing slot, source id and ten counts.

   **Concrete input:** `F=1`, `top=[77,0]`, counts `[1,4,0,0,0,0,0,0,0,0]`, `top_count=1,K=2`. The required output is two complete records beginning `[0,77,1,4,…]`. Copying this lane produces only `[0,0]`. The stated next initialization kernel cannot fill those fields because it no longer binds `top` or `top_counts`.

   **Fix:** add complete record emission inside the allocation twin using the existing five bindings, including sentinel slot and zero counts for an empty retained set. Verify exactly K complete records and untouched surrounding output.

4. **Minor — inactive hydrogen slots execute overflowing mass additions.**  
   **Evidence:** `src/models/ms2/ion.rs:335–336` computes `cand` unconditionally even when `slot` is false.

   **Concrete input:** parent `C2 N305 H0`, adduct 1, peak `4_294_937_021`, uncertainty 0, ppm tenths 100, work budget at least 917. Visit 917 has heavy mass `4_294_937_570`. Only hydrogen 0 fits, but inactive slot 1 still forms:
   ```
   4_294_937_570 + 1_007_825 = 4_295_945_395
   ```
   which wraps to `978_099`.

   The value is discarded, so this is **not a demonstrated acceptance defect**. It nevertheless violates the requested absence of reachable mass overflow, with every parent count below 1023.

   **Fix:** form candidate mass and its dependent calculations only inside the `slot` guard. Keep intentional offset-binary wrapping explicit and separate from mass accumulation.

5. **Minor — the public overshoot helper can wrap its sum before reconciliation.**  
   **Evidence:** `src/models/ms2/allocate.rs:50–54`, `:57–72`. Its documented reconciliation guarantee does not state a bound on the floor sum.

   **Concrete input:** `n=2`, `rest_k=u32::MAX`, floors `[2_147_483_648,2_147_483_648,0,…]`. The exact sum is `2^32`, an overshoot of one. The helper wraps the sum to zero, leaves both shares unchanged and returns `u32::MAX` rather than shaving one and returning zero.

   This is unreachable through the capped allocation lane.

   **Fix:** either document and enforce the helper’s production bounds, or reconcile by consuming remaining headroom per slot without ever forming an overflowing sum.

Within valid record-local inputs, I found no further decision or DFS defect: mixed-radix extraction and the exact `2^32` product handling are correct; verdict subtraction comparisons match all contract boundaries; emitted residuals decode correctly. Identity checks injectivity, complete mappings, bond orders and non-bonds. Allocation’s capped slot decisions preserve baselines, use the non-finite fallback and write exactly K slot words when output capacity is sufficient.

**ion: accept-with-fixes**  
**identity: accept-with-fixes**  
**allocate: reject** — the advertised literal kernel twin still omits required record fields and full-buffer addressing.

---

# Review of the identity, allocation and ion KERNELS and their twins

Reviewer: codex exec (read-only), 2026-10-04. Verdicts: identity reject, allocation accept-with-fixes, ion reject; the findings are the follow-up task.

Four of the five carried-over problems are resolved. The overshoot helper remains unfixed. I also found new correctness and verification issues below.

This was a read-only source review. No files were modified and no cargo commands were run. Independent Python checks covered 30,000 radix/budget cases, 30,000 verdict cases, and the arithmetic counterexamples below. Runtime/compiler behavior remains unverified.

1. **Major — the hash twin panics on ordinary legal graphs in debug builds.**  
   [src/models/ms2/identity.rs:323](src/models/ms2/identity.rs:323), [identity.rs:339](src/models/ms2/identity.rs:339)

   The commutative refinement and final-label sums now use plain `+=`. Device arithmetic wraps; ordinary Rust debug arithmetic checks overflow. Cargo.toml does not disable those checks.

   **Failing input:** `START; ADD(type=4); ADD(type=4, order=1, pointer=0); STOP`, with valid capacities. Both final labels are `0xfd5474be`; their sum is **8,500,341,116**, so the second final-sum addition panics. Higher-degree graphs can overflow during refinement too.

   **Fix:** express intentional wrapping explicitly in the host twin. Kernel spelling can remain plain device addition. The literal-copy constraint must not change host semantics.

2. **Major — evidence counts and records include shifts that fail the specified mass-consistency rule.**  
   [src/tensor/ops/ms2_ion.rs:1284](src/tensor/ops/ms2_ion.rs:1284), [src/models/ms2/ion.rs:1268](src/models/ms2/ion.rs:1268)

   Both implementations calculate base status 0 for an unsupported shift, then increment `count` and emit its record anyway.

   **Failing input:** a whole 2-butanol candidate, zero open valence, positive adduct, and four peaks for its heavy composition with `s=+1`, followed by a peak with `s=0`. These are valid assignment hypotheses under the parent hydrogen ceiling. Only the last peak is candidate evidence under architecture §2.4, but the implementation returns status 2, count 5, and four records containing only the invalid `+1` shifts.

   This also contradicts padding beyond the evidence count: a lone unsupported shift produces status 0 with count 1 and a populated record.

   **Fix:** increment/count/store only when `base > 0`; preserve the first-E ordering among qualifying peaks. If raw composition matches are useful diagnostics, expose them separately and filter them before producing candidate evidence. Existing tests explicitly encode the incorrect behavior at `tests/ms2_ion.rs:1127` and `tests/ms2_ion_kernels.rs:938`.

3. **Major — shape checks do not prove that device addresses fit u32.**  
   [src/tensor/ops/ms2_ion.rs:269](src/tensor/ops/ms2_ion.rs:269), [ms2_ion.rs:800](src/tensor/ops/ms2_ion.rs:800)

   The assignment launcher checks `B*F*N` against **usize**, but the kernel calculates `lane*J*12` in **u32**. Label-mask and evidence addressing have the same narrowing issue; the identity/allocation launchers also lack complete address-range checks.

   **Failing input:** `B=21,846, F=8, N=256, J=8`. The ion tensor contains **4,295,098,368 words**. The last spectrum starts at word **4,294,901,760**; offset 65,536 wraps to zero. Later lanes overwrite earlier lanes, destroying single-writer ownership and diverging from the usize-addressed twin.

   B has no documented absolute maximum. The default 2-GiB workspace budget prevents this particular allocation upstream, but these public launchers do not enforce that budget.

   **Fix:** use checked host arithmetic for every binding’s maximum address and every scalar narrowing; reject layouts exceeding device indexing limits before launch.

4. **Minor — allocation accepts reduced-precision tensors but computes in their storage precision.**  
   [src/tensor/ops/ms2_identity.rs:888](src/tensor/ops/ms2_identity.rs:888), [ms2_identity.rs:1062](src/tensor/ops/ms2_identity.rs:1062)

   The twin and specification use f32. The kernel uses generic `F` for subtraction, exp, accumulation, division, quota and fractional remainder. `FloatElem` admits f16 and bf16.

   **Failing input:** f16 log probabilities `[0, -1.6083984375]`, `top_count=2`, `K=5`, proportional mode. With those same stored values widened to f32, quotas are approximately `[2.4995668, 0.5004333]`, yielding total allocations `[3,2]`. Half-rounded arithmetic gives quotas `[2.5,0.5]`; the smaller-slot tie yields `[4,1]`.

   **Fix:** widen loads and perform allocation arithmetic in f32, or restrict this API to f32. The current contract’s F32 configuration avoids this exposed-API defect.

5. **Minor — finite log probabilities can incorrectly trigger the non-finite fallback.**  
   [src/models/ms2/allocate.rs:39](src/models/ms2/allocate.rs:39), [src/tensor/ops/ms2_identity.rs:992](src/tensor/ops/ms2_identity.rs:992)

   Both implementations define “finite” as strict membership in `(-3e38,3e38)`. That excludes valid finite f32 values.

   **Failing input:** `[0,-3e38]`, `top_count=2`, `K=5`, proportional mode. The specified proportional result is slot counts `[4,1]`; the range check falls back to round robin, producing `[3,2]`.

   **Fix:** detect NaN/infinity accurately, or explicitly specify and validate the narrower input domain.

6. **Minor — the public overshoot helper still overflows before reconciliation.**  
   [src/models/ms2/allocate.rs:151](src/models/ms2/allocate.rs:151)

   **Failing input:** `n=2`, `rest_k=u32::MAX`, floors `[2_147_483_648,2_147_483_648,0,…]`. Debug builds panic; wrapping builds sum to zero and return `u32::MAX`, leaving the overshoot unchanged. Also, `n=9` indexes beyond the fixed array.

   This remains unreachable through the capped allocation lane.

   **Fix:** enforce documented helper bounds, including `n<=8`, or reconcile using overflow-free headroom arithmetic.

7. **Minor — exact-comparison layout validation no longer checks actual buffer extents.**  
   [src/models/ms2/identity.rs:417](src/models/ms2/identity.rs:417)

   `layout_ok` checks capacities and strides, but not whether the action, hash, scratch and stack ranges exist. Missing scratch reads become zero.

   **Failing input:** two legal two-atom type-1 traces, one with a single bond and one with a double bond; hashes `[0,0]`; correct scalar strides; empty scratch; complete actions and stack; sufficient work. Counts and types agree, all missing labels/bond orders read as zero, and the public helper returns **equal** for non-isomorphic graphs.

   Conversely, retaining only the first scratch row of two identical nonempty graphs can make it return **different**.

   **Fix:** validate checked record extents and stack extent before decoding; return unresolved on incomplete layouts. Correctly shaped kernel calls avoid this defect.

8. **Minor — label-mask test expectations are computed but never compared.**  
   [tests/ms2_ion_kernels.rs:613](tests/ms2_ion_kernels.rs:613), [ms2_ion_kernels.rs:645](tests/ms2_ion_kernels.rs:645)

   `want_mask` and `want_state` are discarded. Current callers explicitly compare their small fixtures, so those assertions are useful, but they do not establish the advertised general twin comparison.

   **Failing implementation:** skip all hypothesis indices above zero. Every current positive mask match is at hypothesis 0, so the existing mask cases still pass.

   **Fix:** compare both complete outputs in the harness; add a positive match at hypothesis 1 and a multispectrum case with distinct labels.

9. **Minor — the contracts document has not incorporated the implemented identity/evidence statuses.**  
   [docs/MS2_CONTRACTS.md:224](docs/MS2_CONTRACTS.md:224), [MS2_CONTRACTS.md:552](docs/MS2_CONTRACTS.md:552)

   It still describes evidence status 0 and identity resolution 0 as the available values, and lists candidate bits only through 6.

   **Concrete conflict:** duplicate finished trajectories produce bit 7/resolution 1; budget exhaustion produces bit 8/resolution 2. Those outputs cannot be interpreted from the specified contract alone.

   **Fix:** incorporate architecture §2.4/§4.2’s schema and status additions.

10. **Disposition of the five carried-over problems**

   | Problem | Result | Evidence |
   |---|---|---|
   | Record-local slices | **Resolved** | `identity.rs:216`, `identity.rs:401`, `allocate.rs:195`; full buffers, explicit record/spectrum indices and bases |
   | Incomplete allocation records | **Resolved** | `allocate.rs:79`, kernel `ms2_identity.rs:965`; slot, source and all ten counts emitted |
   | `embedding_ion` test | **Resolved** | `tests/ms2_ion.rs:643`, `:676`, `:685`, `:690`, `:722`; independent expected composition/mass, Some/None checks and category coverage |
   | Inactive-slot mass addition | **Resolved** | host `ion.rs:498`, kernel `ms2_ion.rs:613`; candidate mass formed inside the slot guard |
   | Overshoot helper bound | **Not resolved** | `allocate.rs:151–155`; finding 6 |

11. **Statement-level kernel/twin comparison**

   These are the semantic differences I found, including domain-dependent differences. Scalar expansion and helper inlining preserve decisions unless noted.

   | Kernel | Differences |
   |---|---|
   | **graph hash** | Host returns the hash as well as storing it; kernel only stores. Kernel has an unreachable `cap_a==0` branch after normalizing it to at least 1 (`ms2_identity.rs:208`). ADD/CLOSE constants 2/3 match the host constants. Mixer constants and operand order match. Record/scratch bases, bond triples and bank offsets match. Plain host hash sums panic in checked builds versus device wrapping: finding 1. |
   | **graph identity** | Kernel inlines count decoding and DFS; host calls helpers and returns `(bits,resolution)`. Kernel stores that pair and derives its stack base from the record. Numeric status constants match. Kernel omits the host count decoder’s unused type load. Counts, type/label pruning, adjacency operands, cursors, backtracking, budget checks and bit accumulation otherwise match. |
   | **allocate** | Kernel inlines count loads and finite detection. Float loads are direct; host `lp_at` is guarded, but the preceding valid-range checks make this equivalent on supported layouts. Generic float precision differs from f32: finding 4. Scalar shares, reconciliation, strict-`>` selection and emission order match. |
   | **ion assign** | Host buffer addressing uses usize; kernel uses u32: finding 3. Kernel scalar-expands the nine radices/digits and table accesses. Manual saturation, absolute differences and wrapping subtractions match host helpers. Kernel handles each guarded hydrogen slot immediately; host constructs a visit result then consumes its valid prefix. Valid slots are contiguous, making these orders equivalent. |
   | **ion label mask** | usize host addressing versus u32 device addressing. Float storage is generic in the kernel, but exact 0/1 values are preserved. Boolean flags become 0/1 scalars. Raw-index comparisons, ten-count comparisons, valid flags and state transitions match. |
   | **ion evidence** | usize host addressing versus u32 device addressing. Nine heavy counts are scalar-expanded; helpers are inlined. Type mapping, hydrogen decoding, ceilings, residual tie-break, shift sign, status maximum, bit 7 and first-four emission match—including finding 2. |

   I found no additional constant, offset, operand-order or stride discrepancy within bounded valid layouts. Integer operand grouping remains equivalent modulo \(2^{32}\).

12. **Identity algorithm**

   In wrapping arithmetic, the hash is an isomorphism invariant:

   - Typed initial labels occupy bank 0.
   - Four synchronous rounds alternate banks; each reads only the old bank.
   - Neighbor contributions and final labels use commutative sums.
   - Atom indices select storage/edges; they do not enter labels.
   - Counts enter the final hash; formula identity does not.

   Exact comparison requires equal atom/bond counts, then a complete injective assignment preserving atom types, refined labels and every assigned pair’s bond order—including zero.

   DFS handling is correct on complete buffers: assignment/used/cursor occupy separate banks; successful placement records the next cursor; exhausted child depth resets its cursor; backtracking clears the parent image’s used flag while preserving the parent’s resume cursor.

   Work counts **viable tentative assignments**, not rejected candidates or adjacency probes. A completed mapping can resolve equal using exactly its budget. Zero work can still resolve an immediate contradiction as different, or empty decoded graphs as equal.

   Eligible comparisons are finished, not `INVALID_FINAL`, same spectrum, `j<k`, equal hash. Earlier duplicates remain comparison partners. Equality sets bit 7; any unresolved pair independently sets bit 8 and resolution 2—even if another pair proves equality. Ineligible records receive `(0,1)`.

   At `A=32`, 39 bonds: graph scratch is **181 words**—117 bond words, labels 117–148 and 149–180. DFS scratch is **96 words**—assignment 0–31, used 32–63, cursor 64–95.

   **Soundness:** for legal traces, complete consistent scratch and wrapping arithmetic, I found no false-different or false-equal path. Finding 7 supplies concrete counterexamples for incomplete public-helper buffers; finding 1 prevents normal debug execution before comparison.

13. **Allocation algorithm**

   On capped, complete f32 inputs, both implementations write exactly K complete records:

   - Round robin uses `t % top_count`.
   - Proportional allocation reserves one per retained slot, then floors extra shares and selects largest remainders.
   - Strict `>` in ascending slot order gives smaller-slot ties.
   - Emission groups trajectories by slot.
   - `K<top_count` assigns the first K ranked formulas once each.
   - `top_count==0` writes MAX slot/source and ten zero counts.
   - K≤64/F≤8 guards are present.
   - NaN and ±infinity fall back to round robin, subject to finding 5’s overly narrow detection.

   The scalar `f0..f7`, fractions and picked flags preserve the array algorithm. For f32, operation order is sequential max, sequential exp sum, exp/division, `K' * p`, floor, subtraction, reconciliation and sequential remainder selection.

   Device/host exp differences can cross floor or remainder boundaries. Ordinary cases and exact equal-probability ties are tested; no general test-side exclusion implements architecture §3.2’s `1e-4` boundary criterion.

14. **Ion assignment, mask and evidence**

   Assignment arithmetic matches the specified rules on bounded inputs:

   - Targets use `mz+549` and `mz−549` for the two adducts.
   - Mixed-radix order starts at index 1 with carbon least significant.
   - Exhaustion is exact, including the special \(2^{32}\) product case.
   - Division guards prevent heavy-mass multiplication/addition overflow, including iodine and counts up to 1023.
   - Parent residuals bound each hypothesis’s error.
   - Saturating `half_p` and the inclusive `half_p<=1,007,825` restriction are correct.
   - Hydrogen bounds intersect the mass window with the parent/adduct ceiling.
   - Verdict subtraction comparisons equal the wide contract rule at equality boundaries.
   - Accepted/ambiguous counters and first-J retention are separate.
   - Bits 0/1/2 represent exhaustion/capacity/unavailability.
   - Ion and ion_meta are fully initialized, including padding, all-zero formula slots and unknown precision.

   Label masking correctly uses raw indices, ten counts and valid flags. Partial overlap produces state 1; labels with no retained overlap produce state 2; absent labels produce state 0. Non-state-1 rows are one-hot unassigned. Cleared-valid overflow rows are ignored.

   Evidence correctly decodes heavy counts and parent hydrogens from types, computes `sum ceil(o/3)`/`sum o`, and uses `s=h−H(g)−h_a`. Base statuses and bit 7 are computed correctly; **record/count eligibility is incorrect** under finding 2. Unfinished trajectories write zero rows, although decoding occurs before that gate.

15. **Safety, binding counts and CubeCL pitfalls**

   Array counts are compliant: **hash 3, identity 5, allocation 5, assignment 6, mask 6, evidence 5**. Assignment’s extra `spec` binding supplies peak uncertainty separately from precursor uncertainty.

   With complete bounded layouts, ownership is disjoint by trajectory, spectrum or peak lane. Finding 3 breaks that proof for large layouts.

   Shape validation alone does not validate buffer contents. Evidence trusts `length<=steps` and `kept_n<=J`; malformed values can cause unchecked reads. These are valid upstream-producer invariants, but should be stated explicitly at the public launcher boundary or validated there.

   I found **no direct scalar-argument initialization of a variable subsequently updated inside a rolled loop**. `cap_a=atoms_cap` is adjusted before its scanning loops and only read in them; it is not the documented loop-carried trap.

   Loads inside branches remain in several helpers and hot paths. Source inspection cannot establish whether those trigger backend lowering failures. Eager logical operators are safe within validated shapes/count bounds: radices are nonzero, intentional off-guard subtractions wrap, and mass products occur inside guards. The allocation length/load `&&` at `ms2_identity.rs:926` is unsafe for truncated bindings, but the launcher supplies a complete top_count binding.

16. **Tests and remaining coverage**

   Poisoning and complete comparisons are present for hash/scratch, identity result words, allocation records, assignment/metadata and evidence rows. Identity DFS scratch is returned but not compared; untouched scratch for lanes that never search is legitimate.

   Coverage is meaningfully non-vacuous: independent hash labels, same-count topology/bond-order counterexamples, prism/K3,3, deliberate nontrivial retraces, assignment accept/ambiguous/reject classes, first-J truncation and mask states. The repaired embedding test now checks every adduct/shift category.

   Important gaps remain:

   - Debug-safe intentional hash wrapping.
   - Actual 32-atom/39-bond fixtures; large capacities alone do not fill those bounds.
   - Identity multispectrum isolation and combined duplicate/unresolved outcomes.
   - Device allocation at F=8/K=64, reduced precision and extreme finite values.
   - Device iodine/high-count overflow paths and the inactive-slot regression.
   - Exact hydrogen-window equality and radix-\(2^{32}\) paths in device tests.
   - Independent padding/unstarted-slot guards: the wide-window fixture disables those rows for another reason.
   - Positive label matches after hypothesis 0 and multispectrum masks.
   - Evidence routing across distinct spectra/formula slots and negative-adduct device evidence.
   - Independent evidence qualification; current tests reinforce the shared defect.

**Identity: reject.**  
**Allocation: accept-with-fixes.**  
**Ion: reject.**