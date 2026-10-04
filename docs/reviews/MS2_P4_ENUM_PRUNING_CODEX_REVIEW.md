# Codex review: train-fitted pruning of the formula enumeration (H2)

Reviewer: codex exec (read-only), 2026-10-03. Verdict: reject; the findings are fixed in the follow-up (H3).

1. **Major — Margin widening depends on training order.** [src/models/ms2/formula_enum.rs:590](src/models/ms2/formula_enum.rs:590)  
   Equal extremal fractions retain the first observed numerator/denominator, then widening changes only the numerator. **Failing input:** fit `[C4H8, C6H12]` with margin `2`. H/C becomes `[6/4, 10/4]`, admitting pentane `C5H12`. Reverse the training order and it becomes `[10/6, 14/6]`, rejecting pentane. Other stages admit it in both fits, and it lies within the train-derived domain. Thus reordering the same training molecules changes enumerated support and reported recall.  
   **Fix:** choose an order-independent representation or deterministic tie-break for equal fractions before widening. Add a permutation test asserting identical bounds and enumeration.

2. **Major — The independent ratio oracle has two incorrect acceptance paths.** [tests/ms2_formula_enum.rs:1031](tests/ms2_formula_enum.rs:1031), [tests/ms2_formula_enum.rs:1018](tests/ms2_formula_enum.rs:1018)  
   The lower fraction starts at `0/1` and updates only for a smaller nonnegative fraction, so it never learns a positive minimum. **Failing input:** train `[C2H2, C3H4]`, candidate `C1H0`, margin zero. The candidate satisfies caps, rare bounds and the fitted twice-DBE interval `[4,4]`. The oracle accepts it; production correctly rejects H/C below `1`.  
   Separately, zero carbon returns immediately, bypassing DBE. **Failing input:** train `[H2O]`, candidate `O`. The oracle accepts it; production rejects twice-DBE `2` outside `[0,0]`.  
   **Fix:** initialize both extrema from the first positive-carbon observation; let zero-carbon handling skip only ratio comparisons, then continue to DBE. Test both counterexamples directly.

3. **Minor — The exhaustive pruning test leaves important new rules unexercised.** [tests/ms2_formula_enum.rs:1097](tests/ms2_formula_enum.rs:1097)  
   Its domain contains no rare elements, carbon never crosses a bucket boundary, and the independent comparison uses only margin zero. The widened-fit test merely checks that training compositions still pass. Consequently, removing rare pruning/filtering would survive these checks, and the order-dependent widening in finding 1 is undetected.  
   **Concrete missing case:** train `[CH4, C2H2O2]`, then enumerate gold `C2H2O2`. At the carbon assignment, partial heavy total `2` lies in bucket 0, whose carbon cap is `1`; the valid completion lies in bucket 1, whose cap is `2`. This must survive the suffix-maximum check. The present production code handles it correctly, but the test does not establish that.  
   **Fix:** add independent cases across bucket boundaries, nonmonotone heavy caps, positive margins, rare totals/distinct counts, and combined node/capacity/scored limits. Assert nonzero survivors and specific pruning/rejection counters.

4. **Minor — Unseen interior carbon buckets have an undocumented widening policy.** [src/models/ms2/formula_enum.rs:567](src/models/ms2/formula_enum.rs:567), [src/models/ms2/formula_enum.rs:621](src/models/ms2/formula_enum.rs:621)  
   Unseen interior buckets start with zero caps, then receive `margin`; unseen heavy buckets retain an impossible DBE interval. The documentation explains only buckets beyond the tables.  
   **Concrete input:** train `[CH4, CH4O4, C8H18]`, margin `4`. Carbon bucket 1 was never observed, yet `C4H10` passes every fitted stage: its synthetic carbon cap is `4`, and heavy bucket 1 was observed through `CH4O4`. The independent oracle instead rejects unseen carbon buckets outright.  
   **Fix:** explicitly define and document interior-bucket support and margin behavior, and align the oracle. Track bucket presence if unseen buckets should reject.

The heap ordering, ambiguity alignment, exhaustion logic, carbon-first traversal and reachable-heavy-bucket pruning appear correct by inspection. Report recall predicates are cumulative in the documented leaf-filter order; denominators include every spectrum, out-of-domain gold misses, and fitting uses only the designated training export. No per-molecule JSON output was found. No files were modified and no cargo commands were run.

**Verdict: reject.**

---

# Second review: pruning fixes and the host twin of the device enumeration order (H3)

Reviewer: codex exec (read-only), 2026-10-04. Verdict: reject; the eight findings are the follow-up task.

Static review only: no files changed and no cargo commands run. The previous review file is absent; I used your description of its four findings.

1. **Blocker — the lane and its helpers are not a line-for-line `u32` kernel twin.**  
   [formula_enum.rs:2664](src/models/ms2/formula_enum.rs:2664) calls `dbe_range_contains`, whose endpoints are `i64` and whose implementation contains `i64::checked_neg` at lines 1758 and 1789. Reachable helpers also perform `usize` increments at lines 2453, 2567 and 2625; bucket indices are `usize` at lines 2537 and 2655. Output uses host `Vec` storage and `u16` compositions.

   **Concrete input:** an ordinary validated `CH4` query already executes the `usize` helper loops. A surviving candidate with a DBE interval reaches the helper accepting `i64` endpoints. The signed-negation branches are unreachable from the current lane after its exact filters, but the helper still cannot be copied literally as a `u32` implementation.

   **Fix:** pack DBE endpoints into the specified unsigned biased representation before dispatch; use `u32` loop/index variables and a kernel-compatible output interface. Distinguish the portable lane from host allocation and conversion code. There is **no call to `chem::decide`**, and no `u64` or `i128` arithmetic in the lane’s explicit helper bodies.

2. **Major — counter saturation does not set exhaustion or clear `complete`.**  
   [formula_enum.rs:2856](src/models/ms2/formula_enum.rs:2856) saturates totals, but line 2867 considers only lane exhaustion and scored truncation.

   **Concrete validated input:** domain heavy caps  
   `[255,255,255,7,7,7,7,3,0]`, `heavy_max=65535`, hydrogen bounds `[0,0]`; rare ranges `[0,31]` and `[0,5]`; matching bucket/presence arrays within 64 rows, with carbon presence false; valid bounded ratio fractions. Query: precursor `u32::MAX`, adduct 1, ppm 0, uncertainty 0; both limits `u32::MAX`.

   This produces exactly 16,384 rare lanes. Every lane visits at least all `C,N,O ∈ 0..=75`: their maximum combined mass is `4,204,995,812`, below the window end. Thus visits exceed **7,192,182,784**, while no candidates pass carbon presence. The result saturates `visited` to `u32::MAX-1` but reports `exhausted=false`, `absent=true`, `complete=true`.

   **Fix:** retain a saturation flag during aggregation and include it in exhaustion/status/complete. Test aggregation near the saturation threshold without enumerating billions of vectors.

3. **Major — the independent oracle still rejects valid production results.**  
   [ms2_formula_enum.rs:1093](tests/ms2_formula_enum.rs:1093) treats “no positive-carbon training composition” as requiring zero ratio numerators, ignoring the positive margin.

   **Concrete failing input:** train `[H2O]`, margin 2, candidate `CH2O`. Production fits upper ratio numerators to 2 with denominator 1, admits the candidate’s bucket caps and rare ranges, and admits twice-DBE 2 within `[-2,2]`. The oracle rejects immediately because hydrogen is nonzero.

   **Fix:** apply the margin to the default `(0,1)` fractions before comparison, including this branch. Add this counterexample independently of production helpers.

4. **Major — large tolerances wrap before the wide-window restriction.**  
   [formula_enum.rs:2807](src/models/ms2/formula_enum.rs:2807) calls `chem::tolerance`, whose narrowing cast at `chem.rs:455` truncates a potentially oversized quotient.

   **Concrete failing input:** domain/bounds fitted from `CH4`; precursor `u32::MAX`, adduct 1, `ppm_tenths=10_000_001`, uncertainty 0, unrestricted limits. The true tolerance is `4,294,967,724`; the returned tolerance is **428**. The device twin searches a narrow window and can report complete absence instead of the required wide-window exhaustion.

   **Fix:** clamp or reject the tolerance before narrowing, or enforce a documented query limit. Saturating additions to `half` cannot repair an already wrapped tolerance. This is inherited from the shared helper, but the new entry point exposes it without validation.

5. **Major — mass products are formed before the division guard is applied.**  
   [formula_enum.rs:2266](src/models/ms2/formula_enum.rs:2266), lines 2290 and 2314 compute `checked_mul` before testing `count > allow`.

   **Concrete input:** a `CH4`-centered query with domain nitrogen cap 255 reaches `N=2`, computes `2*m_N`, then rejects it against the division bound. Section 1.4 explicitly requires that rejected products never be formed.

   **Fix:** evaluate the comparison first and form the product only inside the admitted branch.

   This is a literal specification failure, **not a demonstrated Rust overflow**: validated C/N/O products individually fit `u32`, and additions use saturation. Hydrogen multiplication is guarded by its interval.

6. **Minor — rare-table contents and the 16,384 policy differ from the literal specification.**  
   [formula_enum.rs:1972](src/models/ms2/formula_enum.rs:1972) silently drops allowed tuples whose rare mass exceeds `u32::MAX`; they are excluded from the combination count.

   **Concrete input:** iodine cap 34, other rare caps zero, rare total `[0,34]`, distinct `[0,1]`. `(0,0,0,0,0,34)` satisfies the stated tuple rules but is absent because its mass is `4,314,752,048`.

   **Fix:** settle this representation policy explicitly in section 1.4 and define whether `P_max` counts allowed or representable tuples. Otherwise reject artifacts whose required rows cannot be represented.

   Retained rows are correctly lexicographic, unique, and contain the correct counts, mass and sum. Also, the specification’s unconditional “combination 0 is all zeros” conflicts with positive rare minima; implementation sensibly makes that conditional.

7. **Minor — artifact validation does not check every declared cap or table alignment.**  
   [formula_enum.rs:1805](src/models/ms2/formula_enum.rs:1805) checks domain caps, but never checks caps in either ratio bucket table. At line 1852 it checks heavy presence against heavy caps, but not against DBE-table length.

   **Concrete inputs:** start with valid `CH4` artifacts, then set `max_by_carbon[0][0]=256`; validation succeeds despite “every cap ≤255.” Separately, clear `dbe_by_heavy`; validation still succeeds despite its documented alignment with `heavy_seen`.

   **Fix:** validate both cap tables and require heavy caps, presence and DBE tables to have matching lengths. These omissions do not currently cause count overflow because domain caps still limit enumeration.

8. **Minor — several claimed test guarantees remain untested or depend on the implementation itself.**  
   [ms2_formula_enum.rs:2199](tests/ms2_formula_enum.rs:2199) derives exact-fit budget from the same implementation; budget 1 asserts only `visited<=1` and repeatability. The boundary test at line 2317 claims saturating `hi`, but its protonated precursor yields a parent approximately 1.012 million below `u32::MAX`, while its window half is under 0.1 million: **`hi` does not saturate**.

   **Concrete missed cases:** the saturation configuration in finding 2; the oracle case in finding 3; a valid window containing both `CH2` and `CH4` for one heavy vector; nonzero P/Cl/Br/I domains with independently calculated expected sets.

   **Fix:** assert independently calculated visit counts and first visited vectors; exercise actual `hi` saturation, ratio factors exactly `2^20`, multiple surviving hydrogen counts, and all six rare elements. The rare-domain brute-force tests are independent in structure, but the oracle defect above limits their authority.

The four earlier findings are:

| Earlier finding | Assessment | Evidence |
|---|---|---|
| Fit depends on training order under margin | **Resolved** | `formula_enum.rs:662,670` deterministically chooses the smallest denominator on ties; permutation regression at `tests/ms2_formula_enum.rs:1266`. |
| Oracle has two wrong acceptance paths | **Partly resolved** | Positive minima and zero-carbon DBE fall-through are fixed at test lines 1067 and 1110; the new margin counterexample remains at line 1093. |
| Exhaustive pruning leaves rules unexercised | **Partly resolved** | Extended tests exercise rare pruning, heavy-cap leaf refinement and DBE rejection at lines 1569 and 1707. Their domain at line 1475 fixes P/Cl/Br/I to zero and C below 4; individual ratio-stage rejection coverage is not established. |
| Unseen interior buckets have undocumented policy | **Resolved** | Explicit presence fields at `formula_enum.rs:509,519`, observed-only widening at line 694, rejection at lines 766 and 847; regression at test line 1424. |

For the remaining section 1.4 checks: ascending lane nesting, saturating window endpoints, hydrogen interval, per-candidate residual bound, exact positive/negative comparisons, all ratio predicates and bucket presence are correct. Budget stopping occurs before the excess visit, and exact-fit completion works structurally. Global output traversal produces the specified rank prefix, although it does **not implement or expose clamped lane offsets**, nor exercise separate count/fill modes. Ordinary status handling is correct apart from saturation.

With the **same three exact filters enabled**, I find no set counterexample when neither search is exhausted. Carbon-prefix caps use the appropriate carbon row; heavy-prefix caps use a suffix maximum, followed by the exact leaf-total check. Rare maxima are monotone, minima are checked at completion, zero-carbon candidates still face DBE bounds, and every hydrogen in the interval is considered. Mass-overflow rare tuples cannot join either search. If “same bounds” permits `EnumLimits::unfiltered()`, equality is false: fitting `CH3` admits it in unfiltered `enumerate()`, while the device twin rejects odd parity.

The report’s `enumeration_ratio_device_order` denominator is **every spectrum**, including missing gold, unavailable precision, invalid parent mass and exhausted searches (`examples/ms2_formula_report.rs:192,876`). “Scored recall” means gold membership in the retained bounded output; exhausted spectra can still count as hits. Under lane exhaustion, that output is a prefix of the **visited lane results**, not necessarily of the unrestricted candidate stream. The calculation is consistent, but this distinction should be stated in report metadata.

**Verdict: reject.**

---

# Third review: the kernel-expressible twins and the four kernels (ms2_enum)

Reviewer: codex exec (read-only), 2026-10-04. Verdict: reject; the findings are the follow-up to the integration task.

Static review only: no files modified and no Cargo commands run. **The rewritten bodies match, but the dispatch cap contract and several artifact boundaries remain incorrect.**

1. **Blocker — fill ignores the per-spectrum scored cap, violating single-writer ownership.**  
   `src/tensor/ops/ms2_enum.rs:1611` clamps offsets to `min(meta[6], cap_arg)`, but fill passes only the wrapper cap at **line 1888**. `kernel_lane` never reads `META_SCORED_CAP`.

   **Failing input:** fit bounds from `O2` and `S`; use domain caps O=2, S=1, other heavy counts and hydrogen zero, heavy maximum 2. Query precursor `32_979_347`, adduct 1, ppm=1000, uncertainty=100,000. Both formulas join, in different rare lanes. Set `meta[6]=0`, wrapper cap=2, M=2. Offsets are `[0,0]`, scored=0, but both fill lanes write slot 0. Pad subsequently writes both slots. Thus fill has competing writers, and fill/pad ownership overlaps. With `meta[6]=1`, fill still writes slot 1 despite scored=1.

   **Fix:** compute the identical effective cap in offsets and fill, including `meta[6]` and M; reject inconsistent dispatch capacities. Add poisoned fill-only and pad-only tests with differing row caps.

2. **Major — validated DBE endpoints can overflow host packing or incorrectly reject candidates on the kernel.**  
   `src/models/ms2/formula_enum.rs:1934` adds the bias before clamping. Validation at **line 1946** imposes no DBE endpoint restriction. The identical kernel/twin comparison at `src/tensor/ops/ms2_enum.rs:969` / `formula_enum.rs:3273` treats an overflowing upper comparison as rejection.

   **Failing input:** start with valid CH4 artifacts and change its DBE interval to `[0, 2_147_483_647]`. Validation succeeds. CH4 has positive=4, negative=4, twice-DBE=0; it belongs in that interval. Packing makes the upper endpoint `u32::MAX`, so `neg <= max_u32 - shi` fails and CH4 is rejected.

   Set the upper endpoint to `i64::MAX` instead: validation still succeeds, but biasing overflows signed host arithmetic—panic with overflow checks, wrapping otherwise.

   **Fix:** validate or safely clamp endpoints before addition. Compare the biased nonnegative difference after establishing `pos >= neg`, or handle overflowing comparison sides according to their mathematical ordering. Test both examples.

3. **Major — the rare-table limit is checked before excluding unrepresentable tuples.**  
   `src/models/ms2/formula_enum.rs:2124` rejects when 16,384 rows have already been retained, before **line 2138** excludes the next tuple’s excessive mass.

   **Failing input:** rare caps `(F,P,S,Cl,Br,I)=(201,13,0,0,0,5)`, rare total `[0,219]`, distinct `[0,3]`; other domain caps zero. Exactly **16,384** tuples have representable mass. After retaining the last representable tuple, `(201,13,0,0,0,1)` reaches the limit check with mass **4,348,242,381**. It should be excluded, but instead validation returns `Error::Config`.

   **Fix:** compute mass and discard unrepresentable tuples before checking whether another retained row exceeds P_max. Add this exact-boundary regression.

4. **Major — fill never takes the specified capacity exits.**  
   `src/tensor/ops/ms2_enum.rs:1480` / `src/models/ms2/formula_enum.rs:3829` merely disable a write when rank reaches cap. Neither exits a lane whose offset is already at cap nor stops at the first excluded joined rank.

   **Failing input:** the O2/S example above with consistent cap=1. The S lane starts at offset=1 but still enumerates. With cap=0, every fill lane repeats the full count traversal despite owning no output slots.

   This does not change candidate decisions with consistent caps, but violates §1.4’s explicit fill cutoffs and performs unnecessary search work.

   **Fix:** return immediately in fill mode when offset≥effective cap; stop traversal once its next joined rank reaches that cap. Preserve count mode’s full counters.

5. **Major — dispatch does not enforce the lane ceiling; standalone fill also permits lane-address wrapping.**  
   `src/tensor/ops/ms2_enum.rs:1960` and **2147** check multiplication against `usize`, but do not enforce `enum_lanes_max`. Fill also omits a `u32` length check for offsets.

   **Failing input for the dispatch limit:** rare caps F=31, P=15, S=31, other rare caps zero, ranges admitting all tuples. All **16,384** tuples fit mass. B=17 produces **278,528** lanes, exceeding the default 262,144; the wrappers accept it.

   **Concrete wrap in standalone fill:** B=262,145, P=16,384, M=1, correctly shaped buffers. Meta, rare and cand lengths fit `u32`, so fill’s checks pass. At b=262,144, `b * P` at **line 1884** is `4,294,967,296`, wrapping to zero and reading another spectrum’s offsets. Count would reject its oversized stats buffer, so this is an independently callable fill defect, not a successful count→fill pipeline case.

   **Fix:** enforce the configured lane ceiling before launching and validate offsets length/lane-index arithmetic as `u32`.

6. **Minor — ppm validation is bypassed by early status exits.**  
   `src/models/ms2/formula_enum.rs:4161` and **4178** return before `tolerance_u32` validates ppm at **4198**.

   **Failing input:** valid CH4 artifacts, ppm=1001, uncertainty=`u32::MAX`. `enumerate_device_order` returns an unknown-precision result rather than `Error::Config`. An invalid parent similarly bypasses ppm validation. `enumerate` validates ppm first at **line 1522**.

   **Fix:** validate ppm immediately upon entry, before status-dependent returns.

7. **Minor — coverage claims exceed what the tests establish.**  
   `tests/ms2_enum_kernels.rs:946` uses M=1, scored=1. Its “every later slot” ownership assertions have no later slots to inspect. The failed-row test at **1126** substitutes budget zero; it never supplies or checks `peak_count==0`.

   **Concrete missing inputs:** findings 1–3 above; synthetic saturation stats from `tests/ms2_formula_enum.rs:2663`; and a CH2/CH4 window with parent `15_023_475`, adduct 1, ppm=1000, uncertainty=1,007,000, bounds fitted from those two formulas. Both hydrogen counts should survive as ambiguous candidates. The new adjacent-hydrogen test at **3285** instead demonstrates that parity removes CH3.

   **Fix:** use M>scored in isolated authorship tests, vary meta caps independently, launch offsets directly on saturation fixtures, and add the two-surviving-hydrogen case. Test the actual failed-spectrum metadata preparation. Exercise ratio factors exactly `2^20` through enumeration, rather than validation alone.

The eight findings from the **Second review** assess as follows:

| Earlier finding | Assessment | Current evidence / remaining failure |
|---|---|---|
| 1. Not a portable u32 twin | **Resolved** | `formula_enum.rs:3336`; portable helpers use u32 flags/arithmetic. Host allocation and conversion are outside the lane. |
| 2. Saturation misses exhaustion | **Resolved** | `formula_enum.rs:4015`, `ms2_enum.rs:1642`: saturation is retained and contributes to exhaustion; host regression at `tests/ms2_formula_enum.rs:2658`. |
| 3. No-carbon margin oracle | **Resolved** | `tests/ms2_formula_enum.rs:1098`, regression at **2779** applies margin to default fractions. |
| 4. Oversized tolerance wraps | **Partly resolved** | `formula_enum.rs:4198` uses checked-range tolerance; regression at test **2829**. No original tolerance wrap remains, but early exits bypass the required ppm rejection: finding 6. |
| 5. Product precedes division guard | **Resolved** | `ms2_enum.rs:1106`, **1137**, **1376**; corresponding host code at `formula_enum.rs:3455` onward. Products occur only after admission. |
| 6. Rare representation/P policy | **Partly resolved** | §1.4 now explicitly excludes excessive-mass tuples and counts retained rows; `formula_enum.rs:2138` implements exclusion. The exact P_max boundary still fails: finding 3. |
| 7. Missing cap/alignment checks | **Resolved** | `formula_enum.rs:2001`–**2028** validates alignment and both cap tables; regressions at test **2973**. DBE endpoint safety is a separate remaining defect. |
| 8. Weak boundary tests | **Partly resolved** | Independent visits at test **3099**, actual hi saturation at **3221**, additional rare-element cases at **3340**. Remaining concrete gaps are in finding 7. |

For the statement-by-statement comparison, **all 22 corresponding bodies are identical** after removing comments and normalizing buffer-index casts: `decide_u32`, both saturation helpers, read/write/span/address helpers, mass-admission and total helpers, all three exact helpers, verdict, buckets, ratios, DBE, lane, offsets and pad. There are **no differing guards, operand orders, branches, constants, packed offsets or base/stride expressions** between those bodies. Buffer parameter types and CubeCL annotations differ. Findings 1–4 affect shared behavior or dispatch, so parity alone cannot detect them.

Arithmetic and addressing otherwise check out under validated artifacts and a consistent dispatch:

- Mass products follow division guards; additions remain ≤hi. Hydrogen endpoints use guarded differences and quotient/remainder ceiling.
- The decision helper implements `r+E≤tol` and `r>tol+E` through subtraction comparisons without forming overflowing sums.
- Positive DBE totals are ≤2,552; negative totals ≤2,043. Ratio numerators are ≤1,023, so multiplication by `2^20` is ≤1,072,693,248.
- A lane has at most `256³` heavy vectors and four hydrogen values per vector; its joined increment cannot wrap. Aggregate saturation and clamped prefixes are correct.
- Carbon and heavy bucket indices can exceed table rows. `k_buckets` and `k_dbe` mark those cases invalid but still attempt address reads. `k_read` at `ms2_enum.rs:180` protects the actual array load with its full-buffer length; invalid indices can read another packed section or return zero, but cannot re-enable acceptance.
- Count maps `pos` to `(pos/P,pos%P)` and owns one stats pair; offsets owns one spectrum’s prefixes and counters; pad owns one slot. Checked total buffer lengths protect normal pipeline `B*P*2` and `B*M*13` arithmetic. **The cap mismatch in finding 1 defeats fill ownership.**
- With consistent caps, joined<cap, joined==cap and joined>cap all have complete fill/pad coverage. Count and fill make identical join decisions in identical order, including ambiguity and budget exhaustion. Fill’s missing early termination changes work, not its candidate stream.

Status handling is correct for ordinary completed absence, lane exhaustion, scored truncation and aggregate saturation. Host setup correctly returns no candidates for mass overflow, unknown precision and half>1,511,737; unknown precision adds exact-mass-unavailable and formula-absent. Those exits are **host-only tests**, not kernel status tests. The kernel meta layout has no peak-count field or failed-row flag, so these files do not establish handling of actual failed request rows.

For the stated CubeCL pitfalls: I found **no loop-carried accumulator initialized directly from a scalar kernel argument**. The hydrogen loop starts from a computed meta-derived endpoint; bucket flags start from computed locals. There are **no `&&`/`||` division guards** in the copied bodies. Branch-dependent loads remain exposed at `ms2_enum.rs:183` (`k_read`), **774** (heavy caps), and **856** (ratio words). Source identity does not prove backend lowering; these paths need execution on both required backends.

Finally, **yes**, `tests/ms2_enum_kernels.rs:258` poisons stats, offsets, counters and candidates, launches the kernels, and calls `check_launches`. `check_row` at **291** compares every stats, offsets, counters and candidate element with the twins. It uses `Auto`, so the file itself does not prove that both CPU and GPU were exercised. No execution claim is made by this review.

**Verdict: reject.**