# Codex review: formula-evidence kernels (V1 §1.6, task E1)

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree after task E1. Verdict: reject pending fixes (2 major, 2 minor, a list of missing boundary tests). Disposition of each finding is recorded in the tasks document.

No Critical findings. **Verdict: reject pending fixes.** This was a read-only review; I did not run cargo or execute the kernels. Numerical counterexamples below are derived from the source.

1. **Major — Kernel 1 reads an empty buffer when `N = 0`.**  
   [src/tensor/ops/ms2_formula_evidence.rs:350](src/tensor/ops/ms2_formula_evidence.rs:350), also line 363; twin: [src/models/ms2/formula_evidence.rs:164](src/models/ms2/formula_evidence.rs:164).

   **Input:** `B=1, N=0, P=1`, empty `kept` and `kept_f`, `meta=[0,0,0,1,100,0,0,0]`, `spec=[50,0]`. The wrapper accepts these shapes. Selection writes a padding row, then both normalization passes unconditionally load `kept_f[0]`.

   **Wrong result:** the host twin panics instead of returning `ev_peaks=[MAX,0,0,0], ev_w=[0]`; the unchecked kernel performs an out-of-bounds read. Its resulting device behavior is not established.

   **Minimal fix:** handle `N=0` outside the normalization loops, explicitly writing every padding row and zero weight. Alternatively, reject this shape before launch and enforce the same precondition in the twin.

2. **Major — Kernel 1’s eligibility omits `E_ion`, contrary to §1.6’s scope rule.**  
   [src/tensor/ops/ms2_formula_evidence.rs:270](src/tensor/ops/ms2_formula_evidence.rs:270); twin: [src/models/ms2/formula_evidence.rs:107](src/models/ms2/formula_evidence.rs:107).

   **Input:** `B=1, N=2, P=1`, adduct 1, fragment ppm-tenths 100, `U=1,007,325`; peaks:

   | Position | m/z | Intensity | `tol + U` |
   |---|---:|---:|---:|
   | 0 | 50,000,000 | 1.0 | 1,007,825 |
   | 1 | 49,000,000 | 0.5 | 1,007,815 |

   For candidate `C1`, `E_ion=ceil(520/1000)=1`. Position 0 therefore has `half_p=1,007,826 > m_H`; position 1 remains in scope.

   **Wrong output:** both implementations select `[0,50,000,549,500,1]` with weight 1. Under the specified exclusion rule, position 0 must be excluded and the selected row is `[1,49,000,549,490,1]`.

   Kernel 2 correctly refuses to explain the selected out-of-scope peak; the defect is selection and the resulting evidence support.

   **Minimal fix:** supply a candidate-derived, safe spectrum-level `E_ion` bound to selection and include it in the saturated half-width. Because `E_ion` depends on the candidate while selection is shared across candidates, the bound’s policy must be explicit. The present omission needs a specification exception if it is intentional.

3. **Minor — Kernel 2 violates the required guarded-subtraction rule.**  
   [src/tensor/ops/ms2_formula_evidence.rs:810](src/tensor/ops/ms2_formula_evidence.rs:810), also lines 151, 154, 797, 820 and 826.

   **Concrete arithmetic counterexample:** candidate `C1`, evidence target `12,000,000`, tolerance 0, `U=0`, `W=1`. At visit 1, `hm=12,000,000`, `E_ion=1`, and `lo_w=11,999,999`. The unconditional `gap=lo_w-hm` produces `4,294,967,295`.

   This is an **intermediate arithmetic violation**, not a demonstrated wrong final verdict: the subsequent gate discards that ceiling, and the reference uses the same wrapping approach.

   **Minimal fix:** initialize the subtraction results from literals and form them only under their ordering guards. Apply the same guarded operations to the twin/reference path. Existing reference behavior does not exempt this new kernel from the requested crate rule.

4. **Minor — Kernel 2 retains a global-buffer load behind a branch in a loop.**  
   [src/tensor/ops/ms2_formula_evidence.rs:883](src/tensor/ops/ms2_formula_evidence.rs:883); twin: [src/models/ms2/formula_evidence.rs:385](src/models/ms2/formula_evidence.rs:385).

   **Scenario:** candidate `C1`, one valid evidence row with target `12,000,000`, tolerance 119, `U=0`, weight 1 and `W=1`. The expected row is `[1,1,1,1]`; the set mask bit executes the conditional `ev_w` load.

   This directly violates the unconditional-load requirement. **No wrong device output is demonstrated by this read-only review**, so this is a kernel-rule finding rather than a claimed numerical failure.

   **Minimal fix:** load each weight unconditionally before the branch, then conditionally add the loaded value. Mirror that operation order in the twin.

The remaining checks support the following conclusions by inspection:

- **Kernel 2:** for valid selector-produced rows and composition counts in the reference’s domain, explanation matches `ion_assign.accepted >= 1`. The `E_ion` calculation, both-adduct hydrogen cap, hydrogen interval, mass-product guards, mixed-radix prefix and verdict agree. Unknown `U` prevents evaluation. Zero-heavy candidates make zero visits. Flag-0 candidates write four zeros. Early exit preserves the explained set, while `complete` remains determined by the full radix-product budget.
- **Kernel 1:** descending intensity, ascending position, strict successor, NaN exclusion and selected-slot-order normalization agree between kernel and twin. Padding is explicit, subject to finding 1.
- **Kernel 3:** count features copy the same lookup values as `count_features`; parent-mass conversion matches the formula-window kernel for both adducts. Tolerance, sign, unknown metadata handling and features 12–15 agree with the twin. The guarded `4w` construction prevents multiplication overflow; its maximum integer cap is three units below `MAX`, which does not change the resulting f32 value. Padding writes all zeros.
- **Kernel rules:** bindings are 6/6/5. Output ownership is disjoint by lane, including padding. I found no loop-carried variable initialized by copying a scalar kernel argument, and no mask-multiplication selection. Mass products follow their division guards.
- **Dispatch:** absolute indexing and contiguous chunks cover every lane once, including the tail. Nonzero budgets are checked before launch. Required buffer lengths and lane counts are checked against the u32 domain.

For `B=16, M=2048, P=32, W=2048`, kernel 2 has **32,768 lanes**, at most **65,536 peak tests** and **196,608 hydrogen-slot iterations per lane**. One unchunked launch can contain **2,147,483,648 peak tests** and **6,442,450,944 hydrogen-slot iterations**.

With dispatch budget `D`, chunk capacity is `L=max(1,floor(D/65,536))`, and launch count is `ceil(32,768/L)`. When `D<65,536`, one lane still runs per launch, so `D` is not a strict upper bound. Total stage launches are that count plus two.

Kernel 1 costs `O(PN+P)` per spectrum—4,096 selection iterations at `N=128`. Kernel 2 costs `O(P+W(9+P))`; kernel 3 has constant work per candidate. No loop is unbounded or quadratic in `M`; selection is quadratic in `N` only if `P` scales with `N`, whereas the wrapper caps `P` at 32.

The tests provide real host-versus-`ion_assign` comparisons and poisoned kernel-versus-twin comparisons; they do **not** merely call a twin twice. However, kernel 2’s oracle is the new twin, which shares `ion_lane_visit` and `lane_visits_u32` with the reference. Missing cases are:

- **Selection:** `N=0`; the `E_ion` scope boundary above; actual target underflow/overflow for both adducts; saturated half-widths; zero selected-intensity sum; an order-sensitive f32 normalization sum.
- **Evidence:** device-side truncated budgets and exact budget boundaries; `half_p=m_H` and `m_H+1`; hydrogen-cap endpoints for both adducts; the three-hydrogen interval boundary; heavy-mass and hydrogen-mass overflow; flag-1 zero-heavy candidates; zero evidence and unknown `U` exercised directly in kernel 2; mask bit 31; early explanation with `complete=0`; radix products reaching/exceeding `2^32`.
- **Features:** productive device residual cases for both adducts—the current fixture has precursor 0; unknown uncertainty/adduct and both parent-overflow cases on device; `w=0`, saturated widths, and large-width clamping; count-table boundary values.
- **Dispatch/outputs:** `B=0`, `M=0`, `P=0`, `P=33`, address-domain rejection, and tail launch-count assertions. Kernel/twin float comparisons use tolerance, not bit equality; the advertised count-feature “bit-equal” assertion uses float equality rather than `to_bits()`.

**Verdict: reject pending the two Major fixes, kernel-rule compliance fixes, and targeted boundary tests.**