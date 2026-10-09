# Codex review: functional-group detector, vocabulary ms2-fg-v4 (validated matchings, prescribed double-bond degrees)

Reviewer: codex exec (read-only), 2026-10-06. Verdict: both high findings of the v3 review are repaired; no counterexample found to kekulé invariance on closed molecules (connectivity, hydrogen counts, valences and triple bonds fixed) or to fragment soundness. One medium finding remains in the reference tool (hand-built graphs bypass valence validation), plus a defensive-contract split, wording restrictions and two test-coverage notes. Fix task: FG6.

The two previous high findings are repaired. I found no counterexample to closed-molecule kekulé invariance or fragment soundness under the stated fixed-triple definition. One reference-validation finding remains.

I modified no files and ran no cargo. Checks used Python translations of the Rust matching, gadget, and determined-type rules, plus cached RDKit. The statistics were outside this review.

1. **Medium — hand-built reference graphs bypass structural and valence validation.**  
   Location: [functional_groups_ref.py:1050](tools/ms2/functional_groups_ref.py:1050), with permissive molecule construction at [line 635](tools/ms2/functional_groups_ref.py:635).

   Concrete whole graph: atom `0=O(H0,v2)`, atoms `1,2,3=C(H3,v4)`, and singles `(0,1),(0,2),(0,3)`. Oxygen carries bond-order sum three despite its declared valence two.

   **Expected:** reject the graph before enumeration/counting, as `MolGraph::new` does.  
   **Actual:** `process_graph` returns a successful record, no skip/error, and **three ether instances**. It also accepts a lone declared `C(H3,v4)` with no bonds as a whole molecule, despite its unsatisfied valence.

   **Minimal fix:** validate endpoints, self/duplicate bonds, supported orders, connectivity, and `H + incident order sum == declared valence` before enumerating whole-molecule forms. Return an error for malformed stored graphs. All 800 current fixture forms passed my closure check, so this does not invalidate their present results.

2. **Disposition of the previous counterexamples**

   - **`O=S1(C)=NC=CC=C1`: repaired.** Both assignments are enumerated. The translated gadget reports all six ring bonds delocalised and the exocyclic bonds fixed. Both forms have zero alkene and zero imine.
   - **Eleven-atom fused S/P parent, fragment on original atoms `0…8`: repaired.** The disputed `(0,1)` C=N bond is undecided in the fragment; the hydroxyl is undetermined. The current parent has three valid assignments and contains hydroxyl in all three. Thus the old parent-side omission is repaired too.
   - **Twelve-vertex blossom counterexample: repaired.** The translated matcher returns a valid maximum matching of size five, agreeing with brute force; no perfect matching exists.

3. **Edmonds and validation**

   The bookkeeping follows the standard formulation:

   - `parent`, `base`, `used`, and blossom marks are fresh for each root search.
   - Both blossom arms are marked, carrying the opposite closing-edge endpoint as the initial child.
   - Contraction updates every vertex whose current base belongs to the blossom and queues previously unused vertices.
   - LCA walks use base representatives and parent links through matched vertices.
   - Starting from the retained matching after deleting a matched edge, or deleting two vertices and freeing their partners, preserves the required matching preconditions.

   My independent brute-force comparisons passed on **6,000 random graphs** and **59,463 constrained searches**, including searches with repeated contractions. Every returned matching was valid and had the expected cardinality.

   Successful alternative-match evidence is checked against the **searched adjacency**, covers the intended vertex set, and satisfies the tested-edge constraint. Forced-edge results are additionally lifted and validated on the full gadget. The symmetric-difference witness checks are sufficient given two validated perfect matchings.

   One defensive-contract qualification: [line 1222](src/models/ms2/functional_groups.rs:1222) and the corresponding single-edge branch conflate an **invalid** returned matching with a valid nonperfect result, potentially returning fixed. They do not implement the documented “invalid matching → loud assertion/Unknown” behavior. I found no graph causing the rewritten matcher to return an invalid matching, so this is not a reproduced detector failure. Split those cases to make the documented protection accurate.

4. **Tutte gadget and the domain definition**

   The correspondence is exact. At atom `v`, its `k−d(v)` cores consume precisely that many ports; the remaining `d(v)` ports must match outward. Conversely, every prescribed-degree assignment can match its single-bond ports to the cores.

   - `d=k` correctly gives no cores and forces every candidate incident edge double.
   - `d>k` cannot arise from a valid stored assignment.
   - A single with a `d=0` endpoint cannot become double while preserving that endpoint’s double count.
   - Triple-bearing atoms retain their eligible single/double edges; they are not discarded wholesale.

   I checked **`C#S1=NC=CC=C1`**: RDKit admits it within the atom vocabulary, the triple is fixed, and all six ring bonds are delocalised. Therefore the documentation’s blanket statement that triples’ incident singles are fixed by valence should be removed.

   With connectivity, H, valence, and triple bonds fixed,

   `d(v) = valence(v) − H(v) − heavy_degree(v) − 2·triple_count(v)`.

   Thus there is no additional single/double rearrangement that preserves those quantities while changing an atom’s double count.

   The triple restriction matters. The admitted neutral graphs **`C1#CC=C1`** and **`C1=C=CC=1`** have identical connectivity, H counts, and valences, but their counts are respectively `{alkene:1, alkyne:1}` and `{alkene:3}`. They are outside the module’s equivalence relation.

   Also describe projected witnesses as **alternating closed trails**, allowing repeated original atoms. Two triangles sharing `S(H0,v6)` can exchange their double assignments without any simple even atom-cycle. The gadget handles this correctly. “Delocalised” remains a combinatorial label, including nonaromatic cases such as cyclobutadiene.

5. **Fragment rule and tests**

   Residual one cannot accommodate an additional stored double: an outside double consumes two valence units. A residual-one atom with zero inside doubles therefore remains `d=0` in an order-preserving completion and contributes no gadget vertices.

   Stable witnesses preserve the inside degree assignment and extend into the parent. The sealed-end search conservatively covers alternating walks to any open boundary, including hypervalent and cumulene cases. I found no determined type absent from its parent.

   Checks passed for:

   - **170 fixture molecules / 800 forms:** translated bond statuses and reference counts agree.
   - **17,083 connected fragments / 33,154 decided bonds:** parent agreement and determined-type inclusion hold.
   - **500 additional random closed graphs:** including 105 movable `d=2` atoms.
   - **53,952 fragment permutation comparisons:** no disagreement.

   The Rust test oracle is independent of library helpers and uses assignment backtracking rather than gadget matching.

   Coverage needs precise wording. `random_ring_d2_parent` guarantees a `d=2` atom on a topological cycle, **not** an alternating witness through that atom. Add an assertion that some generated `d=2` atoms actually have variable incident bonds.

   The exhaustive fragment test uses only the first stored form and samples above 2,000 subsets. The S/P fixture’s first form has the disputed C–N bond **single**; the old counterexample requires it double. The dedicated regression selects that form and would catch the exact old witness defect. The exhaustive test alone does not guarantee that reproduction, nor do odd-cycle assertions prove nested contraction occurred.

6. **Python reference**

   Its assignment backtracking is methodologically independent of Edmonds/Tutte, and cap overflow is loud rather than silently truncated. Its remaining weakness is the hand-built input validation described above.

**Verdict**

- **(a) Closed-molecule kekulé invariance:** holds as far as these attacks establish, for valid closed V0 atom-type graphs with connectivity, H counts, valences, and triple bonds fixed.
- **(b) Fragment soundness:** holds as far as these attacks establish: determined types are a subset of parent types, and decided bond statuses agree with the parent.

Documentation should require an order-preserving induced fragment of a valid closed parent, state that decidedness is sufficient and conservative, retain the fixed-triple restriction, and avoid extending these claims to tautomers, arbitrary valence-preserving triple rearrangements, or chemical resonance generally. These are translation-based checks, not execution of the Rust tests.