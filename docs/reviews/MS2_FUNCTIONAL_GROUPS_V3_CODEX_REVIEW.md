# Codex review: functional-group evaluation, vocabulary ms2-fg-v3 (exact delocalisation, paired donor peaks)

Reviewer: codex exec (read-only, on a snapshot of the tree), 2026-10-06. Verdict: the unrestricted invariance and fragment-soundness claims are rejected — the pi graph drops ring atoms with two double bonds (hypervalent S, P), so some valence-preserving forms are missed, and the blossom path reconstruction can return a path with a nonexistent edge, which lets a fragment report a type its parent lacks (constructed counterexample); the diagnostics, the reference error handling and the paired statistics are accepted. Fix task: FG5.

**Two high-severity findings remain: the π graph excludes valid bond-order rearrangements involving hypervalent atoms, and blossom path reconstruction can make a fragment report a type absent from its parent.**

I modified no files and ran no cargo. Checks used a Python translation of the Rust matching/status code and cached RDKit. All **159 fixture molecules / 785 stored forms** matched the reference. Another **5,500 sampled connected fragments / 14,975 decided bonds** from conventional ring cases showed no disagreement. These are Python checks, not executions of the Rust tests.

1. **High — perfect matchings of this H do not cover all admitted valence-preserving forms.**

   Locations: [functional_groups.rs:860](src/models/ms2/functional_groups.rs:860), explanatory claim at [line 85](src/models/ms2/functional_groups.rs:85); the reference repeats the restriction at [functional_groups_ref.py:179](tools/ms2/functional_groups_ref.py:179).

   Concrete neutral molecule: **`O=S1(C)=NC=CC=C1`**. RDKit accepts it with zero formal charges and radicals, and every atom fits the V0 vocabulary.

   Number atoms as follows: `0=O`, `1=S(H0,v6)`, `2=C(H3)`, `3=N(H0,v3)`, and `4…7=C(H1)`. Bonds common to both forms are `0=1` and `1–2`.

   ```text
   Form A ring: 1=3–4=5–6=7–1
   Form B ring: 1–3=4–5=6–7=1
   ```

   The forms preserve connectivity, hydrogens, valence and each atom’s double-bond count. They differ by flipping one alternating six-cycle.

   **Expected under the module’s stated alternating-cycle definition:** all six ring bonds are delocalised; neither form contributes an alkene or imine.

   **Actual:** sulfur is excluded because it has two doubles, and its ring double partner is pruned. Both forms have an empty delocalised set:

   | | Alkene | Imine |
   |---|---:|---:|
   | Form A | 2 | 0 |
   | Form B | 1 | 1 |

   Independent enumeration of all single/double assignments satisfying the atom valences finds **two forms**. The reference generates **one** from either input. Complete enumeration of an incorrectly restricted H therefore cannot expose this failure.

   **Minimal fix:** support prescribed double-bond degrees through a general factor/b-matching formulation, including vertices requiring two doubles. Alternatively, explicitly reject these cyclic hypervalent graphs and narrow the invariance claim. Merely pruning their partners is insufficient.

   The pruning is justified for ordinary carbon cumulene centres and terminal nitriles. Likewise, terminal oxygen pins ordinary sulfonyl, phosphoryl and exocyclic carbonyl doubles. But the explanation that *every* two-double or triple-bearing atom is pinned by valence is false for the admitted hypervalent domain.

2. **High — blossom reconstruction creates nonexistent edges, bypassing the stable-witness condition.**

   Locations: [functional_groups.rs:1221](src/models/ms2/functional_groups.rs:1221), contraction calls at [line 1256](src/models/ms2/functional_groups.rs:1256), witness acceptance at [line 1005](src/models/ms2/functional_groups.rs:1005).

   `mark_path` assigns `parent[v] = parent[m]`. Blossom lifting instead needs the vertex across the closing edge, followed by the appropriate child while walking toward the base. The current assignment can reconstruct an “augmenting path” containing a nonexistent edge.

   **Concrete parent:** these atom-type IDs, in atom order:

   ```text
   [1, 5, 16, 1, 15, 1, 1, 1, 9, 8, 4]
   ```

   Thus `0=C(H0)`, `1=N(H0,v3)`, `2=P(H0,v5)`, `4=S(H0,v6)`, `8=OH`, `9=O(H0)`, `10=CH3`; atoms `3,5,6,7` are C(H0).

   ```text
   Doubles: (0,1), (2,3), (4,5), (6,7), (4,9)
   Singles: (0,3), (0,8), (1,7), (2,4), (2,5),
            (2,7), (3,6), (4,10), (5,6)
   ```

   This closed, neutral parent is accepted by RDKit; its SMILES is:

   ```text
   CS1(=O)=C2C3=C4N=C(O)C3=P421
   ```

   Take the **connected induced fragment on atoms `0…8`**. Both parent and fragment fit the 16-atom/four-closure limits. Only fragment sulfur `4` has residual valence, namely **3**.

   Testing fragment double `(0,1)` returns the purported path:

   ```text
   [1,7,6,2,3,0]
   ```

   **There is no `(6,2)` edge.** The path omits unstable sulfur `4`, so every listed vertex passes `residual <= 1`. A genuine alternating witness uses sulfur `4`.

   **Expected:** `(0,1)` is undecided in the fragment; consequently hydroxyl `(0,8)` is undetermined.

   **Actual:** the fragment reports `(0,1)` as decided-delocalised and **hydroxyl as determined**. In the parent, sulfur’s additional double removes it from H, `(0,1)` is fixed, and the hydroxyl is excluded by the carbon’s fixed C=N.

   This directly demonstrates:

   ```text
   fragment determined types ⊄ parent types
   ```

   The same reconstruction defect can corrupt the general maximum-matching helper. For the following ordered edge list on 12 vertices, independent enumeration finds no perfect matching, but the translated helper reports one using nonexistent pairs `(0,7)` and `(3,4)`:

   ```text
   (3,9) (0,8) (2,8) (0,9) (4,9) (5,10) (0,1) (7,11)
   (4,8) (3,10) (1,11) (9,11) (2,10) (7,9) (0,3) (6,9)
   ```

   **Minimal fix:** pass the closing-edge child into `mark_path`, update parents using that child, and propagate the matched vertex while walking upward. Validate reconstructed paths for real adjacency, alternation and distinct vertices before using them as stability witnesses. Add the parent/fragment regression above and explicit nested-blossom cases.

   I did **not** find a separate closed-graph allowed-edge Boolean mismatch in the planted near-perfect searches. That does not repair the demonstrated incorrect witness or general matching result.

The four re-review findings have these dispositions:

| Earlier finding | Disposition |
|---|---|
| Bounded alternating-cycle detector | **Bound removed; earlier named cases repaired.** The new unrestricted correctness claim still fails for the reasons above. |
| Bounded reference form generator | **Matching enumeration is complete within H**, with a loud 4,096-matching cap. Its shared H construction retains finding 1’s blind spot. |
| “Uncloseable” diagnostic | **Fixed.** `closing_fragment_not_found` explicitly describes search failure, and staged searches precede layer expansion. |
| Counting errors accepted as skips | **Fixed.** Counting exceptions become errors and cause failure; the known validation export requires zero skips and exact molecule keys. |

The planted-graph brute force at [ms2_functional_groups.rs:1386](tests/ms2_functional_groups.rs:1386) is independent of blossom: it recursively enumerates disjoint vertex pairs, then counts edge membership across matchings. Its weakness is coverage, not circularity. The comparison checks Booleans rather than reconstructed witness validity, and supplies sorted edges.

For the fragment rule, the **stable-cycle condition is sound when applied to a genuine alternating cycle**: residual ≤1 prevents a witness vertex from gaining a second double or triple outside. The sealed-end search is conservative; accepting any reached open valence can introduce uncertainty unnecessarily, but I found no decided-fixed counterexample. Conventional fused cuts, pyrrole-type five-rings, quinones and tropone produced no additional bond disagreement in my sampled checks. The blossom defect nevertheless defeats the implemented soundness guarantee and can enlarge the determined type set.

The donor comparison is wired as follows:

- [experiment.rs:104](src/models/ms2/experiment.rs:104) guarantees a different-molecule donor by rejection sampling across **the whole validation spectrum set**. There is no precursor bucket, mass matching or collision-energy matching. Selection is deterministic under the **checkpoint’s training seed**, rather than the evaluator’s `--seed`.
- [experiment.rs:503](src/models/ms2/experiment.rs:503) substitutes peak IDs, m/z, intensities, raw/actual peak counts and **peak m/z uncertainty**. Recipient precursor, adduct, polarity, collision energy and instrument remain. `intensity_scale` is the export path’s common value `0`.
- Donor peaks undergo the recipient’s precursor eligibility filter; peaks above precursor +2 Da can disappear. Evidence features and encoder inputs are recomputed from the resulting donor peaks.
- The precursor formula **candidate pool/enumeration remains unchanged**: its inputs are recipient precursor/adduct/tolerances. Formula scores, retained top formulas, trajectory allocation and generated structures may change because scoring uses the changed encoder/evidence inputs. Holding these downstream choices fixed would measure a different intervention.
- [functional_groups_eval.rs:935](src/models/ms2/functional_groups_eval.rs:935) resamples recipient molecules and includes all their spectra, with **identical resamples for own and donor**. Bounds are percentiles of replicate **own-minus-donor differences**, not subtraction of marginal confidence limits.

The positive intervals support a **nominal, conditional aggregate effect**. A lower bound of +0.001 is not automatically invalid, but it is weak evidence near the boundary: the count-model k=8 micro-F1 lower bound is actually **0.00081067**, estimated from 1,000 replicates. These intervals condition on one checkpoint, donor assignment and generation seed. The stronger aggregate evidence-model F1 intervals support dependence; they do not measure what fraction of correct predictions “comes from” peaks.

Per-type intervals are implemented correctly under the stated denominator convention. The four positive recall-difference intervals are confirmed:

| Type | Point | 95% interval |
|---|---:|---:|
| Ester | 0.09554 | [0.01388, 0.18938] |
| Ketone | 0.12295 | [0.01438, 0.23389] |
| Hydroxyl | 0.06780 | [0.00467, 0.13170] |
| Fluoride | 0.12925 | [0.00699, 0.24744] |

The reading’s multiplicity caveat is appropriate: these are unadjusted indications. Strictly, thiol has no truth support and no defined recall test, so “28 types tested” describes the vocabulary rather than 28 defined tests. A zero-crossing interval does not establish independence or equality.

I found **no stale active v2 vocabulary or fixture reference** in `src`, `examples`, `tools` or `tests`. The v3 fixture and four v3 reports identify `ms2-fg-v3`. The old review documents and archived v2 reports are intentional history.

**Verdict: reject the unrestricted v3 invariance and fragment-soundness claims; accept the repaired diagnostics, reference error handling and paired statistical implementation.** The stored pilot numbers remain descriptive results, but fixture agreement does not validate the universal claims.

Sentences in “Functional-group evaluation” that the numbers do not support:

- [Line 1558](docs/MS2_SUBSTRUCTURE_TASKS.md:1558): “the result does not depend on the kekulé form.” The hypervalent ring counterexample contradicts the unrestricted claim.
- [Line 1562](docs/MS2_SUBSTRUCTURE_TASKS.md:1562): “a fragment of the true parent is never credited with a type the parent lacks.” The hydroxyl counterexample contradicts this.
- [Line 1580](docs/MS2_SUBSTRUCTURE_TASKS.md:1580): “so the rule itself does not cap recall.” Zero search failures establishes witnesses for the evaluated parents; it is not a universal statement about the model’s reachable fragment family.
- [Lines 1675–1677, beginning here](docs/MS2_SUBSTRUCTURE_TASKS.md:1675): “about a twentieth of the score” is a ratio of F1 differences, not an attribution; “most of what it gets right comes from the precursor mass, the formula and the training distribution” requires further controls.
- [Line 1685](docs/MS2_SUBSTRUCTURE_TASKS.md:1685): carboxylic-acid recall is “unchanged.” Its estimated difference is approximately +0.01 with interval [−0.15,+0.16]; that establishes neither equality nor equivalence.
- [Line 1695](docs/MS2_SUBSTRUCTURE_TASKS.md:1695): “the vocabulary is kekulé-invariant.” It needs a restricted-domain qualification or the detector fix above.