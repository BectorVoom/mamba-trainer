# Codex re-review: functional-group evaluation, vocabulary ms2-fg-v2 after the numbering fix

Reviewer: codex exec (read-only), 2026-10-06. Verdict: the unrestricted kekulé-invariance claim is rejected (the alternating-cycle search is bounded at 22 atoms and excludes four-cycles: hexacene and an expanded porphyrin-like macrocycle are counterexamples; the reference generator shares the bound); most evaluation fixes accepted; the pilot numbers stand as measured for the stored representations; several sentences of the reading in the tasks document overstate what the tables support. Fix task: FG4.

The fixes resolve most of the earlier review, including the five-ring numbering defect. **Kekulé invariance remains broken for larger, valid fused systems**, and the reference enumeration shares the detector’s blind spot.

I modified no files and ran no cargo commands. Read-only checks used cached RDKit: all **221 stored fixture forms matched their reference counts**. An independent bond-assignment enumeration exposed the counterexamples below. A Python translation of the relevant Rust fragment rules checked **868 connected induced fragments without a soundness counterexample**; this is not execution of the Rust tests.

**Remaining findings**

1. **High — the 22-cycle bound breaks Kekulé invariance.**  
   [functional_groups.rs:673](src/models/ms2/functional_groups.rs:673), closure test at [line 903](src/models/ms2/functional_groups.rs:903).

   **Concrete molecule: hexacene, C₂₆H₁₆.** Number its two rows `0…12` and `13…25`, with horizontal edges between consecutive row atoms and rungs `(2j,13+2j)` for `j=0…6`.

   Form A has doubles:
   ```
   (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,25)
   (13,14) (15,16) (17,18) (19,20) (21,22) (23,24)
   ```
   Form B replaces `(10,11),(12,25),(23,24)` with `(10,23),(11,12),(24,25)`.

   | | Expected | Actual |
   |---|---|---|
   | Form A | `arene_ring=6`, `alkene=0` | `arene_ring=5`, `alkene=2` |
   | Form B | Same counts | `arene_ring=6`, `alkene=0` |

   These assignments differ by flipping one alternating six-cycle. Some other bonds then require a **26-cycle witness**. Thus the documentation’s assertion that every bond in a larger fused sheet has a smaller sufficient witness is false—even below the stated 69-heavy-atom pilot maximum.

   An expanded porphyrin-like example also fails: a six-pyrrole macrocycle with four imine nitrogens, one pyrrolic NH and one pyrrolic N-methyl nitrogen has two assignments yielding respectively:
   ```
   tertiary_amine=1, imine=4, alkene=11, heteroaromatic_five_ring=1
   secondary_amine=1, imine=4, alkene=11, heteroaromatic_five_ring=1
   ```
   Searching through the complete graph instead gives `heteroaromatic_five_ring=6` and none of those amine/imine/alkene types.

   The lower bound is incomplete too: **biphenylene’s delocalised bond set changes across assignments** because four-cycles are excluded. Its 28 type counts nevertheless remained identical in my five-form check.

   **Minimal fix:** determine alternating-cycle membership without the 22-cycle truncation, and include four-cycles under the stated mathematical definition. Increasing the constant merely moves the failure boundary. Add hexacene and the substituted expanded macrocycle as regressions.

   **Work-limit behavior:** at 512,000 visits, previously found delocalised bonds remain decided, while other single/double bonds become unknown—including previously classified fixed bonds ([line 839](src/models/ms2/functional_groups.rs:839)). This suppresses instances; retaining only witnesses found before exhaustion can also make results depend on atom traversal order. The ordinary detector/evaluator does not surface `limited`. I did not reproduce exhaustion on the named small molecules. A limit hit should be exposed and invalidate the affected evaluation, rather than silently produce ordinary labels.

2. **Medium — the Kekulé regression generator cannot detect that blind spot reliably.**  
   [functional_groups_ref.py:542](tools/ms2/functional_groups_ref.py:542), bounded cycle enumeration at [line 614](tools/ms2/functional_groups_ref.py:614).

   **Scenario:** the expanded macrocycle above has two valid, hydrogen- and valence-preserving assignments connected by a cycle longer than 22. The fixture generator explores flips only on cycles of length 6–22, with additional caps of 16 forms and 64 cycles. It can omit the alternate assignment entirely.

   **Expected:** invariance checks exercise both assignments.  
   **Actual:** “zero mismatches” can mean the relevant assignment was never generated. The 221-form agreement is valid evidence for those stored forms, not a completeness proof.

   **Minimal fix:** add an independent assignment enumerator for small regression graphs, or commit explicitly constructed alternate assignments outside the detector’s search bounds. Report truncation explicitly.

3. **Medium, diagnostic — `rule_uncloseable_instances` can report a closable instance as uncloseable.**  
   [functional_groups_eval.rs:1352](src/models/ms2/functional_groups_eval.rs:1352), rejection at [line 1370](src/models/ms2/functional_groups_eval.rs:1370).

   **Concrete parent:**
   ```
   n1cccc(C(C)(C)C)c1Oc1nc(C(C)(C)C)c(C(C)(C)C)c(C(C)(C)C)c1C(C)(C)C
   ```
   RDKit numbering gives the ether anchor `[9,10,11]`. Whole-layer expansion visits sizes **3 → 7 → 13 → 23**, so `closable` returns false before testing the completed rings.

   Yet the connected induced fragment
   ```
   [0,1,2,3,4,5,9,10,11,12,13,18,23,28,29]
   ```
   contains 15 atoms and determines that same ether: both pyridine rings are complete, the relevant C=N bonds are delocalised, and every atom within distance two of the oxygen is closed.

   **Expected:** this ether contributes zero uncloseable instances.  
   **Actual:** it contributes one.

   **Minimal fix:** call this count “instances without a witness found by layer expansion,” or search selectively/exhaustively before declaring impossibility. The current **zero** on the pilot remains useful: successful searches really do construct fragments and run `fg_instances`; it is negative results that overclaim.

4. **Medium — the live reference test still permits detector failures to become accepted skips.**  
   [functional_groups_ref.py:877](tools/ms2/functional_groups_ref.py:877), acceptance checks at [ms2_functional_groups.rs:841](tests/ms2_functional_groups.rs:841).

   **Scenario:** one ordinary ethanol record succeeds, while a reference regression raises exceptions on the remaining supported molecules. `export_molecule_record` converts those exceptions into skips. The test accepts the result because `compared + skipped == export_count`, `compared > 0`, and the compared molecule agrees.

   **Expected:** counting failures on supported exported graphs fail the regression.  
   **Actual:** extensive failures can pass.

   **Minimal fix:** distinguish unsupported-input skips from counting exceptions; fail on the latter. For the known supported validation export, require zero skips and matching molecule keys.

**Disposition of the ten earlier findings**

| Earlier finding | Disposition |
|---|---|
| 1. Kekulé-dependent labels | **Partially fixed.** Small-system counterexamples are repaired; hexacene and expanded macrocycles still fail. |
| 2. Reference counts describe a different stored graph | **Fixed.** Export counting preserves supplied orders without sanitizing/re-Kekulizing. |
| 3. Recipe oracle presented as model ceiling | **Fixed in code/JSON naming; still overstated in the tasks document.** It now explicitly measures recipe coverage. |
| 4. Actual candidate sizes differ from stated 3–16 family | **Fixed by disclosure and additional metrics.** Evaluation includes one-atom candidates and separately reports `min_atoms_3`. |
| 5. Misleading chemical classifications | **The cited cases are fixed.** Formaldehyde counts as aldehyde; pyrrole/N-methylpyrrole are excluded from conventional amines; carbonic-acid centers fail the remaining-C/H condition. |
| 6. Macro bootstrap changes supported types | **Fixed.** Original `types_used` is held across replicates; undefined held components contribute zero. |
| 7. Prior omits empty set | **Fixed.** `tau=2` includes it; all-empty training masks select it. |
| 8. Formula prior uses best combined candidate score | **Fixed with a narrower source definition.** It uses maximum formula log probability **among eligible records**, not necessarily the highest retained formula hypothesis overall. |
| 9. Live reference failure/zero comparison passes | **Partially fixed.** Nonzero process exit and zero comparisons fail; per-molecule counting exceptions can still pass as skips. |
| 10. Impossible acid uncertainty | **Fixed for the cited methyl-acetate core.** O/H0 witnesses ester uncertainty, not acid uncertainty. |

**Other requested checks**

- Independent enumeration found invariant counts for azulene **2 forms**, biphenylene **5**, indole **2**, benzimidazole **2**, purine **2**, acenaphthylene **3**, indene **2**, 2-hydroxypyridine **2**, 2-aminopyridine **2**, and pyridazine **2**. Ordinary four-pyrrole porphyrin had two invariant assignments. Tropone, 2-pyridone and p-benzoquinone each had one assignment under fixed parent hydrogens and per-atom double-bond requirements. Pyridone versus hydroxypyridine remains a tautomer distinction; charged pyridine-N-oxide is outside this neutral atom vocabulary.
- For the requested fragment types, unknown required doubles remain undetermined; complete and possible five-rings supply exclusions; the distance-two boundary check protects incomplete five-rings. Acid/ester/amide require the carbonyl carbon closed, so an omitted remaining substituent cannot be assumed C/H. I found no ≤16-atom counterexample without search exhaustion.
- The corrected five-ring exclusions use actual cycle edges. Sorted anchor sets deduplicate instances; the generic first-match mappings are interchangeable under the current symmetric predicates/exclusions. Six-ring enumeration tests cycle edges before deduplication. I found no additional demonstrated numbering failure below the work limit.
- Null denominators follow the documented convention. Note that this also makes F1 null when precision and recall are both zero, even with nonempty prediction and truth totals.
- `min_atoms_3` is computed and serialized in all four reports. For evidence-real/full/k=8, unrestricted F1 is **0.50581944**, versus **0.50587248** for min-3; recall is unchanged. The current driver prints the row.

**Verdict: reject the unrestricted v2 Kekulé-invariance claim; accept most evaluation-plumbing fixes, subject to the remaining diagnostic/test issues above.** The published pilot numbers remain measured results for their supplied representations.

The reported reading contains these unsupported or overstated claims:

- [Tasks:1515](docs/MS2_SUBSTRUCTURE_TASKS.md:1515), [1616](docs/MS2_SUBSTRUCTURE_TASKS.md:1616): universal Kekulé invariance is contradicted by the examples above.
- [Tasks:1596](docs/MS2_SUBSTRUCTURE_TASKS.md:1596): “reaches the prior’s recall” should be **approximately matches**. The point estimate is lower, 0.5747 versus 0.5823; overlapping intervals do not establish equivalence.
- [Tasks:1602](docs/MS2_SUBSTRUCTURE_TASKS.md:1602): “intervals just apart” is false. Macro F1 overlaps at **0.333016–0.333793**; instance precision overlaps at **0.592154–0.595128**. A paired difference interval is needed to assess improvement.
- [Tasks:1600](docs/MS2_SUBSTRUCTURE_TASKS.md:1600): the listed per-type directions match the table, but “clearly” is unsupported by per-type uncertainty. The reports also compare different trained checkpoints, not solely an input ablation.
- [Tasks:1604](docs/MS2_SUBSTRUCTURE_TASKS.md:1604): the proposed oxygen/nitrogen-loss explanation is not established by type-set scores.
- [Tasks:1606](docs/MS2_SUBSTRUCTURE_TASKS.md:1606): “draws halogens whenever the formula has them” requires formula-conditioned denominators. The table establishes higher shuffled fluoride/chloride recall, not that mechanism or universal behavior.
- [Tasks:1611](docs/MS2_SUBSTRUCTURE_TASKS.md:1611): model recall exceeds label-union recall **in aggregate**; it does not establish that relationship per spectrum. Their marginal intervals overlap.
- [Tasks:1613](docs/MS2_SUBSTRUCTURE_TASKS.md:1613): scaffold results are numerically similar, not identical or proven equivalent.
- [Tasks:1619](docs/MS2_SUBSTRUCTURE_TASKS.md:1619): neither parent-derived row is a ceiling for unrestricted model predictions. They measure label-union and recipe-family coverage.
- The lower micro F1 than the prior, higher macro F1, overprediction counts, and poor sulfonyl/sulfonamide recall **are supported** by the displayed tables.