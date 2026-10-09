# Codex review: functional-group evaluation (vocabulary ms2-fg-v1, task FG1)

Reviewer: codex exec (read-only), 2026-10-05. Verdict: reject until Kekulé handling, reference graph consistency and the scope of the oracle row are corrected; the replay and set-metric implementation is largely sound and the "determined" rule has no counterexample for fragments that keep the parent atom types and bond orders. The fixes are task FG2 (vocabulary ms2-fg-v2).

The evaluation plumbing is mostly correct, but the aromatic definitions are representation-dependent, and the reported oracle is not a ceiling for the model’s fragment family. These affect interpretation of the model numbers.

I made no changes and ran no Cargo commands. Using an existing cached RDKit environment, I checked **107 fixture graphs × 26 types: zero reference mismatches**. I also reproduced the Kekulé counterexamples below. Rust behavior was assessed from source.

Paths below are relative to the repository.

1. **High: equivalent Kekulé forms receive different labels and scores.**  
   [`functional_groups.rs:995`](src/models/ms2/functional_groups.rs:995), ring pattern at line 412; carbon exclusions at line 428.

   Concrete naphthalene graph: cycles `0–1–2–3–4–5–0` and `4–5–6–7–8–9–4`. Atoms 4 and 5 are C/H0/valence4; the other atoms are C/H1/valence4. All edges are single except these double bonds:

   | Kekulé form | Double bonds | Actual `arene_ring` | Actual `alkene` |
   |---|---|---:|---:|
   | A | 0–1, 2–3, 4–5, 6–7, 8–9 | 2 | 0 |
   | B | 0–1, 2–3, 4–9, 5–6, 7–8 | 1 | 2 |

   Expected: equivalent representations should give the same chemical type set. Actual: a form-A prediction against a form-B parent has precision 1, recall ½, F1 ⅔ and Jaccard ½, despite predicting the same molecule.

   This also affects the carbon exclusions:
   - `Nc1ccccn1`, 2-aminopyridine: reversing all six ring bond orders changes `primary_amine` from 1 to 0. The attachment carbon alternates between C=C and C=N.
   - `Oc1ccccn1`: the same reversal gains or loses `hydroxyl`.
   - Pyridazine: reversing the ring phase changes `imine` from 2 to 0, exchanging two C=N bonds for an N=N bond.

   Rust and the direct RDKit counter agree on these errors. Accepting both phases of an isolated alternating six-ring does **not** establish general Kekulé invariance.

   **Minimal fix:** normalize equivalent Kekulé assignments before detection and scoring, and apply the same normalization to the reference. For fragments, normalization must preserve atom hydrogen counts, valence and open valence. If invariance is deliberately excluded from v1, describe the metric as agreement on bond-order motifs and explicitly disclose these penalties.

2. **High: export-reference counts can describe a different graph from the graph written beside them.**  
   [`functional_groups_ref.py:467`](tools/ms2/functional_groups_ref.py:467), original bonds copied at line 473.

   `SanitizeMol` perceives aromaticity; the subsequent `Kekulize` can select another assignment. Nevertheless, the output retains `m["bonds"]`.

   Reproduced with naphthalene in the RDKit numbering for `c1ccc2ccccc2c1`: use double bonds `(0,9), (1,2), (3,8), (4,5), (6,7)`, with all other molecular edges single. Direct counting of that stored graph gives `arene_ring=2, alkene=0`. `export_molecule_record` returns `arene_ring=1, alkene=2` alongside the original bonds.

   Expected: reference counts correspond to the supplied graph. Actual: an export comparison can report Rust/reference disagreement caused by reference normalization alone.

   **Minimal fix:** preserve the supplied orders during reference preparation, or write the actual post-normalization graph and explicitly compare normalized representations. Add this alternate-form export regression.

3. **High: the oracle excludes model-generatable fragments.**  
   [`functional_groups_eval.rs:296`](src/models/ms2/functional_groups_eval.rs:296); [`targets.rs:208`](src/models/ms2/targets.rs:208), `max_cuts=2` at line 75.

   Concrete parent: a central `C–O–C` ether, with each central carbon bearing three four-carbon alkyl arms. This is a neutral 27-heavy-atom tree.

   A seven-atom induced fragment containing `C–O–C` and the first carbon of two arms on each side determines `ether`: both central carbons have residual valence 1, so neither can acquire an excluded double bond.

   Expected unrestricted fragment oracle: includes `ether`. Actual recipe oracle: excludes it. Two cuts remove at most eight atoms from the core-containing component, leaving at least 19 atoms, above the 16-atom limit. Components without the core contain no ether.

   **Minimal fix:** compute the oracle over the actual connected fragment family and model atom/closure limits. Otherwise rename it **“two-cut recipe coverage”**, remove “ALL connected induced subgraphs” from its documentation, and do not present it as a model recall ceiling. Its union also has no top-k constraint.

4. **Medium: the evaluated candidates do not obey the stated 3–16-atom family.**  
   [`functional_groups_eval.rs:226`](src/models/ms2/functional_groups_eval.rs:226); [`grammar.rs:252`](src/models/ms2/grammar.rs:252).

   STOP is legal once one atom exists; device validation likewise requires at least one atom. Evaluation accepts `state.atoms() > 0`. The driver takes the upper bound from the checkpoint.

   Concrete candidate: C/H0/valence4 double-bonded to O/H0/valence2, followed by STOP, under a C1O1 conditioning budget. Expected under the stated family: excluded because it has two atoms. Actual: credited with `carbonyl`; the test at `tests/ms2_functional_groups.rs:379` explicitly expects this.

   **Minimal fix:** align generation and evaluation with the promised size bounds, or disclose the actual family and use it consistently for the oracle.

5. **Medium: some names conceal chemical classification errors, beyond the documented aromatic quirks.**  
   [`functional_groups.rs:335`](src/models/ms2/functional_groups.rs:335), amine patterns at lines 345–355.

   Concrete examples:
   - Formaldehyde `C=O`: expected `aldehyde=1`; both implementations return 0 because the pattern requires carbon H1 rather than allowing H2.
   - Pyrrole `c1cc[nH]c1`: conventionally a pyrrolic aromatic nitrogen; both return `secondary_amine=1`.
   - N-methylpyrrole `Cn1cccc1`: both return `tertiary_amine=1`.
   - Carbonic acid `OC(=O)O`: both return two `carboxylic_acid` instances, although these are carbonic-acid OH motifs.

   The literal table documents how these results arise, but the chemical naming caveats are not explicitly documented. Agreement here is shared specification error, not chemical validation.

   **Minimal fix:** allow H2 aldehyde carbon; exclude pyrrolic nitrogen from conventional amine classes; restrict carboxylic-acid centers or rename that class. If v1 intentionally retains literal motifs, document these examples explicitly.

6. **Medium: macro bootstrap intervals change the set of types being averaged.**  
   [`functional_groups_eval.rs:572`](src/models/ms2/functional_groups_eval.rs:572), support threshold at line 689.

   Concrete dataset: ten one-spectrum molecules with type A, all missed, and ten with type B, all correctly predicted. Point macro F1 is 0.5. A molecule-bootstrap draw containing nine A and eleven B observations drops A below support 10 and reports macro F1 1.0; the opposite imbalance reports 0.0.

   Expected: uncertainty for the reported macro average over the declared `types_used`. Actual: uncertainty also reflects changing that denominator between replicates.

   **Minimal fix:** select supported types once from the original split and pass those IDs into bootstrap recomputation, with an explicit convention for zero replicate support.

7. **Medium: the prior search omits the empty prediction set.**  
   [`functional_groups_eval.rs:473`](src/models/ms2/functional_groups_eval.rs:473).

   Concrete train set: only saturated hydrocarbons, such as propane and butane, so every mask is empty. Expected under the implemented conventions: an empty prior achieves micro F1 1. Actual: the sole threshold is zero, which selects all 26 types and achieves F1 0.

   **Minimal fix:** include an empty-set candidate in threshold optimization and retain the stated smallest-set tie-break.

8. **Medium: the formula prior uses the best candidate’s formula, not the top-ranked formula hypothesis.**  
   [`functional_groups_eval.rs:247`](src/models/ms2/functional_groups_eval.rs:247); claimed “top-1 retained formula” at line 498.

   Concrete records: N-containing formula A has formula log probability −0.1 and trace score −10; N-free formula B has −1 and −1. Formula A ranks first as a formula, but `top_formula` becomes B because its combined candidate score is better. The prior consequently removes nitrogen groups.

   Expected from the documentation: filter using A. Actual: filter using B. This uses inference-available information, so it is **not validation leakage**, but it makes the baseline depend on decoder performance.

   **Minimal fix:** take the top retained formula from formula-search output, or rename and document this as a best-candidate-formula prior.

9. **Medium: the live export comparison can fail or compare zero molecules and still pass.**  
   [`tests/ms2_functional_groups.rs:464`](tests/ms2_functional_groups.rs:464), comparison loop at line 472.

   Concrete scenario: an exception in the reference script while processing an ordinary ethanol export produces a nonzero exit. Expected: reference regression fails the test. Actual: the test prints “SKIP” and returns successfully. If every molecule is skipped by the tool, the empty mismatch list also passes.

   **Minimal fix:** distinguish missing prerequisites from tool failure; fail on nonzero execution after prerequisites are available; assert expected molecule keys/counts and acceptable skips. Use an explicit optional integration-test designation where necessary.

10. **Low, diagnostic-only: impossible acid uncertainty is reported.**  
    [`functional_groups.rs:1134`](src/models/ms2/functional_groups.rs:1134).

    Concrete fragment: methyl acetate’s `C(=O)–O` core, retaining the parent’s O/H0 type but omitting its methyl carbon. Expected: ester is incomplete; carboxylic acid is impossible because the oxygen’s known parent hydrogen count is zero. Actual: both acid and ester are flagged undetermined.

    **Minimal fix:** respect the oxygen hydrogen constraint in partial-core diagnostics. This does not change scored sets, but inflates `mean_undet`.

The chemistry audit of all 26 definitions follows. “Documented” means the literal pattern or representation notes specify the behavior; it does not imply conventional chemical nomenclature.

| ID / type | Implemented chemistry and concrete cases | Assessment; Rust/reference agreement |
|---|---|---|
| 1 `carbonyl` | Every C=O, including acids, amides, quinones and two anchors in CO2. | Documented generic overlap; agree. |
| 2 `carboxylic_acid` | Formic/acetic acids match; carbonic acid gives two, carbamic acid also matches. | Literal breadth documented; non-carboxylic-acid naming caveat undisclosed; agree. |
| 3 `ester` | Anhydride flanks, carbonate flanks, carbamates and lactones match. | Anhydrides explicitly documented; other breadth follows the table and fixture. Lactones are cyclic esters; carbonate/carbamate matches are broad ester motifs. Agree. |
| 4 `amide` | Urea gives two; carbamates, imides and lactams match. | Documented amidic overlap, chemically defensible as motifs; agree. |
| 5 `aldehyde` | Acetaldehyde matches; formic acid/formates excluded; formaldehyde missed. | H1 restriction documented, formaldehyde naming error undisclosed; agree incorrectly. |
| 6 `ketone` | Quinone carbonyls match: `O=C1C=CC(=O)C=C1` gives two. | Appropriate ketone motifs; agree. |
| 7 `hydroxyl` | Phenols, enols and hemiacetal OH match; acid OH excluded. | Broad hydroxyl class is reasonable; heteroaromatic phase dependence is erroneous. Agree. |
| 8 `ether` | Acetals, epoxides and furan match; ester oxygens excluded. | Acetal/epoxide ether links are reasonable. Furan inclusion needs disclosure as a heteroaromatic motif; agree. |
| 9 `primary_amine` | Alkylamines/aniline match; acyl and C=N-adjacent carbons excluded. | Ordinary cases agree; 2-aminopyridine phase dependence is an error. |
| 10 `secondary_amine` | Dialkylamines/piperidine match; pyrrole also matches. | Pyrrolic-N caveat undisclosed; both share it. |
| 11 `tertiary_amine` | Trialkylamines match; N-methylpyrrole also matches. | Pyrrolic-N caveat undisclosed; both share it. |
| 12 `nitrile` | Neutral C≡N motif. | Appropriate within the domain; agree. |
| 13 `imine` | Any C=N, including pyridine, oximes and amidine-like motifs. | Pyridine explicitly documented; heteroaromatic counts/presence can change by phase. Agree. |
| 14 `alkene` | C=C unless either carbon belongs to a matched alternating C/N six-ring. Furan/pyrrole and some fused aromatic bonds count. | Five-ring behavior explicitly documented; fused-ring dependence insufficiently disclosed. Both share the chemical misclassification. |
| 15 `alkyne` | Any C≡C. | Appropriate within the domain; agree. |
| 16 `thiol` | C–S/H1, including thioacetic acid `CC(=O)S`. | Literal breadth documented; thioacid-versus-thiol caveat not explained. Agree. |
| 17 `thioether` | C–S–C with S valence 2; thiophene also matches. | Valence restriction documented; aromatic sulfur inclusion undisclosed. Agree. |
| 18 `sulfonyl` | S/H0 bearing two =O, including sulfones and sulfonic-acid/sulfate derivatives. | Broad sulfonyl motif; agree. Sulfoxides are outside the atom-type domain. |
| 19 `sulfonamide` | Two S=O plus S–N. | Appropriate broad motif within the domain; agree. |
| 20 `phosphoryl` | P/H0=O; allowed phosphorus is valence 5. | Appropriate representation-specific motif; agree in-domain. Python lacks the explicit H0 filter, but every supported P type has H0. |
| 21 `fluoride` | C–F. | Covalent organofluorine motif, not ionic fluoride; documented pattern; agree. |
| 22 `chloride` | C–Cl. | Same naming distinction; agree. |
| 23 `bromide` | C–Br. | Same naming distinction; agree. |
| 24 `arene_ring` | Alternating six-cycle of C/N, including pyridine; incomplete fused-ring coverage. | Heteroarene breadth documented; Kekulé dependence is the major error. Agree on an identical graph. |
| 25 `iodide` | C–I. | Covalent organoiodine motif; agree. |
| 26 `carbamate_or_urea` | N–C(=O)–N or N–C(=O)–O, including carbamic acid. | Broad composite motif documented; agree. |

Tautomers are also different labels. RDKit confirmed that 2-pyridone `O=c1cccc[nH]1` gives `{carbonyl, amide, alkene}`, while 2-hydroxypyridine `Oc1ccccn1` gives `{hydroxyl, imine, arene_ring}` in its selected phase. There is no tautomer normalization. This is a representation policy requiring disclosure, rather than a Rust/reference mismatch.

Sulfur is restricted to valence 2 with H0/H1 or valence 6 with H0; phosphorus to valence 5 with H0. Sulfoxides, ordinary phosphines and other unsupported states are excluded before reference counting. Their absence must not be interpreted as detector sensitivity.

The **determined rule appears sound for induced fragments retaining the exact parent atom types and bond orders**:

- **Double-bond exclusions:** acetamide’s C–NH2 core with carbon residual 2 is uncertain when its carbonyl O is omitted. A carbon with residual 1 cannot hide a double bond, so accepting it is sound. The same reasoning applies to hydroxyl and ether exclusions.
- **Single-neighbor exclusion:** formic acid’s H1–C=O core has residual 1 and cannot be determined as aldehyde with its OH omitted. Acetaldehyde with the methyl neighbor included has residual 0 and is determined.
- **Alkene ring exclusion:** a proper benzene fragment is either rejected by a fully present alternating ring or has an alternating path to an open boundary and remains uncertain. The search reaches boundary atoms at distances through five bonds.
- **Arene completeness:** omitting a ring atom prevents its six-atom/six-bond match. Open substituent attachments do not invalidate a fully present ring.

I found no counterexample to that restricted soundness claim. It does not cover changing Kekulé forms, tautomers or atom metadata.

The rule is conservative in usefulness. No class is inherently impossible within 16 atoms, but alkenes require substantially more surrounding structure than their two-atom core. Even a propene fragment missing its methyl neighbor is uncertain. Six-ring detection needs all six atoms; aldehydes need their carbon’s remaining neighbor; tertiary amines need all three carbon attachments and enough context to settle their exclusions. Thus recall combines model coverage with detector certainty. Finding 3 means the current oracle cannot cleanly separate these effects.

The remaining evaluation checks passed source inspection:

- Eligibility uses finished, device-valid, nonduplicate records. Replay uses each record’s own conditioning formula.
- Ranking is descending raw formula-plus-trace score, with ascending trajectory ties.
- Replay failures remain as empty candidates and occupy top-k slots. They affect candidate coverage, but contribute no instances to instance precision.
- Default evaluation includes every loaded validation spectrum, including unlabeled spectra and spectra with no eligible candidate. `--limit-spectra` deliberately evaluates a prefix and reports the evaluated count.
- Micro metrics, per-type metrics, mean per-type macro F1, Jaccard and exact match follow their documented formulas.
- Set bootstrap resamples molecules and keeps their spectra together. Point estimates remain spectrum-weighted.
- Candidate/instance metrics pool **all eligible candidates**, regardless of the displayed k. Instance precision verifies only that an instance’s **type** occurs in the parent; it does not validate its location, multiplicity or containment.
- Tau uses one parent mask per train molecule only. I found no validation fitting. Generation requests use spectrum information, not the parent graph. Label/oracle rows explicitly use parent-derived information.

Empty predictions deserve particular care. With 100 nonempty truths and no predictions, reported micro precision is **1**, recall **0**, F1 **0**. With one correct claim and 99 abstentions, precision remains **1**. Most empty predictions do not numerically increase the ratio, but precision alone hides coverage. Empty-truth datasets have recall 1; both-empty spectra receive Jaccard and exact match 1. Instance precision and conditional `cand_all_real` also return 1 when their relevant denominator is empty.

Generic motifs can dominate micro scores. “Specific” removes **only carbonyl**, retaining alkene, arene, ether and hydroxyl. The JSON provides prior comparisons, per-type support/results and macro averages, so readers can assess this, subject to the baseline and bootstrap fixes above. The console summary emphasizes micro metrics and does not expose all those qualifications.

The reference is **independent at the implementation level**: RDKit SMARTS and separate Python filters replace Rust backtracking. It is not an independent chemical taxonomy: its cores, exclusions, anchor policy and ring criterion reproduce the same specification, including shared mistakes. The committed-fixture test checks every stored molecule and every type and fails loudly on mismatches. The optional live export test does not provide that guarantee.

**Verdict: reject** the current evaluation claim/report interpretation until Kekulé handling, reference graph consistency and oracle scope are corrected. The core replay and set-metric implementation is largely sound.

**What a reader of the reported numbers must be told:**

- These are overlapping structural-motif type sets; current results depend on Kekulé form and tautomer.
- “Specific” excludes only carbonyl.
- High precision can accompany near-total abstention; coverage and recall must accompany it.
- Instance/candidate statistics pool all eligible candidates and validate type presence, not structural correctness.
- Replay failures occupy ranking slots as empty predictions.
- The current oracle measures two-cut recipe coverage, not the complete model fragment family or a top-k ceiling.
- Macro metrics cover only types supported by at least ten spectra; spectra weight point estimates, while molecules are bootstrap clusters.
- Label/oracle rows use parent information; the formula prior currently uses the best candidate’s formula.
- RDKit agreement validates implementation agreement, not conventional chemical correctness.