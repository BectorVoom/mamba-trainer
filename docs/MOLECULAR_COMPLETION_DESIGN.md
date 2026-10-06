# Mamba-3 molecular completion with categorical composition

Status: revised proposal, progress clarified 2026-10-04; request-layer
MVP and a first native model added 2026-10-05. Bounded completion reference code
and separate database/ranking baselines exist; a strict versioned JSON request
layer (`molecular-completion-request-v1`, `src/models/ms2/completion_request.rs`,
`mamba3_rl.molecular_completion_run`, `docs/MOLECULAR_COMPLETION_API.md`)
wraps the tested bounded audit.

A first native **completion-conditioned Mamba-3 model** now exists
(`src/models/ms2/completion_model.rs`, protocol
`molecular-completion-generate-v1`, `mamba3_rl.MolecularCompletionModel`): a
substructure-set encoder feeding the existing decoder, trained by teacher
forcing and sampled under an exact-completion grammar on the device. It has
been trained and evaluated in one **synthetic setting only** — the query is a
held-out molecule's exact composition plus substructures cut from that molecule
with its own hydrogen counts. In that setting its top-25 recovery is **low**
(at a matched training budget, 3.4% of 379 validation molecules at 64 samples
per query and 5.0% at 256, against 0.5% for a formula-only control and 0%
untrained; 5.8% and 8.4% after longer training, not yet converged): see
[experimental progress](MOLECULAR_COMPLETION_EXPERIMENT.md). This is not high
prediction accuracy, and it says nothing yet about real fragment evidence or
calibration.

Added 2026-10-06, after the user fixed the query definition ("substructure
molecular graphs refer to functional groups") and asked for three more
capabilities: functional-group queries (Ertl's algorithm, with three acceptance
rules), **formula hypotheses from a mass**, **stereochemistry** as exact
perception and enumeration (the data carries no stereo labels, so not
prediction), and an optional **physical-verification** tool (force-field
geometry and structural alerts, never a stability claim). With functional-group
queries the model's top-25 recovery is 3–8% depending on the sample budget; a
model-free count shows why and what would change it — see "What evidence
reaches 90%" in the [experimental progress](MOLECULAR_COMPLETION_EXPERIMENT.md).
Still missing: explicit precursor conventions, a candidate-database stage in
front of or instead of free generation, matched Transformer and retrieval
baselines, calibration, and any use of measured spectra. The model is intended
for a Kaggle competition in which the mass and the functional groups would have
to be inferred from a measured spectrum; every result here takes them from the
answer.
The trained Ridge baseline is a different model and task. This document
incorporates the Fable review and the corrections listed below.

## Objective and scope

Generate ranked molecular graphs from precursor neutral mass, target neutral mass,
and supplied substructure graphs. Return explicit evidence for mass agreement,
substructure containment, chemical validity, and physical verification. These are
distinct properties; none establishes unique experimental identity by itself.

This is a proposed completion stage alongside the repository MS2-to-substructure
implementation, not a change to its frozen contracts. Its candidate substructures
are a possible future input. Generic supplied substructures require an adapter with explicit hydrogen
semantics.

The meaning of the user's second mass is still unresolved. Requests must declare
`mass_role`: target molecule, target fragment, or neutral loss. The provisional
design assumes target fragment. Do not deploy that interpretation as an implicit
default. For a whole-target request, explicitly specify how the precursor relates
to it; redundant measurements are not independent structural evidence.

## Primary prediction target: top-25 recovery

The user accepts a shortlist of up to 25 candidates. The primary accuracy metric
is therefore **top-25 exact-identity recovery**: the fraction of held-out queries
whose correct target is among the first 25 distinct returned molecular identities
under the declared identity policy. Top-1 is a secondary diagnostic, not the
primary success requirement. **The target is 95% top-25 recovery** (set by the
user on 2026-10-05), counted over all selected queries as defined below. It was
set after the first synthetic measurements, so it is a predeclared target for
every later evaluation, not for those runs. The best recorded result is 8.4%
(synthetic setting, 256 samples per query): the gap is more than a factor of
ten, and the measurements point at evidence rather than training time as the
limit — in that run recovery was 75% for molecules of at most 12 heavy atoms and
0% from 29 atoms up, 40% when the supplied substructures together held at least
as many atoms as the molecule and 0% when they held under a quarter. Whenever
the target was sampled at all it was inside the top 25. Reaching 95% therefore
needs queries that pin the molecule down far more tightly (near-complete
substructure coverage, a candidate database, or measured-spectrum evidence), not
only a better decoder.

Emit at most 25 distinct, valid candidates; duplicates do not occupy extra slots
and fewer returned candidates are not padded. Candidate acceptance still requires
composition, valence, containment and the declared mass-evidence checks. Fix the
formula/search/sampling budget independently of knowledge of the target, and
record failures, unresolved mass and truncation. Report recovery over all selected
queries plus clearly labeled domain/eligibility subgroups; do not silently remove
failed or abstained requests from the headline denominator.

A shortlist objective favors coverage and diversity as well as ranking. Include
formula-hypothesis coverage, graph validity, containment and unique-candidate count
at K=25, and compare retrieval and Transformer baselines at the same evidence and
compute budget. A selective-confidence gate is a separate metric: failure of the
previous top-1 confidence gate does not itself measure top-25 recovery.

## Input and chemical domain

Each request contains neutral masses with units, rounding uncertainty, tolerance,
mass role and neutralization convention; substructure graphs with provenance and
certain/tentative status; and a versioned chemistry domain. **The substructure
graphs are functional groups** (user, 2026-10-05): the groups of Ertl's
algorithm, each given as its own atoms with their hydrogen counts. A request
states how the list is to be read — each group contained somewhere
(`contained`), the groups as separate occurrences (`disjoint_occurrences`), or
the complete list of the molecule's functional groups
(`complete_functional_groups`). If masses are derived
from ions, retain the charge, adduct and hydrogen-shift hypotheses used to derive
them. Unknown precision means mass evidence is unavailable, not zero uncertainty.

For upstream V0 substructures, retain `(element, parent hydrogen count, valence)`
atom types and explicit bond orders from contract sections 4.2 and 4.5. Hydrogens
are counted once through atom types. Never sanitize an open substructure as an
isolated capped molecule and silently reuse its resulting hydrogen counts.

For arbitrary substructures whose parent hydrogens are unknown, represent that
uncertainty as candidate atom types or reject the input as insufficiently
specified. Do not invent hydrogen counts. Parent-relative types can be preserved
when completing the parent; generating a rearranged or hydrogen-shifted fragment
may require a separately specified retyping operation. A mass shift alone does
not locate those hydrogens or determine a new graph.

V0 permits only its existing neutral-atom vocabulary. Net-neutral zwitterions,
radicals, isotope labels and unsupported stereochemical distinctions remain
outside this initial completion domain. Broader charge support must follow a
versioned extension, including the V1 plans in contract section 4.6.

## Formula and mass hypotheses

Enumerate a bounded set of target formulas, then rank them using the conditioning
encoder. Record formula-search truncation separately from chemical exclusion.
For numerical acceptance reuse the integer arithmetic and rounding bounds in
contract section 5. With absolute residual `r`, total error bound `E`, and tolerance
`tol`:

- Accept when `r + E <= tol`.
- Reject when `r > tol + E`.
- Otherwise retain a separate `mass_boundary_ambiguous` status.
- Unknown precision disables the corresponding mass decision.

The neutral-mass adapter must use the applicable neutral-composition arithmetic
bound; do not blindly import the electron term from a charged-ion calculation.

With explicitly defined neutral species in a closed atom-balance interpretation,
`f_precursor = f_target + f_complement` is a conservation constraint. With the
repository's parent-subgraph/fragment-ion interpretation, retain the bounded
hydrogen shift from contract section 4.3 and distinguish parent-relative hydrogen
counts from ion composition. Heavy-atom conservation is necessary but does not
prove a fragmentation pathway. Unknown attachment bond partitions can also leave
the cut count, and therefore allowed shift, unresolved. Keep compatible hypotheses
instead of replacing an unknown cut count with the sum of residual valences.

## Model architecture

1. Encode each substructure using a shared message-passing network. Keep atom
   embeddings for attachment pointers and use permutation-invariant pooling across
   the substructure set. Tentative substructures contribute soft evidence; only
   confirmed requirements become hard containment constraints.
2. Encode masses, uncertainties, formula hypotheses, missingness and mass role.
3. Use the existing Mamba-3 implementation to predict construction actions with
   explicit graph memory and factorized atom/bond/pointer heads. Feed remaining
   composition, residual valence and containment progress into each step.
4. Mask illegal actions using an authoritative reference definition shared with
   device implementations. Check whether any action is legal before softmax.
5. Rank completed graphs, deduplicate identities and return calibrated confidence
   only after calibration against the declared prediction target.

Start with the existing SISO configuration and canonical BFS actions. Model width,
depth, MIMO, beam search and free-order rewriting are experiments, not prerequisites.
Use independent formula-stratified sampling for the first trainable version, as in
the upstream design. Preserve exact graph state outside the recurrent state.

The encoder must not treat supplied graph labels as chemical information. Label
permutations can test invariance, but action labels must still be legal BFS traces.
Use canonical traces initially; arbitrary construction-order augmentation requires
a new action grammar. Different valid histories need not have identical hidden
states or individual path probabilities.

### What the first native version implements (2026-10-05)

| Design item | Implemented | Deferred |
|---|---|---|
| 1. Substructure encoder | Shared message passing over typed pattern graphs (24 atom slots, one weight per bond order, open-valence feature), atom embeddings as cross-attention memory: atom memory equivariant to atom order, pooled context and the decoder's predictions invariant to pattern order and atom order (tested within numerical tolerance) | Tentative (soft) substructures: the request layer rejects them |
| 2. Mass / formula encoding | The exact composition, as `ln(1 + count)` features. A request may give a neutral mass or a precursor m/z with adduct instead: formulas are enumerated with the three-valued mass decision, filtered by necessary conditions, and the sampling budget is split over them (no learned formula ranker) | A learned formula prior; `mass_role` other than target molecule; precursor/fragment relations |
| 3. Mamba-3 action decoder | The existing `Ms2Decoder` unchanged: factorized kind/type/bond/pointer heads, residual valences in the pointer keys | Remaining composition and containment progress as per-step inputs |
| 4. Legality masks | `completion-exact-v2`, one host reference with device twins (see "Complete molecules versus existing subgraphs") | Containment-aware legality |
| 5. Ranking | Distinct identities ranked by sample frequency, at most 25, never padded | Calibrated confidence; a learned reranker |
| Stereochemistry | Per candidate: stereo elements and the exact number of distinct stereoisomers from graph automorphisms, each enumerable on request; kinds not modelled are flagged (`stereo-perception-v2`) | Predicting which stereoisomer; axial, atropisomeric, phosphorus/sulfur and constrained-nitrogen stereo |
| Physical verification | A separate RDKit tool: structural alerts, 3D embedding, force-field relaxation, stereo re-checked from coordinates (`physical-verification-v1`) | Electronic-structure refinement; any statement about thermodynamic or kinetic stability |

The decoder's learned conditioning on the substructures is weak in the first
runs (most completed samples do not contain them), which is the measured
bottleneck; the deferred per-step containment input of item 3 is the design's
own answer to it and is the next thing to build.

## Categorical composition and its implementation boundary

Specify an ambient category of finite typed incidence graphs, with atoms and
bond incidences represented explicitly, and injective structure-preserving maps
for overlaps. Chemical restrictions such as no self-bonds, allowed bond orders,
and valence limits are additional application conditions. This avoids assuming
that chemically valid simple graphs are closed under all required constructions.

An overlap span `S_i <- O_ij -> S_j` describes shared atoms and bonds. Its pushout
identifies precisely the specified shared structure. Adding a bond between distinct
atoms is a separate operation, not an identification. Multiple overlap maps must
be globally consistent: their quotient must not collapse distinct atoms of any
required embedded substructure or equate incompatible types. Pairwise matches alone
do not establish this. Count composition on the resulting identified atoms.

Residual attachment valence does not specify attachment bond multiplicity. For
example, residual two may allow one double bond or two single bonds. Enumerate or
predict the bond-order partition as well as overlap maps and attachment endpoints.
The graph encoder's embeddings and symbolic substructure embeddings are different
objects and should have distinct names in code.

For general rewrites, use spans `L <- K -> R` with explicit positive and negative
application conditions. DPO dangling and identification conditions are structural
conditions; they do not replace valence checks, duplicate-bond checks or mass
bookkeeping. A rewrite trace is a construction certificate, not evidence that the
same steps are a chemical reaction mechanism.

The first version uses this formalism to specify overlap consistency and reference
tests, while executing the existing BFS construction grammar. Seeding an arbitrary
partial graph or applying unrestricted DPO edits cannot be assumed compatible with
that grammar. A later rewrite decoder needs its own versioned state, rule matching,
termination bounds, device implementation and tests.

Measure the benefit of explicit overlap reasoning against containment checks alone
at matched search budget. Naming a graph merge a pushout is not an accuracy or
efficiency improvement. A curated fragmentation-rule library is a separate possible
deliverable; the cited graph transformation software is not itself evidence of a
validated rule library for this dataset.

## Complete molecules versus existing subgraphs

`TraceState` currently enforces a composition upper bound, and permits STOP with
remaining composition and open valences. That is correct for its subgraph task.
A completion-specific wrapper must additionally require:

- Exact target composition, including hydrogens.
- Zero residual valence for every atom under the chosen closed-molecule domain.
- Connectedness, supported chemical validation and all required embeddings.
- An explicit mass-evidence status; ambiguous/unavailable evidence cannot be
  reported as accepted mass agreement.

Reject or retain nonterminal branches according to these completion rules without
changing existing subgraph STOP semantics. A legal subgraph STOP is not a legal
completion STOP. No available continuation means branch failure, not permission
to return an incomplete molecule.

Implemented as grammar version `completion-exact-v2`
(`TraceState::new_exact`; on the device, budget flag value 2 of the existing
replay, sampler and validation kernels — no new kernel). Three rules, each a
necessary condition for a complete molecule, so no completable continuation is
ever forbidden: STOP only on exact composition with zero residual valence; an
atom may attach only where every earlier atom is already closed (later tokens
can never reach them); and an action is legal only if the state after it still
passes four cheap necessary conditions (hydrogens attainable by the remaining
atoms, an open attachment site, closability once all atoms are placed, valence
parity and bounds). Measured: an exhaustive search over six small formulas keeps
every molecule and every completing trace (at most six atoms, one closure);
dead-end prefixes fall from hundreds or thousands to 0–14; the canonical trace
of each of the 150,425 training molecules replays legally; device and host
masks agree on every token of 17,876 real molecules (495,285 tokens) on Metal
and of 354 on both backends. Beyond those checks the claim rests on the
necessity argument for each rule. The conditions are not sufficient: at 32 atoms about 30% of a
trained model's samples still end in a dead end none of them explains.
Connectedness is guaranteed by the grammar; required embeddings are checked on
the host for every finished sample, with the exact replay as the authority.

The frozen V0 limits are 16 atoms and four closures. The completion model uses
32 atoms and six closures (trace length 40), which the existing 32-bit masks
support unchanged and which keeps 354 of the 379 validation molecules. Current
legality masks impose
a maximum of 32 atoms, despite the host `MolGraph` accepting larger graphs. A
larger completion domain needs wider or segmented masks, audited pointer capacity,
device-kernel changes and a contract version. Increasing a model configuration
alone does not remove the limit. An eight-bit pointer does not itself impose a
32-atom limit; the present masks do.

## Physical verification

Keep the on-device structural generator separate from an optional asynchronous
physical-verification service. Structural candidates are not labeled physically
verified before that service returns evidence. This separation preserves the
existing generator's on-device validation and selection requirement; the broader
service is a new deployment component.

For a bounded shortlist, generate conformers, optimize supported structures, and
check convergence, clashes, stereochemistry and connectivity. Refine selected
candidates with an appropriate electronic-structure method if needed. Record
unsupported parameters and calculation failures explicitly.

Declare phase, solvent where applicable, temperature, charge, spin, method and
acceptance criteria. A local minimum with no significant imaginary vibrational
modes is evidence at that computational level, not a kinetic lifetime or shelf
stability guarantee. Raw force-field energies across arbitrary molecules are not
a universal ranking scale. Observed transient fragments and stable neutral
products remain different prediction targets.

Return graph identity, formula, mass status/residual, required substructure maps,
construction trace, domain version, search-completeness status, ranking score,
and physical status with method-specific evidence. Unspecified stereochemistry
must remain unspecified in the identity claim.

## Training and evaluation gates

First run the [bounded ambiguity experiment](MOLECULAR_COMPLETION_EXPERIMENT.md)
and the [database-first evaluation](MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md).
Then use known molecules with parent-relative open-substructure extraction for
supervised completion. Separate synthetic containment labels from experimentally
supported fragment assignments. Split by molecule/scaffold before sampling
substructures to avoid leakage.

Train formula and legal-action likelihood first, adding overlap-map supervision
where unambiguous. Treat symmetry-equivalent mappings as equivalent targets rather
than contradictory labels. Later add physical-ranking supervision with explicit
label provenance; invalid calculations are not automatically unstable negatives.

Evaluate formula coverage, complete-graph validity, containment, top-25 recovery
(primary; top-1/top-10 secondary),
diversity, calibration and abstention with search/domain exclusions reported.
Compare the existing grammar, overlap-aware conditioning, and a matched Transformer
policy before attributing gains to Mamba-3 or category theory. Physical pass rates
must report how many candidates were selected for verification.

Follow [test guidelines](test_guidline.md): CPU and actual GPU correctness plus
Rust/Python parity for exposed behavior. Profile before performance optimization.
An unavailable device or dependency is a recorded limitation, not a passing check.

## What the experiments establish about prediction accuracy

The experimental purpose is to assess whether the proposed model can achieve
high molecular prediction accuracy. The current answer is **not demonstrated**:
the completion-conditioned model has now been trained and evaluated in one
synthetic setting, where its top-25 recovery is between 3% and 9%. That is a first
measurement of an early model (weak use of the substructures, a reduced
architecture, one seed), not an upper bound on the design — but nothing recorded
supports a claim of high accuracy.

| Evidence | Established result | What it does not establish |
|---|---|---|
| Bounded synthetic completion audit | Exact graph counts for 21/26 resolved complete queries; two larger searches censored | Learned top-1/top-k accuracy; 80.8% is a search completion rate |
| Database/substructure experiments | Cheap evidence narrows candidate pools; sampled MassSpecGym bonded-pair median remains 32, including censored queries | A universal accuracy ceiling or benefit from a Mamba decoder |
| Ridge spectrum-to-fingerprint baseline | On 200 val queries, top-1 12/200 vs uniform 8/200; top-10 49/200 vs 43/200; calibration gate accepts 0/200 | High predictive accuracy, statistically demonstrated improvement, or a test of the proposed mass/substructure-conditioned generator |
| ChEBI metadata ranker | Small held-out slice top-1 0.54; selective precision 0.88 at coverage 0.50 | Transfer to hard same-formula pools or performance of the designed model |
| External checks and QM9 controls | Bounded pool coverage/ranking and restricted computational evidence | Validated precursor fragmentation, a deployed physical verifier, or general high-accuracy generation |
| Native completion-conditioned model, synthetic setting (2026-10-05) | Exact composition + substructures cut from the target (about 43% of its atoms), 150,425 training molecules, 379 identity-fold validation molecules: top-25 13/379 (3.4%, 95% interval 1.6–5.5) at 64 samples, 19/379 (5.0%, 2.9–7.4) at 256; formula-only control 2/379; untrained 0/379; after 64,000 steps (no matched control) 22/379 and 32/379 (8.4%, 5.8–11.4) | High accuracy; anything about mass-derived formulas, measured fragments, scaffold-novel molecules (1/171, then 7/171, at 256 samples), calibration, a Mamba advantage over a Transformer, or physical validity |

A post-hoc reaggregation of the frozen 200-query Ridge val ranks, performed after
the user selected K=25, gives 91/200 (45.5%) raw top-25 recovery for the predictor,
versus uniform 68/200 (34.0%), mass residual 72/200 (36.0%) and prior 74/200
(37.0%). This is a new metric on existing predictions, not retraining or a new
held-out test. It includes seeded fallback tails for unscored candidates;
35/200 pools have at most 25 members. No selective-confidence acceptance claim
is made, and the designed Mamba model remains unevaluated. Details and source hash:
`experiments/molecular_completion/20261003_predictor/top25_reanalysis.json`.

The Ridge baseline uses measured spectra to rank a supplied candidate pool. The
proposed model conditions on masses and supplied typed substructures and constructs
graphs. Their inputs, representation and prediction task differ. A weak Ridge
result neither validates nor rules out the Mamba design, and complete reference
recovery does not substitute for learned-model accuracy.

The 2026-10-05 synthetic run covers parts of the list below: a frozen (synthetic,
oracle) interpretation and identity policy, a trained model with two controls,
saved checkpoints and per-query predictions with intervals, and every excluded
or failed query kept in the denominator. It does not cover: a target set in
advance (the 95% target was set afterwards), calibration, a scaffold-disjoint untouched test set (the validation fold was
used for checkpoint selection), retrieval and Transformer baselines, or
legitimate (non-oracle) conditioning inputs.

A direct test still needs:

1. A frozen request interpretation (`mass_role`, precursor relationship and
   hydrogen semantics), declared domain and prediction identity policy.
2. The numerical top-25 recovery target — now set: 95% — plus validity,
   containment, abstention coverage and calibration criteria, which are not. The user accepts top-25 as the
   primary metric; a 90%-precision selective gate is a different requirement.
3. Molecule/scaffold-disjoint training, validation and held-out test queries with
   legitimate conditioning inputs; oracle overlap and synthetic extraction are
   separately labeled, with no target-information leakage.
4. A trained completion-conditioned model and matched retrieval/Transformer
   baselines, evaluated with the same evidence, domains and search budgets.
5. Saved checkpoints, per-query predictions and confidence intervals, with
   out-of-domain, pool-missing, unresolved and search-truncated cases retained in
   the reported accounting. Keep calibration and model selection off the test set.

Existing reports: [database](MOLECULAR_COMPLETION_DATABASE_RESULTS.md),
[spectral baseline](MOLECULAR_COMPLETION_SPECTRAL_RESULTS.md),
[Ridge predictor](MOLECULAR_COMPLETION_PREDICTOR_RESULTS.md), and
[external checks](MOLECULAR_COMPLETION_REMAINING_RESULTS.md). These describe
separate bounded experiments, not an implementation of this entire architecture.

## Review disposition

Accepted: bounded graph domain, parent-relative hydrogen semantics, unknown bond
partitions, three-valued mass decisions, legal traversal constraints and separate
physical verification. Additionally identified: completion needs stronger STOP
conditions than the existing subgraph grammar.

Qualified: hydrogen transfer does not invalidate atom conservation for explicitly
defined species; it changes the mapping from parent-relative subgraphs to measured
ions. The on-device rule applies to the existing generator, not every possible
new service. Multiple valid BFS orders exist, although unrestricted rewrite orders
are not supported by the current grammar.

Corrected: DPO structural conditions are not chemical checks; candidate count is
not an accuracy ceiling without a conditional probability model; exhaustive
enumeration is not presumed practical at 16 atoms. Category theory remains an
explicit specification whose operational value must be measured.

## References

- MS2 repository contracts, sections
  4.2–4.6 and 5.
- [Mamba-3 paper](https://arxiv.org/abs/2603.15569).
- [Chemical graph transformation software](https://arxiv.org/abs/1603.02481).
- [RDKit chemistry and conformer methods](https://rdkit.org/docs/RDKit_Book.html).
- [xTB vibrational analysis](https://xtb-docs.readthedocs.io/en/latest/hessian.html).
