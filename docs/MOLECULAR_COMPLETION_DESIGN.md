# Mamba-3 molecular completion with categorical composition

Status: revised proposal, 2026-10-03. No implementation, training result, stability
result, or performance measurement is claimed. This document incorporates the
Fable review and the corrections listed below.

## Objective and scope

Generate ranked molecular graphs from precursor neutral mass, target neutral mass,
and supplied substructure graphs. Return explicit evidence for mass agreement,
substructure containment, chemical validity, and physical verification. These are
distinct properties; none establishes unique experimental identity by itself.

This is a proposed completion stage alongside the MS2-to-substructure work in
progress in the local workspace, not a change to its frozen contracts. That work
is not yet part of `main`; its candidate substructures are a possible future
input. Generic supplied substructures require an adapter with explicit hydrogen
semantics.

The meaning of the user's second mass is still unresolved. Requests must declare
`mass_role`: target molecule, target fragment, or neutral loss. The provisional
design assumes target fragment. Do not deploy that interpretation as an implicit
default. For a whole-target request, explicitly specify how the precursor relates
to it; redundant measurements are not independent structural evidence.

## Input and chemical domain

Each request contains neutral masses with units, rounding uncertainty, tolerance,
mass role and neutralization convention; substructure graphs with provenance and
certain/tentative status; and a versioned chemistry domain. If masses are derived
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

The frozen V0 limits are 16 atoms and four closures. Current legality masks impose
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

First run the [bounded ambiguity experiment](MOLECULAR_COMPLETION_EXPERIMENT.md).
Then use known molecules with parent-relative open-substructure extraction for
supervised completion. Separate synthetic containment labels from experimentally
supported fragment assignments. Split by molecule/scaffold before sampling
substructures to avoid leakage.

Train formula and legal-action likelihood first, adding overlap-map supervision
where unambiguous. Treat symmetry-equivalent mappings as equivalent targets rather
than contradictory labels. Later add physical-ranking supervision with explicit
label provenance; invalid calculations are not automatically unstable negatives.

Evaluate formula coverage, complete-graph validity, containment, top-k recovery,
diversity, calibration and abstention with search/domain exclusions reported.
Compare the existing grammar, overlap-aware conditioning, and a matched Transformer
policy before attributing gains to Mamba-3 or category theory. Physical pass rates
must report how many candidates were selected for verification.

Follow [test guidelines](test_guidline.md): CPU and actual GPU correctness plus
Rust/Python parity for exposed behavior. Profile before performance optimization.
An unavailable device or dependency is a recorded limitation, not a passing check.

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

- MS2 repository contracts (currently in progress outside `main`), sections
  4.2–4.6 and 5.
- [Mamba-3 paper](https://arxiv.org/abs/2603.15569).
- [Chemical graph transformation software](https://arxiv.org/abs/1603.02481).
- [RDKit chemistry and conformer methods](https://rdkit.org/docs/RDKit_Book.html).
- [xTB vibrational analysis](https://xtb-docs.readthedocs.io/en/latest/hessian.html).
