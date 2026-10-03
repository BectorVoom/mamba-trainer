# Bounded molecular-completion ambiguity experiment

Status: first restricted pilot run, 2026-10-03. See the
[pilot results](MOLECULAR_COMPLETION_PILOT_RESULTS.md). The broader experiment
specified here remains pending.
Parent design: [molecular completion](MOLECULAR_COMPLETION_DESIGN.md).

## Question

Within a declared small chemical domain, how many distinct complete molecular
graphs agree with target mass and required open substructures? Measure the effect
of adding precursor constraints separately. This is an input-information audit,
not a claim about a trained model's achievable accuracy or physical stability.

## Initial domain and budgets

Start with two to six heavy atoms, C/N/O atom types drawn from the supported V0
vocabulary, connected closed-shell graphs and at most one independent cycle.
Use the current bond-order rules and count hydrogens through atom types. Exclude
unsupported stereo/charge states explicitly. Report results for this restricted
domain; a single-ring restriction excludes chemically legitimate larger domains.

Provisional per-query limits:

| Resource | Initial limit |
|---|---:|
| Formula enumeration work | 100,000 count-vector visits |
| Graph action extensions | 100,000 |
| Embedding-match search nodes | 100,000 |
| Canonicalization extensions, cumulative | 100,000 |
| Retained unique graphs | 10,000 |
| Wall-clock watchdog | 30 seconds |

These are safety bounds for a first measurement, not measured throughput targets.
Deterministic work limits define reproducible experiments. The wall-clock watchdog
is a separate failure/truncation status, because it can differ across machines.
Also record peak memory; stop with an explicit resource status on allocation limits.

## Query construction

1. Build a fixed, provenance-recorded set of small reference molecules covering
   chains, branching, unsaturation and a ring. Check each reference is inside the
   exact domain and representable by the existing grammar.
2. Extract one or more open subgraphs while preserving parent hydrogen counts.
   Include overlapping, disjoint, symmetric and insufficiently informative sets.
   Do not include ground-truth cross-substructure atom correspondence in the main
   unknown-overlap condition. Known correspondence is a separately labeled oracle.
3. Derive neutral target masses using the repository arithmetic and explicit
   precision metadata. Include accepted, rejected, boundary-ambiguous and unknown
   precision fixtures. Distinguish synthetic measurements from instrument data.
4. For precursor-condition comparisons, use independently validated compatible
   parent/target pairs and explicit conventions. If none are available, report
   that arm as not evaluated; do not fabricate a precursor by adding an arbitrary
   mass and call it fragmentation evidence.

## Reference search

Enumerate bounded formulas over the declared domain. For accepted mass candidates,
enumerate legal BFS graph traces from START using `TraceState` and the formula
budget. Do not seed the state with an arbitrary input subgraph: the existing grammar
does not expose unrestricted partial-graph completion.

At each possible STOP, require exact composition and zero residual valences before
performing typed subgraph-containment checks. Required embeddings are injective
within each supplied substructure; distinct substructures may share target atoms.
Unknown cross-substructure overlap is inferred by these mappings. When correspondence
is supplied, verify it across all maps. Count a target graph once, not once per
embedding or traversal. Canonical traces give identities within the stated graph
representation, not stereochemical identity beyond it.

Do not prune a partial target merely because a required substructure is not yet
contained. Initially perform containment only on completed graphs; add pruning
later only with a sound reference argument and tests. Keep all formula branches
needed for a completeness claim. Canonicalization or matching failures invalidate
that claim just as graph-search truncation does.

Mass-ambiguous candidates are a separate unresolved class. Unknown precision cannot
yield a mass-constrained zero count; optionally run a separately labeled structural
audit with mass filtering disabled.

## Output and interpretation

For every query record domain, input hash, seed, candidate formulas, numeric mass
status, unique accepted graphs, unresolved hypotheses, ground-truth recovery,
every work counter, elapsed time and termination reason.

Use distinct statuses:

- `complete`: all relevant formula, trace, match and identity searches completed.
- `search_budget_exhausted`: candidate count is a lower bound, not an exact count.
- `mass_evidence_unresolved`: boundary ambiguity or unavailable precision prevents
  a definitive mass-constrained count.
- `unsupported_input`: chemistry or semantics outside the declared domain.
- `reference_error`: arithmetic, canonicalization or other validation failure.

Record multiple reasons if applicable. Only a complete search with resolved mass
evidence may certify zero compatible graphs, and only within its declared domain.
Do not equate zero candidates in the restricted domain with global inconsistency.

Report exact-count distributions only for completed, resolved queries, alongside
the completion rate and censored lower-bound distribution. Report counts as
constraints are added: mass only, one substructure, multiple substructures, known
overlap oracle, and precursor evidence if available. Count monotonicity is a useful
correctness check when the domains and evidence are fixed.

More candidates means weaker identifiability under the chosen constraints. It does
not imply a `1 / count` bound on accuracy unless candidates are conditionally
equiprobable. Learned chemical priors may be strongly nonuniform.

## Acceptance and next decision

- [ ] Independent tiny fixtures have manually established completion sets, not
      only outputs generated by the same search code.
- [ ] Tests cover composition exhaustion, residual valence, overlap double counting,
      incompatible shared atom types, symmetry, ambiguous masses, missing precision,
      invalid input and each resource limit.
- [ ] Complete searches recover all in-domain fixture targets and never certify a
      truncated count as exact. Repeated deterministic runs agree.
- [ ] CPU/GPU checks cover shared mass and legality behavior; exposed Rust/Python
      semantics agree. Record unavailable hardware rather than skipping silently.
- [ ] The report includes domain exclusions, every incomplete search, input
      provenance and exact configuration. No accuracy or performance claim is made
      from an unmeasured estimate.

Proceed to a small supervised Mamba-3 completion experiment if the audit is correct
and manageable. If searches mostly truncate, shrink the domain or revise the
enumerator before drawing identifiability conclusions. If ambiguity remains high,
evaluate ranking and calibrated abstention or seek additional evidence. Do not
automatically scale model size to compensate for missing information.
