# Bounded molecular-completion ambiguity experiment

Status: bounded reference audit completed on 2026-10-04. Of 26 synthetic
queries, 21 completed with resolved mass evidence, two had unresolved mass
evidence, one had unsupported oracle input, and two exhausted their search
budgets. The broader corpus experiment and precursor comparison remain pending.
See
[ambiguity results 2026-10-04](MOLECULAR_COMPLETION_AMBIGUITY_RESULTS.md)
and the earlier first-restricted pilot run (2026-10-03):
[pilot results](MOLECULAR_COMPLETION_PILOT_RESULTS.md).
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
| Modeled retained identities + frontier memory | 64 MiB |
| Wall-clock watchdog | 30 seconds |

These are safety bounds for a first measurement, not measured throughput targets.
Deterministic work limits define reproducible experiments. The wall-clock watchdog
is a separate failure/truncation status, because it can differ across machines.
Record whole-process peak RSS separately. The modeled memory limit excludes
parser, caller and temporary allocations; it is not an allocator-level RSS cap.
Input limits are eight substructures, 24 total pattern atoms and 16 correspondence
pairs.

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
Unknown cross-substructure overlap is inferred by these mappings. When complete
correspondence is supplied, enforce its equivalence classes across all maps:
different classes must map to different target atoms. An empty supplied relation
means all pattern atoms are disjoint; an absent relation means overlap is unknown.
Count a target graph once, not once per embedding or traversal. Canonical traces give identities within the stated graph
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

## Measured audit results (2026-10-04)

The resolved completion rate was 21/26 (80.8%). Exact-count distributions include
only these completed queries; unresolved, unsupported and truncated queries are
reported separately.

| Exact compatible graphs | Completed queries |
|---:|---:|
| 0 | 2 |
| 1 | 10 |
| 2 | 5 |
| 3 | 3 |
| 5 | 1 |

The full-domain C5H12 and C6H14 searches retained three and five graphs,
respectively, before reaching the graph-extension limit. These are lower bounds,
not established exact counts. All supplied in-domain references in completed
searches were recovered; an unrelated reference supplied as a negative control
was not recovered. Recovery is unknown when its shared resource budget is exhausted.

For two singleton methyl patterns at C2H6O mass, unknown overlap admits ethanol
and dimethyl ether (two graphs). The known-disjoint oracle admits only dimethyl
ether (one graph), demonstrating that absent and empty correspondence differ.

Final validation passed 34 completion integration tests on CPU and 34 with wgpu
features, seven internal completion tests on CPU, and seven Python API tests
against a rebuilt and installed CPU wheel. The independent Python enumerator
matched all 24 fixtures within its four-heavy-atom limit; the two larger fixtures
were explicitly excluded from that cross-check. Earlier chemistry, contract,
decoder and formula regressions also passed on CPU and GPU. The completion
reference search itself runs on the host; shared device mass and grammar behavior
was checked on Apple M1 / Metal.

The final release audit took 1.581 seconds with whole-process peak RSS of
7,585,792 bytes (7.23 MiB), measured after compilation with `/usr/bin/time -l`.
This is a bounded audit measurement, not an optimization or throughput claim.
No performance optimization or model training was performed. Precursor evidence
remains `not_evaluated` because no independently validated compatible pairs were
provided.

Artifacts are recorded under
`experiments/molecular_completion/20261004_completion_ambiguity/`:
`completion_report.json` contains per-query results; `supervisor_distribution.json`
separates exact counts from censored lower bounds; `supervisor_experiment_metadata.json`
records commands, input/source hashes, platform and peak RSS; `final_validation.json`
records final validation commands and exit codes. Earlier failed environment and
compile attempts are preserved alongside successful reruns.

## Acceptance and next decision

- [x] Independent tiny fixtures have manually established completion sets
      (`experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json`
      with hand-authored `expected` counts; two identity sets hand-authored
      for `c2h6o`/`c2h7n`; the pilot and fixture-mirror cross-check both
      independently reproduce the small C/O counts).
- [x] Tests cover composition exhaustion, residual valence, overlap double
      counting, incompatible shared atom types, symmetry, ambiguous masses,
      missing precision, invalid input and each resource limit
      (`tests/ms2_completion.rs`: 34 tests covering
      `formula_visit_limit`, `graph_extension_limit`,
      `embedding_node_limit`, `canonicalization_limit`,
      `retained_graph_limit`, `memory_bound`, watchdog (zero and overflow),
      and the irresolvable-no-zero rule for ambiguous/unknown precision).
- [x] Complete searches recover all in-domain fixture targets and never
      certify a truncated count as exact. Repeated deterministic runs agree
      (`fixture_expectations_hold`, `results_are_identical_across_runs`,
      deterministic parity vs the independent Python mirror).
- [x] CPU/GPU checks cover shared mass and legality behavior; exposed
      Rust/Python semantics agree. `grammar_replay` device-host parity and
      `formula_top` mass-table twin parity are executed on both CPU and
      `wgpu` (Metal) exit 0 (`supervisor_validation.json`,
      `supervisor_gpu.log`, and final checks in `final_validation.json`). Python mirror parity is run in
      `tests/ms2_completion.rs`; the PyO3 binding tests
      (`bindings/python/tests/test_ms2_completion.py`) run green over the
      installed wheel (`experiments/molecular_completion/20261004_completion_ambiguity/pkg-venv`
      is build output, not a source artifact).
- [x] The report includes domain exclusions, every incomplete search
      (`docs/MOLECULAR_COMPLETION_AMBIGUITY_RESULTS.md` per-fixture table),
      input provenance and exact configuration. No accuracy or performance
      claim is made from an unmeasured estimate. The measurement protocol
      separates identity/structural workloads (modeled storage counters)
      from the one whole-process peak RSS sampled by `/usr/bin/time -l`.

Next decision: the completion audit is correct and manageable in this
restricted domain except for the 5/6-atom full-domain fixtures, which
genuinely truncate at default budgets. Do not scale the model: raise the
trace-enumerator budget bound or shrink the domain before drawing
identifiability conclusions. If ambiguity remains high, evaluate
ranking and calibrated abstention or seek additional typed-evidence, not a
larger unsupervised model. The precursor arm remains explicitly
`not_evaluated` pending genuinely independently validated parent/target
pairs.
