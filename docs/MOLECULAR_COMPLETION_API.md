# Molecular completion request API (bounded reference MVP)

Status: implemented bounded reference MVP, 2026-10-05. This exposes a
strict versioned JSON request layer over the tested bounded audit in
`src/models/ms2/completion.rs`. It does **not** implement a trainable
Mamba conditioning model, a database/precursor service, or a physical
verification service.

## Rust

```rust
use mamba3::models::ms2::completion_request::molecular_completion_run;

let response_json = molecular_completion_run(&request_json)?;
```

## Python

```python
import mamba3_rl

response_json = mamba3_rl.molecular_completion_run(request_json)
```

Both are thin marshals over the same Rust implementation; Rust tests
and Python tests share the same fixture and expectations.

## Request (`molecular-completion-request-v1`)

All fields are validated strictly: wrong types and unknown fields are
rejected, and substructures/sub-objects carry explicit chemistry
semantics rather than guessed defaults.

```json
{
  "protocol": "molecular-completion-request-v1",
  "id": "ethanol_target_molecule",
  "provenance": "synthetic fixture: ...",
  "mass_role": "target_molecule",
  "neutralization": "already_neutral",
  "target_mass": {
    "units": "microdalton",
    "value": 46041865,
    "ppm_tenths": 100,
    "uncertainty_uda": 50,
    "source": "synthetic"
  },
  "domain": {
    "version": "completion-bounded-v1",
    "elements": ["C", "N", "O"],
    "min_heavy": 2,
    "max_heavy": 6,
    "max_ring_closures": 1
  },
  "substructures": [
    {
      "atoms": [3, 9],
      "bonds": [[0, 1, 1]],
      "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
      "provenance": "synthetic: free C(H2)-O(H1) bond",
      "certainty": "confirmed"
    }
  ],
  "correspondence": [],
  "budgets": {"retained_graphs": 10000, "watchdog_ms": 30000, "formula_visits": 100000,
              "graph_extensions": 100000, "embedding_nodes": 100000,
              "canonical_expansions": 100000, "memory_bytes": 67108864},
  "seed": 1
}
```

Rules:

- `id` and `provenance` (top level and per substructure) must be
  non-empty strings.
- `mass_role` must be `"target_molecule"`. `"target_fragment"` and
  `"neutral_loss"` are rejected with an actionable error: precursor
  conventions and fragment evidence are pending. Any supplied
  `precursor` mass is likewise rejected, never silently ignored.
- `neutralization` must be `"already_neutral"`.
- `target_mass.units` must be `"microdalton"`; `value` a
  non-negative integer; `ppm_tenths` an integer `<= 1000`;
  `uncertainty_uda` an integer or `null` (`null` = unknown precision,
  which disables the mass decision and never certifies zero graphs).
- `domain.version` must be `"completion-bounded-v1"`; `elements` a
  non-empty subset of `["C", "N", "O"]`; heavy bounds within
  `2 <= min_heavy <= max_heavy <= 6`; `max_ring_closures <= 1`.
- Every substructure needs `parent_hydrogen_semantics`
  (`"v0_parent_hydrogen_counts"` — V0 atom types already carry parent
  hydrogens), a non-empty `provenance`, and `certainty`. `certainty`
  other than `"confirmed"` (e.g. `"tentative"`) is rejected: tentative
  patterns are soft evidence and are not supported as hard
  constraints in this MVP. Unknown parent hydrogen semantics are
  rejected rather than silently hardened into constraints.
- `correspondence` is optional: **absent** means unknown overlap;
  **present as `[]`** means known-disjoint; a list of
  `[[[s1, a1], [s2, a2]], ...]` pins forced equalities. `null` is
  rejected — absent already means unknown.
- `budgets` fields must be positive integers; `seed` a non-negative
  integer. Both are forwarded to the existing audit adapter unchanged.

## Response

```json
{
  "protocol": "molecular-completion-request-v1",
  "query_id": "ethanol_target_molecule",
  "provenance": "...",
  "input_hash": "...",
  "ranking": {"status": "not_evaluated", "reason": "..."},
  "physical_verification": {"status": "not_evaluated", "reason": "..."},
  "identity_ordering": "deterministic baseline order; not calibrated confidence",
  "audit": { "...full bounded CompletionReport..." },
  "decoded_graphs": [{"identity": "...", "atoms": [...], "bonds": [[a, b, order], ...], "composition": "C2H6O"}]
}
```

- `audit` is the complete bounded `CompletionReport` (status,
  statuses, termination reasons, mass/domain/search statuses,
  counters, recovery, certifies_zero, accepted identities). Its
  `protocol` remains `completion-bounded-v1`.
- `decoded_graphs` decodes each accepted canonical identity through
  the existing grammar replay into typed atoms/bonds and composition.
- Ranking and physical verification are always `not_evaluated` with
  explicit reasons; identity order is the deterministic baseline
  enumeration order, not a calibrated confidence.

## Worked example

See `examples/molecular_completion_request.json`. A zero-uncertainty
variant (`"uncertainty_uda": null`) returns
`audit.mass_status == "unavailable"` and never certifies zero.

## Trained-model generation (`molecular-completion-generate-v1`)

Status: trained-model sampling behind one strict JSON protocol, implemented
once in Rust (`src/models/ms2/completion_api.rs`) and marshalled by Python.
The fixture (`tests/fixtures/ms2/completion_tiny.*`) is shared by the Rust
and Python suites.

### Rust

```rust
use std::path::Path;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_api::{CompletionService, GENERATE_PROTOCOL};

let device = Device::<Auto>::default();
let service = CompletionService::load(Path::new("tests/fixtures/ms2/completion_tiny.ckpt"), &device)?;
let description_json = service.describe();
let response_json = service.generate_json(&request_json)?;
```

### Python

```python
import mamba3_rl

model = mamba3_rl.MolecularCompletionModel("tests/fixtures/ms2/completion_tiny.ckpt")
description_json = model.describe()
response_json = model.generate(request_json)
protocol = mamba3_rl.molecular_completion_generate_protocol()
```

Both are thin marshals over the same Rust implementation; malformed
requests raise `ValueError` with an actionable message naming the field.

### Request (`molecular-completion-generate-v1`)

All objects reject unknown fields; wrong types are errors naming the field.

```json
{
  "protocol": "molecular-completion-generate-v1",
  "id": "ethanol-example",
  "provenance": "synthetic fixture",
  "mass_role": "target_molecule",
  "composition": {"C": 2, "H": 6, "O": 1},
  "substructures": [
    {"atoms": [3, 9], "bonds": [[0, 1, 1]],
     "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
     "certainty": "confirmed", "provenance": "synthetic: C(H2)-O(H1)"}
  ],
  "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
}
```

Rules:

- `id` and `provenance` (top level and per substructure) must be
  non-empty strings.
- `mass_role` must be `"target_molecule"`. Other roles are rejected with
  the same precursor-conventions-pending message the request layer uses.
- `composition` keys are element symbols of `chem::ELEMENTS` (unknown
  symbol, zero heavy atoms, or a count that does not fit are errors;
  absent elements are 0). This is a supplied exact composition.
- Substructures: at most 8 patterns and at most 24 atoms in total (the
  encoder limits `MAX_PATTERNS` / `PATTERN_SLOTS`; more is a schema
  error), atom type ids valid, bonds valid for `MolGraph::new`, each
  connected, `parent_hydrogen_semantics` must be
  `"v0_parent_hydrogen_counts"`, `certainty` must be `"confirmed"`
  (tentative rejected: soft evidence is not supported), non-empty
  `provenance`. Patterns may overlap in the target, so their sizes never
  add up: each single pattern must hold at most `max_atoms` atoms and at
  most as many heavy atoms as the composition has; only the total has to
  fit the encoder.
- `generation`: `trajectories` 1..=1024, `temperature` > 0, `seed` u64,
  `returned` 1..=25. For mass input `trajectories` is the total budget,
  split evenly over the selected formulas.
- A composition or substructure outside the model's domain limits is an
  `unsupported_input` response with no candidates, not an exception:
  composition heavy atoms above `max_atoms`, any single pattern above
  `max_atoms` or above the composition's heavy-atom count, or any pattern
  with more ring closures than `max_ring_closures`. The response names
  the violated limit in `unsupported` (`limit`, the limit's value in
  `allowed`, the observed value in `observed`), carries the request's
  count in `accounting.requested_trajectories` with every executed
  counter at 0 (`trajectories` is the executed count, so 0), and reports
  `search` and `ranking` as `not_evaluated`. Malformed JSON and schema
  violations are errors.

### Substructure semantics (`substructure_semantics`)
Optional request field (default `"contained"`); anything else is a schema
error. It picks the host acceptance rule for the supplied substructures
and is echoed in every response as
`"substructure_semantics": {"value": ..., "meaning": "<one line>"}`.
`input_hash` covers the field.

- `"contained"`: every pattern is contained somewhere; patterns may
  share atoms. The historical rule.
- `"disjoint_occurrences"`: the patterns are distinct occurrences — one
  joint injective embedding maps them onto pairwise disjoint atom sets
  (the audit's joint search with the known-disjoint correspondence).
- `"complete_functional_groups"`: the patterns are the complete list of
  the molecule's functional groups (`functional-groups-ertl-v1`) — the
  candidate's own functional groups, as a multiset of typed graphs up to
  isomorphism, must equal the supplied multiset. This implies
  `disjoint_occurrences` (equal multisets of whole groups are disjoint by
  construction); only the multiset comparison runs. Finished candidates
  that contain every supplied group disjointly but carry more groups are
  rejected as extra groups; candidates lacking a group are rejected as
  missing groups (both counted inside `rejected_containment`).

When to use which: random or hand-drawn fragments are `contained`;
functional groups treated as occurrences are `disjoint_occurrences`; a
full functional-group list is `complete_functional_groups`.

Note the complete mode depends on the documented approximation of
aromaticity in `functional-groups-ertl-v1` (`aromatic-ring-v2`): group
boundaries on (hetero)aromatic systems follow that perception, not
RDKit's. Models trained before the v2 change used `aromatic-ring-v1`
(single 5/6-rings only); v2 additionally judges every pair of rings
sharing exactly one bond as one circuit, which changes which atoms are
functional groups and hence the training inputs.

Before sampling, each request passes a necessary composition fit under
`disjoint_occurrences` and `complete_functional_groups`: the summed
element counts and hydrogens of all patterns must fit the composition
(summed, because the occurrences are disjoint), and under
`complete_functional_groups` the patterns' heteroatom counts must
additionally equal the composition's (`functional-groups-ertl-v1` marks
every heteroatom). A request failing it is well-formed but unsatisfiable:
the response is `status: "no_candidates"` with
`"infeasible": {"reason": ...}` (not `unsupported_input`), no candidates
and every executed counter at 0. A work-limit outcome during acceptance
is `containment_unresolved`, never acceptance.

### Fingerprint evidence (`fingerprint`)

Optional request field for models with a fingerprint encoder:

```json
"fingerprint": {"name": "morgan4096", "bits": [[7, 0.92], [43, 0.31]], "threshold": 0.1}
```

- `name` must be `"morgan4096"` (anything else is a schema error).
- `bits` is a list of `[index, probability]` pairs: `index` below 4096,
  `probability` in `(0, 1]`.
- `threshold` in `(0, 1]` (default 0.1): entries below it are dropped before
  token selection.
- The fingerprint must be computed or predicted outside this crate: the Rust
  crate cannot compute this fingerprint itself, so it is always an input
  (true bits exported by `tools/ms2/export_fingerprints_mist.py` for
  training, predicted probabilities at inference).

The model selects at most its `fingerprint_slots` entries with the highest
probability (ties by lower index) as tokens. The response echoes
`"fingerprint": {"name": "morgan4096", "tokens_used": N,
"entries_dropped": M}`. A model without a fingerprint encoder given a
fingerprint returns `unsupported_input` naming `"fingerprint"`.
`input_hash` covers the field. An empty fingerprint (no entries at or above
the threshold) contributes an exact-zero pooled vector, so it conditions on
nothing. MIST-like synthetic training/evaluation noise comes in two variants
recorded in the experiment report: per-spectrum histograms (every spectrum's
prediction is one row) and per-molecule-averaged histograms (one averaged row
per molecule, the averaged-panel setting), selected with `--fp-noise-level
spectrum|molecule` (default `spectrum`).

### Mass input (`target_mass`)

The request carries exactly one of `composition` (as above) or
`target_mass`; both or neither is a schema error naming the rule.
`mass_role` stays `"target_molecule"`.

```json
"target_mass": {"units": "microdalton", "value": 46041865, "ppm_tenths": 100, "uncertainty_uda": 50, "source": "synthetic"},
"neutralization": "already_neutral",
"formula_search": {"hypotheses": 8, "nodes_visited_max": 2000000}
```

or, for an ion, `"neutralization": {"precursor_ion": {"adduct": "[M+H]+"}}`
where `value` is then the precursor m/z in micro-m/z and the adduct is one
of `chem::ADDUCTS` by name (the neutral mass and its error then come from
the existing precursor path of `enumerate`).

- `target_mass.units` must be `"microdalton"`; `value` fits `u32`;
  `ppm_tenths` 0..=1000; `uncertainty_uda` is required but may be `null`
  (unknown precision); `source` is a non-empty string. The integer
  `4294967295` (`u32::MAX`, the unknown-precision sentinel) is a schema
  error: unknown precision must be spelled `null`.
- `formula_search.hypotheses` is 1..=32 (default 8);
  `nodes_visited_max` is 1..=50,000,000 (default 2,000,000);
  `pruning` is `"train_fit"` (default: exact chemical filters plus the
  checkpoint's train-fit `RatioBounds` and domain caps) or
  `"chemical_only"` (exact filters only: hydrogen ceiling, parity, DBE;
  element caps only from the model's `max_atoms`; no `RatioBounds`);
  `allocation` is `"equal"` (default: even trajectory split with the
  remainder redistributed) or `"train_frequency"` (largest-remainder split
  by training-frequency prior weights).
- `uncertainty_uda: null` (unknown precision) is accepted and yields a
  response with `mass_evidence.status = "unavailable"`, no search and no
  candidates: never an exception and never a claim that no molecule exists.
- A checkpoint without formula artifacts gives `status: "unsupported_input"`
  with `unsupported.limit = "formula_artifacts"`.

Processing, in order, each stage counted in the response (`formula_search.stages`
names every stage with its class: `necessary` for the mass verdict, the exact
chemical filters, the model domain, the substructure bound and
completability; `empirical` for the train-fit DFS pruning and the train-fit
bounds; `budget` for the first-`hypotheses` selection and the trajectory
allocation). All stage counts come from the one primary traversal, except
`joined_chemical` (a diagnostic rerun, see below):

a. Enumerate under the requested `pruning` with every exact chemical filter
   on (hydrogen ceiling, parity, DBE `>= 0`). Under `"train_fit"` (default)
   the checkpoint's `RatioBounds` and domain caps additionally apply; under
   `"chemical_only"` element caps come from the model's `max_atoms` alone.
   The neutral path uses the neutral-composition error bound plus the
   observation uncertainty only (no adduct conversion, so no `+1`); the
   precursor path keeps the existing `uncertainty + 1` bound (the `+1` is the
   adduct conversion's rounding bound, counted once) with the ppm tolerance
   at the observed precursor m/z. A precursor whose neutralisation leaves
   the `u32` range is `mass_overflow`: the enumerator's status propagates,
   never a search around a substituted zero mass.
b. Keep compositions the model can hold (heavy atoms <= `max_atoms`).
c. Substructure lower bound (necessary for containment under the active
   `substructure_semantics`): for every supplied pattern, per element its
   heavy-atom count must not exceed the composition's, and the pattern's
   hydrogens (sum over its atom types) must not exceed the composition's
   hydrogens. Under `"disjoint_occurrences"` and
   `"complete_functional_groups"` the bound sums over all patterns
   instead of checking each pattern on its own; under
   `"complete_functional_groups"` the summed heteroatom counts must
   additionally equal the composition's. This prunes formula hypotheses
   strongly in the disjoint modes.
d. Completability pre-check: `TraceState::new_exact(limits, composition)`
   `.feasibility().all()` on the empty state.
e. Order the survivors by verdict (`accepted` before `boundary_ambiguous`),
   then absolute mass residual ascending, then the enumerator's canonical
   order; ambiguous rows are never dropped. Select the first `hypotheses`.
   This order is a deterministic default, not a formula probability. If the
   enumerator reported `exhausted`, `formula_search.status =
   "search_exhausted"` (a nearer formula may remain unvisited) even when
   the selection also truncated; else if more survive than are selected,
   `"truncated"`; else `"complete"`. Both facts stay visible as the
   independent booleans `formula_search.truncated` and
   `formula_search.search_exhausted`. The primary enumerator capacity is
   unbounded on this path so it cannot bind before selection: a bound
   capacity would report `search_exhausted`, never a silently wrong
   `"complete"`.
f. Generation: the request's `generation.trajectories` is the total budget,
   assigned over the selected formulas by largest-remainder rounding of the
   allocation weights (uniform under `"equal"`, so the even split's remainder
   is redistributed one each to the first formulas; `(count + 1) /
   sum(count + 1)` over the selected formulas with `alpha = 1` under
   `"train_frequency"`, where `count` is the checkpoint's training count of
   the formula text and the weights are a training-frequency prior, not a
   calibrated probability), with at least one trajectory per selected
   formula while the total allows; when `total < selected`, only the first
   `total` formulas get one trajectory each and the rest are reported as
   `not_sampled`. Formulas sharing one assigned trajectory count generate in
   one batched call with one `CompletionRequest` per sampled formula (ids
   derived from the request id and the formula's canonical text, all
   distinct). Outcomes are indexed by selected position, so pooled
   candidates always carry their own source formula's mass metadata and
   ranking weight (and every pooled candidate's graph composition is
   checked against its source formula, as an error, not an assert). The
   assigned sum is exactly the total whenever anything is selected: a
   fixed budget is spent, and `accounting.unused_trajectories`
   (`requested - executed`) is 0 then. When formulas joined but nothing
   (or not everything) sampled, `formula_search.unsampled_reason` names the
   first stage that left zero rows (`by_train_fit`,
   `all_excluded_by_domain`, `by_substructures`, `by_completability`) or
   `budget` when the total did not cover the selection; it is `null` when
   every selected formula sampled. Top-level `status` stays `no_candidates`
   in those cases.
g. Pool the accepted identities of all formulas; rank by `samples`
   descending, then `best_log_prob` descending, then formula order, then
   trace order under `"equal"`; under `"train_frequency"` rank by the
   explicit estimate `weight * samples / trajectories` first (ties as in the
   equal order); cut to `returned`. Different formulas can never be the same
   identity.

The diagnostic chemical-only rerun behind `joined_chemical` (what the exact
filters alone admit) has its own bounded row capacity (200,001 rows) and its
own node budget (at most 2,000,000 nodes, capped by what the primary left):
when its own limit binds, `joined_chemical` is `null` with a reason naming
the limit instead of an unbounded second search.

Formulas carry no learned prior under `"equal"` and only the explicit
training-frequency prior under `"train_frequency"`. There is no learned
formula ranker.

### Response

```json
{
  "protocol": "molecular-completion-generate-v1",
  "query_id": "...", "provenance": "...", "input_hash": "<sha256 of the canonical (key-sorted) request JSON>",
  "status": "ok | no_candidates | unsupported_input",
  "substructure_semantics": {"value": "contained | disjoint_occurrences | complete_functional_groups", "meaning": "<one line>"},
  "model": {"version": "...", "grammar": "...", "chemistry": "...", "checkpoint_sha256": "...", "max_atoms": 32, "max_ring_closures": 6},
  "candidates": [{"rank": 1, "atoms": [...], "bonds": [[a, b, order], ...], "composition": "C2H6O", "samples": 41, "sample_fraction": 0.640625, "best_log_prob": -1.25}],
  "unresolved": 0,
  "accounting": {"requested_trajectories": 64, "trajectories": 64, "finished": 60, "dead_end": 4, "truncated": 0, "other_status": 0, "rejected_replay": 0, "rejected_containment": 3, "containment_unresolved": 0, "identity_unresolved": 0, "distinct": 5},
  "unsupported": {"limit": "max_atoms", "allowed": 12, "observed": 13},
  "infeasible": {"reason": "..."},
  "ranking": {"status": "sample_frequency", "calibrated": false, "reason": "..."},
  "mass_evidence": {"status": "not_evaluated", "reason": "composition was supplied, no mass was evaluated"},
  "search": {"status": "sampled", "exhaustive": false},
  "physical_verification": {"status": "not_evaluated", "reason": "..."},
  "stereochemistry": "unspecified"
}
```

- `accounting.requested_trajectories` is always the request's
  `generation.trajectories`; `accounting.trajectories` is the executed
  count (equal on `ok` / `no_candidates`, 0 on `unsupported_input`,
  where every other counter is 0 too).
- `unsupported` is present only on `unsupported_input` and names the
  violated limit: `limit` is `max_atoms`, `composition_heavy_atoms` or
  `max_ring_closures`, `allowed` the limit's value, `observed` the
  observed value.
- `infeasible` is present only when the composition pre-check fails (see
  above): the request is well-formed but no molecule with its composition
  can satisfy the active semantics, so nothing was sampled (`trajectories`
  0) and the status is `no_candidates`.
- On `unsupported_input` nothing is sampled, so `search` and `ranking`
  are `{"status": "not_evaluated", "reason": ...}` instead of the
  sampled / sample-frequency bodies above.
- `best_log_prob` is the trace log-probability at temperature 1,
  whatever the request's sampling temperature.
- `input_hash` is the SHA-256 of the canonical request JSON: object
  keys sorted recursively and explicitly (independent of any JSON
  library map-ordering feature), whitespace-free. Reordering keys or
  changing whitespace leaves it unchanged; changing any value changes
  it. It covers the new mass fields too.

Mass response additions: every candidate gains
`"mass": {"formula": "C2H6O", "computed_uda": 46041865, "residual_uda": 0,
"status": "accepted" | "boundary_ambiguous"}`; top level
`"mass_evidence": {"status": "accepted" | "boundary_ambiguous" | "rejected" |
"search_incomplete" | "mass_overflow" | "unavailable" | "not_evaluated",
...}` — derived only from the formula search, never from whether sampling
happened: `accepted` when the search joined at least one `Accept` row;
`boundary_ambiguous` when it joined only `Ambiguous` rows; `rejected` when
the search completed and no formula passed the mass verdict (under
`train_fit` pruning this additionally requires the chemical-only rerun to
complete empty — train-fit-only exclusions are `unsampled_reason:
"by_train_fit"`, never mass rejection); `search_incomplete` when a node
budget or capacity bound stopped the search (primary or the disambiguating
rerun), so absence proves nothing; `mass_overflow` when precursor
neutralisation left the `u32` range; `unavailable` for unknown precision
(`null` uncertainty, no search); `not_evaluated` only for composition
requests — with the named §5 error parts under
`mass_evidence.error_terms_uda` (`observation_uda` the supplied
uncertainty, `composition_uda` the largest per-row composition term over the
joined rows, `neutralisation_uda` 1 on the precursor path for the adduct
conversion and 0 for already-neutral masses); and
`"formula_search": {"status", "truncated", "search_exhausted",
"unsampled_reason", "pruning", "allocation", "joined",
"joined_chemical", "joined_chemical_reason", "excluded_by_train_fit",
"after_domain", "after_substructures", "after_completability", "selected",
"sampled", "ranking", "stages", "formulas":
[{"formula", "computed_uda", "residual_uda", "verdict", "mass_status",
"weight", "trajectories", "finished", "accepted_candidates", "sampled",
"sampling", "stage"}], "enumerator": {"nodes_visited",
"hydrogen_checks", "rows_joined", "rejected_*" counters, "exhausted"}}`.
`status` is `"search_exhausted"` whenever the enumerator exhausted (even
when the selection also truncated), else `"truncated"`, else `"complete"`
(`"unavailable"` / `"mass_overflow"` when no search ran); `truncated` and
`search_exhausted` repeat the two facts as independent booleans.
`unsampled_reason` is `null` when every selected formula sampled, else the
first stage that left zero rows (`by_train_fit`,
`all_excluded_by_domain`, `by_substructures`, `by_completability`) or
`budget` when the total did not cover the selection.
`joined_chemical` is what the exact filters alone admit (the chemical-only
rerun with its own bounded capacity and node budget; `null` with a reason
in `joined_chemical_reason` when the primary already joined above the rerun
row cap, the node budget is spent, or the rerun binds its own limit);
`excluded_by_train_fit` is the difference (`joined_chemical - joined`);
`stages` names every stage with its `necessary` / `empirical` / `budget`
class and entering/leaving counts — `exact_chemical_filters` leaves the
within-traversal count (verdict-passing minus exact rejects),
`train_fit_dfs_pruning` is the DFS empirical-pruning line (no rows change
across it), `train_fit_bounds` runs from there to `joined`;
`formulas[i].stage` is `sampled` | `not_sampled`, `weight` the selection
weight (uniform under `"equal"`, the training-frequency prior under
`"train_frequency"`), `mass_status` the source verdict.
`accounting` sums over the formulas, keeps `requested_trajectories`, and
reports `unused_trajectories` (`requested - executed`; 0 whenever anything
is selected, since the assigned sum is exactly the total).

Top-level `ranking` names the rule actually used: `sample_frequency`
under `"equal"`, `train_frequency_weighted_estimate` (with the
`weight * samples / trajectories` estimate in the reason) under
`"train_frequency"`; both uncalibrated (`calibrated: false`).

The numeric request id passed to `CompletionModel::generate` is the first
8 bytes of the SHA-256 of the `id` string.

### Stereo enumeration (`stereo-perception-v2`)

Scope, stated honestly: the training data has no stereo labels and the
model's evidence (a formula and substructures) carries none, so nothing here
predicts a stereoisomer. Every candidate is made stereo-aware instead: which
stereo elements its graph supports, how many distinct stereoisomers they
generate, and each of them on request, under a convention an external tool
can turn into isomeric SMILES. `MolGraph` identity stays constitutional;
ranking is unchanged (constitutional).

Request (optional):

```json
"stereo": {"expand": 2, "max_elements": 10}
```

- `expand` in `0..=64` (default 0): how many canonical stereoisomers each
  candidate carries under `stereoisomers`.
- `max_elements` in `1..=12` (default 10): most potential stereo elements
  analysed before the candidate reports `unresolved: too_many_elements`.
  Inside `perceive` the limit is clamped to an absolute cap of 16 with
  checked shifts and allocation sizes, so an excessive caller limit reports
  `unresolved` instead of panicking or over-allocating.
- Unknown fields are schema errors; `input_hash` covers `stereo`.

Each candidate gains:

```json
"stereo": {"version": "stereo-perception-v2", "assignment": "unspecified",
           "tetrahedral_centers": [{"atom": 2, "ligands": [0, 3, 5, "H"]}],
           "double_bonds": [{"atoms": [1, 4], "reference": [0, "lone_pair"]}],
           "not_stereogenic": 1, "unsupported": [], "raw_assignments": 4,
           "distinct_stereoisomers": 3, "resolution": "resolved",
           "molecule_wide_exact": true,
           "stereoisomers": [{"tetrahedral": ["cw", "ccw"], "double_bonds": []}],
           "stereoisomers_truncated": false}
```

- `tetrahedral_centers` / `double_bonds` hold the stereogenic elements
  only, in assignment order; `not_stereogenic` counts the potential elements
  dropped from the lists. Ligands are heavy-atom indices, `"H"` for hydrogen
  and `"lone_pair"` for a nitrogen lone pair.
- `raw_assignments` is `2^k` for `k` potential elements, `distinct` the
  exact number of distinct stereoisomers within the supported kinds; both
  are `null` when unresolved, with
  `"resolution": "unresolved: <reason>"` (`too_many_elements`,
  `too_many_automorphisms`, `work_limit_exceeded`).
- `stereoisomers` is present only when `expand > 0`: canonical
  representatives (lexicographically smallest assignment **vector** of each
  orbit — first element most significant — values `"cw"` / `"ccw"` and
  `"cis"` / `"trans"` aligned with the two element lists), at most `expand`
  entries; `stereoisomers_truncated` says whether more orbits exist.
- `molecule_wide_exact` is true iff the computation resolved **and** no
  unmodelled kind from the list below is present: any molecule that may
  carry stereo of an unmodelled kind is never reported molecule-wide exact.

Convention (implemented exactly; `tools/ms2/completion_stereo_check.py`
maps it to RDKit):

- Tetrahedral: ligand list `L` is the centre's heavy neighbours in ascending
  atom index, then `H` when the centre has a hydrogen. `cw` (1) means:
  looking from `L[0]` toward the centre, `L[1] -> L[2] -> L[3]` runs
  clockwise; `ccw` (0) the opposite.
- Double bond `(a, b)` with `a < b`: each end's reference ligand is its
  lowest-index heavy substituent, else its hydrogen, else (nitrogen) its
  lone pair. `cis` (0): the two reference ligands are on the same side;
  `trans` (1): opposite sides.
- An assignment is one value per potential element, in element order:
  tetrahedral centres by atom index, then double bonds by `(a, b)`.

Potential elements: tetrahedral carbons with total coordination four (heavy
neighbours plus hydrogens) and at most one hydrogen, every bond single;
C=C, C=N and N=N bonds of order 2 whose ends each carry two single heavy
substituents, one heavy plus one hydrogen, or (nitrogen) one heavy or one
hydrogen plus the lone pair — excluding ends with two hydrogens, cumulene
ends, and bonds whose smallest ring has fewer than 8 atoms. A ring double
bond whose smallest ring has 8 or more atoms counts only when it is
isolated in every ring through it: neither end may have a ring neighbour
(other than its partner, in the same ring system — ring systems come from
ring-bond connectivity, so the verdict never depends on atom numbering)
that carries a double bond or is a lone-pair donor (N, O, S with only
single bonds). Stereogenicity and the distinct count are exact: orbits of
all `2^k` assignments under the constitutional graph's automorphisms (at
most 20,000 automorphisms and a 100,000-unit work limit; exceeding either
is `unresolved`, never a guess).

Not modelled, reported per candidate under `unsupported` (so the count is
never misread as molecule-wide truth when it is not): `axial_cumulene` (a
carbon or nitrogen with two double bonds), `conjugated_large_ring` (a
non-isolated double bond in a ring of 8 or more, including conjugation
through lone-pair donors: possible aromaticity), `constrained_nitrogen_center`
(a nitrogen with three single bonds in a 3- or 4-membered ring or at a
bicyclic bridgehead — flagged as uncertain, never counted),
`atropisomer_axis_possible` (a single bond between two sp2 ring atoms in
different rings with at least three substituted ortho positions — a
candidate hindered axis), `phosphorus_center` (any phosphorus with three or
more heavy neighbours), `sulfur_center` (any sulfur with three or more heavy
neighbours). Amines are never centres (inversion, not stereo).
Atropisomerism (hindered rotation) and conformational stereo (ring puckers,
rotamers) are otherwise out of scope entirely. Known limitation (reported by
`tools/ms2/completion_stereo_check.py` on real molecules, never hidden):
isomeric SMILES cannot express a centre whose ligands RDKit's own symmetry
perception conflates, so the SMILES sets may differ there even when the
exact count is right (see the tool's disagreement analysis).

Top level, `stereochemistry` is now
`{"status": "enumerated_not_predicted", "reason": "stereo elements and
distinct stereoisomers are derived from each candidate graph; the model
assigns no preference among them; kinds listed under a candidate's
`unsupported` are not modelled"}`.

### Fixture provenance

The committed fixture (`tests/fixtures/ms2/completion_tiny.*`) is defined
on the CPU backend and was reproduced byte-identically on the machine that
generated it; cross-machine byte identity is not claimed.

### Scope

Supplied exact composition; substructures with parent hydrogen counts;
sampled, not exhaustive; ranking uncalibrated; no mass, precursor or
physical evidence.

## Optional physical verification (`physical-verification-v1`)

A separate Python tool, `tools/ms2/completion_physical_verify.py`, reads a
generation response and adds force-field geometry checks and structural alerts
per candidate and per expanded stereoisomer. It depends on RDKit and is not part
of the wheel. It reports optimisation convergence and alerts; it does not
evaluate stability. Statuses, the alert inventory and usage are documented in
[`tools/ms2/COMPLETION_PHYSICAL_VERIFY.md`](../tools/ms2/COMPLETION_PHYSICAL_VERIFY.md).
