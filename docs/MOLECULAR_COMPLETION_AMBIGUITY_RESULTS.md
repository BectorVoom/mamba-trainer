# Bounded molecular-completion ambiguity audit: measured results

Run date: 2026-10-04. This completes the bounded reference audit in
[MOLECULAR_COMPLETION_EXPERIMENT.md](MOLECULAR_COMPLETION_EXPERIMENT.md).
The broader corpus experiment and precursor comparison remain pending.

## Result

Of 26 synthetic queries, 21 completed with resolved mass evidence (80.8%).
Two queries have unresolved mass evidence, one has incompatible oracle input,
and two exhausted graph-extension budgets. Exact counts exclude these five queries.

| Exact compatible graphs | Completed queries |
|---:|---:|
| 0 | 2 |
| 1 | 10 |
| 2 | 5 |
| 3 | 3 |
| 5 | 1 |

The full-domain C5H12 and C6H14 searches retained 3 and 5 unique graphs,
respectively. These are lower bounds; neither search proves completeness.
All supplied in-domain references in completed searches were recovered.
A deliberately unrelated reference is retained as a negative recovery control.

## Implementation and scope

OpenCode implemented the Rust enumerator, Python binding, independent audit
and initial tests. Local review finished validation and reporting after the
OpenCode provider returned HTTP 429 (endpoint unavailable).

The Rust search starts at START under an exact formula budget and uses existing
TraceState transitions. Only connected STOP graphs with exhausted composition
and zero residual valence enter matching. Existing subgraph STOP behavior remains
available. Typed embeddings are injective within each pattern; unknown overlap
allows different patterns to share target atoms. The complete correspondence
oracle enforces both shared classes and distinct classes. An empty supplied
relation means known disjoint atoms, while an absent relation means unknown overlap.
Canonical BFS traces deduplicate graphs independently of traversal and embedding.

The domain is supported neutral V0 C/N/O types, two to six heavy atoms and at
most one independent cycle. Stereo, other elements and charged chemistry are
outside this audit. These counts describe the graph representation and mass
arithmetic; they make no claim about physical stability or a trained model.
Synthetic neutral masses use integer arithmetic and explicit precision metadata.
Ambiguous or missing precision cannot certify zero.

The manually authored manifest includes chain, branch, unsaturation, ring,
nitrogen, symmetry, overlapping/disjoint constraints, inconsistent constraints,
mass rejection and uncertain mass. Pinned typed graph sets independently verify
ethanol/dimethyl ether and ethylamine/dimethylamine. The two-methyl pair shows
the oracle distinction: unknown overlap retains 2 C2H6O graphs; known disjoint
atoms retain only dimethyl ether (1). Reference metadata does not constrain search.

## Per-query results

| Fixture | Status | Mass | Graphs | Recovery | Termination |
|---|---|---|---:|---|---|
| c2h6o_mass_only | complete | accepted | 2 | true | none |
| c2h6o_ch2oh_bond | complete | accepted | 1 | true | none |
| c2h6o_ch3_symmetry | complete | accepted | 2 | unknown / not supplied | none |
| c2h6o_incompatible_n_atom | complete | accepted | 0 | unknown / not supplied | none |
| c3h8o_mass_only | complete | accepted | 3 | true | none |
| c3h8o_other_target_not_recovered | complete | accepted | 3 | false | none |
| c3h8o_disjoint_ch3_and_ch2oh | complete | accepted | 1 | true | none |
| c2h7n_mass_only | complete | accepted | 2 | true | none |
| c2h7n_nh2_selects_ethylamine | complete | accepted | 1 | true | none |
| c3h4_mass_only_unsaturation | complete | accepted | 3 | true | none |
| c3h4_triple_bond_selects_propyne | complete | accepted | 1 | true | none |
| c3h6_ring_cyclopropane | complete | accepted | 1 | true | none |
| c3h6_disjoint_double_and_methyl | complete | accepted | 1 | true | none |
| c4h8_mass_only | complete | accepted | 5 | true | none |
| c4h8_square_ring_selects_cyclobutane | complete | accepted | 1 | true | none |
| c2h6o_oracle_overlap_shares_ch2 | complete | accepted | 1 | true | none |
| oracle_requires_incompatible_types | unsupported_input | not_evaluated | 0 | unknown / not supplied | unsupported: unsupported_input: correspondence pairs incompatible shared atom types |
| c2h6o_mass_far_off | complete | rejected | 0 | unknown / not supplied | none |
| c2h6o_mass_boundary_ambiguous | mass_evidence_unresolved | ambiguous | 0 | unknown / not supplied | none |
| c2h6o_precision_unavailable | mass_evidence_unresolved | unavailable | 0 | unknown / not supplied | none |
| c2h6o_overlap_unknown | complete | accepted | 1 | true | none |
| c2h6o_full_domain_cn_o | complete | accepted | 2 | unknown / not supplied | none |
| c5h12_full_domain_cn_o | search_budget_exhausted | accepted | 3 (lower bound) | unknown / not supplied | graph_extension_limit |
| c6h14_full_domain_cn_o | search_budget_exhausted | accepted | 5 (lower bound) | unknown / not supplied | graph_extension_limit |
| c2h6o_two_methyls_unknown_overlap | complete | accepted | 2 | true | none |
| c2h6o_two_methyls_known_disjoint | complete | accepted | 1 | true | none |

Every JSON report records input hash, domain, seed, accepted and ambiguous
formulas, identities, unresolved hypotheses, recovery, elapsed time, cumulative
work counters and termination reasons. Recovery is unknown when its own work
cannot complete within the shared resource budget.

## Validation

- Final completion integration tests: 34 on CPU and 34 with wgpu features.
- Internal completion tests: 7 on CPU.
- Rebuilt and installed CPU Python wheel: 7 API tests.
- Independent Python enumerator: all 24 affordable fixtures match semantic
  expectations; the two 5/6-atom queries are explicitly outside its scope.
- Earlier chemistry, contract, decoder and formula regression suites passed
  on CPU and GPU. The GPU run also exercises shared grammar replay and mass
  twin parity on Apple M1 / Metal. The reference search itself runs on the host.

Final commands, exit codes and logs are in `final_validation.json`; earlier
regressions and recovered environment failures are in `supervisor_validation.json`.
The initial CPU run encountered a corrupt non-executable build artifact and then
disk exhaustion. Rebuilding the artifact and clearing disposable incremental
caches resolved both. An intermediate test compile failure from using Clone on
non-Clone graph queries was corrected. Failed logs remain alongside passing runs.

The independent Python audit uses type multisets, edge assignments and permutation
identities rather than the Rust trace enumerator. Parity compares semantic fields,
not work counters. The PyO3 API invokes the authoritative Rust implementation.

## Resources and reproduction

Per-query defaults: 100,000 formula visits, graph extensions, embedding nodes
and cumulative canonical expansions; 10,000 retained identities; 30-second
watchdog; 64 MiB modeled frontier plus retained identity storage. Input caps
are eight patterns, 24 total pattern atoms and 16 correspondence pairs.
The modeled memory bound excludes temporary, parser and caller allocations.
It is distinct from whole-process RSS and is not an allocator-level RSS cap.
Completed-candidate counters count STOP candidates examined, including those
rejected by exact composition or residual-valence checks.

The final release run took 1.581 seconds and peaked at 7,585,792 bytes
(7.23 MiB) whole-process RSS, measured by `/usr/bin/time -l`
after compilation. This is a bounded audit measurement, not an optimization
or throughput claim. No performance optimization was performed.

```sh
cargo build --release --no-default-features --features cpu --example ms2_completion
target/release/examples/ms2_completion \
  --fixtures experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json \
  --out /tmp/completion_report.json --seed 1
python3 tools/ms2_completion_audit.py \
  experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json
```

Artifacts: `experiments/molecular_completion/20261004_completion_ambiguity/`.
`completion_report.json` is the authoritative final report; the supervisor copy
contains identical bytes. `supervisor_distribution.json` separates exact counts
from censored lower bounds. `supervisor_experiment_metadata.json` records source
SHA-256 hashes, fixture hash, commands, platform, Rust version and repository HEAD.
The recorded HEAD identifies the base checkout; these changes are uncommitted.

| Artifact | SHA-256 |
|---|---|
| fixtures.json | `c6da25873818c3b4e6f0bb98fdf21bf5eb5ec1e495dae159797775d5b15d56a2` |
| completion_report.json | `43f39c6b4dcd9b3133f4a52c75644c70bfbeb34cc418ce6a3038c3e66385bcf9` |
| python_audit_report.json | `9441a4d921c85a0eff4105b79af82d86d938604f85ca09efce3ce3f526226a82` |

## Next decision

Precursor evidence is `not_evaluated`: no independently validated compatible
parent/target pairs were provided. No training run belongs to this audit.
Use the completed small-domain fixtures for further supervised pilot work.
Before drawing exact full-domain 5/6-atom conclusions, increase measured search
budgets or narrow the domain. A larger model does not resolve this search
censoring; remaining ambiguity motivates ranking, calibrated abstention or
additional evidence.
