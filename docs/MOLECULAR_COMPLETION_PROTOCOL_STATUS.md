# Remaining protocol: status matrix (2026-10-03, second-review repairs)

Source of truth: measured outputs in `experiments/molecular_completion/20261003_remaining/`
(`quality_summary.json`, `qm9_summary.json`, `external_summary.json`,
`EXTERNAL_PIN.json` in `data/pinned/`). Nothing below is claimed without an artifact.
Earlier reviewed experiments/modules are preserved and unchanged in behavior.

| Item | Status | Evidence |
|---|---|---|
| A. MassBank external validation (150 fixed queries, frozen checkpoint) | **completed** | prefix recall 54/150, predictor top25 21/150 (21/54 present); ChEBI nested S1 76/150, S2\|S1 76/76, S3\|S2 76/76; `external_rows.csv`, `chebi_stage_rows.csv` |
| B. GNPS second check (126-block MGF) | **completed (limited)** | 124/126 `no_structure` counted; 2 structurized scored 0/2; whole-token adduct + charge + count + precision enforcement; per-row provenance |
| C. QM9 restricted control (subset + full 133,885, exact isomorphism) | **completed** | eligible 239; allowed exact 81; excluded 60; GDB-only 1; no-match 97 (reproduces independent replay); stationary-point members 80; linearity-validated counts |
| D. Quality gates (denominators, full cost, exact scaffolds) | **completed** | eligible = complete+present+parseable+valid ranks (156); qid-matched pairs; Murcko gate COMPLETE with versioned policy |
| E. Official MassSpecGym baseline run | **blocked (concrete)** | no retrieval checkpoint (releases: zero assets; HF: simulation .ckpt only); cached massspecgym==1.3.1 hashed; import fails at pulp-solver IndexError, no binaries, no downloads |
| F. PubChem conditional expansion | **completed (limited); 4 selected formulas unavailable** | 45 canonical-unresolved misses → first 5 formulas in fixed order; 1 cached formula evaluates 1 query (0 recovered), 4 caches unavailable; 0 API calls, truncated lists, no absence proof or frozen rescoring |
| G. Confidence/subgroup adequacy (frozen 200, Wilson + paired) | **completed** | CIs on every rate incl. stage presence (all-selected + eligible); no expansion, no tuning |
| val200 library lookup (MassBank/GNPS coverage) | **completed** | LIBRARYLOOKUP only; `library_lookup.csv` |
| Reference-NN arm (redesigned) | **completed** | separate references (3,000 MB + 126 GNPS); eligibility 112/152; conditional 51/112; `external_nn_rows.csv` |
| Ingress budget | **not verified; prior dependency attempt exceeded scratch quota** | Prior CUDA dependency download bytes are unknown/exceeded; no further downloads; final reruns use cached inputs and 0 API calls |
| Tests (8 preexisting + 3 repaired modules) | **completed** | 212 tests, all 11 suites exit 0; plus 7 supervisor regressions; `test_exits.json` |

Conditional-not-triggered: none (the one conditional, PubChem, triggered on measured
canonical-unresolved misses and was attempted cache-only within caps; four
selected formulas remain unavailable).

Blocked: official-baseline execution and further PubChem coverage of uncached
formulas (no further downloads authorized within the remaining budget). No
retrieval checkpoint was found in the inspected official locations. The official
stack cannot import without an LP solver binary. Prior pip-incident ingress is
reported as unknown/exceeded, never <1GB. Stage caps met (QM9 68 s, external 150 s wall).
