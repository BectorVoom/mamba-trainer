# Independent Codex review — 2026-10-03

Implementation agent: OpenCode `opencode/muse-spark-1.3-contributor-free`, session `ses_effe7da48ffeggjEiOEqsfmqCj`.

Final verdict: accepted for the bounded reference-spectrum experiment. No serious unresolved code findings after repair and second review.

Initial review reproduced unenforced connectivity split overlap, invalid-query export/denominator failures, and an unsupported positive-control inference. OpenCode repaired these with regression tests, added duplicate-content guards and coverage-only comparison, and corrected costs, sensitivity denominators and baseline labeling. Supervisor made final wording corrections: equal target ranks do not prove identical candidate ordering; approximate identity coverage does not establish general connectivity coverage.

Independent verification: all seven molecular-completion Python suites passed (90 tests, exit 0); bounded 200-validation/50-training-control experiment rerun exited 0. Both repaired CSVs are byte identical to the implementation agent outputs; deterministic summary content matches after excluding timings and output paths. All pre-existing module/report hashes unchanged. See verification_checks.json and verification_repaired/.

Validation: 200/200 targets in formula pools; 10/200 queries have any referenced candidate, 0/200 targets have references. Spectral top-1 4%, equal aggregate uniform result; spectral predictions 10, correct 0. This measures sparse reference availability in a structure-disjoint validation sample, not the ability of a learned spectral model to generalize.

Control: among 15 training-query pools with multiple reference-backed candidates, cosine top-1 15/15 versus availability-only 4/15. This is a narrow pipeline check, not validation performance. Duplicate query spectra are excluded by content as well as identifier.

Limitations: first-in-file biased sample, provider SMILES identity for candidate references, no calibrated thresholds, no learned predictor. CPU Python implementation only; GPU and Rust parity not applicable. Unit test output contains harmless ResourceWarnings from temporary fixture readers and prints a fixture summary; these do not affect recorded results.

Next research step: a small trained spectrum-to-fingerprint predictor with training-only model selection and identical formula-pool evaluation. This has not been implemented in this run.

Verification command:
`python3 -m unittest tools.ms2_completion_ambiguity tools.ms2_database_retrieval tools.ms2_msgym_corpus tools.ms2_chebi_corpus tools.ms2_rank_abstain tools.ms2_oop_enumerate tools.ms2_spectral_rank`

Rerun command:
`python3 tools/ms2_spectral_rank.py --max-queries 200 --max-control 50 --out-dir experiments/molecular_completion/20261003_spectral/verification_repaired`
