# Independent Codex review — trained spectral predictor, 2026-10-03

Implementation: OpenCode `opencode/muse-spark-1.3-contributor-free`. Initial implementation session `ses_effd4bd6dffeN77v1tzGx5g254`; separate repair session `ses_effa5e2dbffe2ZtUc8ck4GCHiv`.

Verdict: accepted as a bounded exploratory CPU baseline; no serious unresolved code findings after repairs and second review. Results do not establish deployment readiness or general spectral-model performance.

Initial review reproduced wrong target-parseable eligibility, missing validation connectivity-identity handling, capability gates accepting unusable predictions, signed-score confidence margin mismatch, and misleading persisted cost scope. Repairs added regressions, shared target-independent inference capability, finite-score margins, retained identity-excluded rows, proper evaluation subgroups, output-inclusive timing, and strict JSON null for abstain-all thresholds.

Independent verification: all eight molecular-completion suites pass (121 tests, exit 0). The independent 5000-capped-training/200-validation run exits 0. All three CSVs are byte identical, model checkpoint arrays exactly match without pickle, and deterministic summaries match after excluding timings/output paths. All pre-existing source/report hashes unchanged. See verification_checks.json and verification/.

All 200 targets remain in the original full formula pools. Predictor top-1 12/200 vs shuffled/prior 8/200; 156 target-parseable queries have predictor8/156 vs shuffled/prior4/156. 188 inference-capable queries and12 unavailable model queries are separately accounted. Training calibration uses897 held-out molecules with candidate pools and fails to reach90% precision, so the gate abstains on all200 validation queries. True-query graph data is evaluation-only; fitting uses3269 train molecules, with1036 calibration molecules and molecule/content separation.

The report was corrected after review to distinguish descriptive paired-rank changes from statistical conclusions, avoid attributing peak RAM without a profile, and avoid claiming pool-coverage failure where pool recall is100%. Small-sample training, graph parser domain and fingerprint collisions remain material limits. CPU scientific-Python baseline has no GPU or Rust implementation to test.

Follow-up direction: controlled improvement of chemistry coverage, fingerprints or trained spectral model with train-only selection and independent confirmation; this experiment does not decide which change will work. No next experiment has been started and no changes committed.

Verification commands and exit codes: see verification_tests.stderr.log, verification.stdout.log, verification.stderr.log; experiment command matches RUN_COMMANDS.txt with output directory verification/.
