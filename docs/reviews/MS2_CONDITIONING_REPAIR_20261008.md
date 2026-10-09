# Predicted-fingerprint generation repair — 2026-10-08

**Status: training code repaired; molecular identification remains inadequate.**
The predeclared 6,000-step adaptation still recovers zero targets. An exploratory
follow-up using the earlier checkpoint and one mass-derived formula finds one
of 300 targets; fingerprint reranking places it seventh. This is not a
competition-ready model or evidence of reliable de novo identification.

The source report and `conditioning_audit_20261007/summary.json` describe an
oracle-trained decoder evaluated with incomplete predicted fingerprints.
Exact fingerprints train confidence bucket 8 only. Real MIST uses mostly
buckets 1–7, and its bit membership differs substantially from complete true
fingerprints. The audit already showed that binarising probabilities alone
does not solve recovery. None of these results proves that MIST's signal is
intrinsically insufficient for a better-trained generator.

## Changes

- `CompletionTrainer::set_learning_rate` validates the rate and updates both
  AdamW and the saved configuration, preserving moments and the step counter.
- The spectral example applies an explicit `--lr` after checkpoint load.
  Omitting it preserves the saved rate; reports record the actual rate.
  Previously the flag was ignored on load while its value was reported.
- Predicted conditioning never substitutes true bits when a prediction is
  missing. Spectral and structure-only examples use empty conditioning in
  that case. Source counts include extra examples; a prediction file with
  zero training coverage fails before training.
- Omitted `--fp-eval` follows `--fp-train`. An explicit oracle control remains
  available. `--eval-offset` separates the NLL monitoring slice from the
  generation assessment slice and rejects an empty selection.

The learning-rate defect and fallback paths are real bugs, but they do not
explain away the old 0/300 result. The audited lineage used exact conditioning;
its saved learning rate was 3e-4. The necessary experiment is training with
real predicted inputs, which was performed here.

## Experiment and results

The rebuilt binary reproduced the original checkpoint's MIST NLL within
1e-4. Adaptation started at step 81,000, used 6,000 updates at learning rate
1e-4, batch 32 and evidence dropout 0.1, without a scaffold or structure-only
extras. Of 192,000 examples before dropout, 191,934 used real predictions and
66 had missing predictions; none used an exact fallback. Training/validation
trace and skeleton overlap were both zero.

NLL monitoring used retained validation molecules 300:600. All validation
molecules have a first-spectrum prediction. On that same monitoring slice,
NLL changed from 107.971 to a minimum of 29.337 at 1,000 updates and then rose
to 32.516 at 6,000 updates. Both checkpoints were preserved. The final-step
checkpoint was the predeclared primary result; the early checkpoint and
concentrated beam were assessed afterward and are exploratory.

Generation and scoring use the audit's independent 300-molecule roster,
including empty lists. The usual search splits width 256 over eight inferred
formulas. The concentrated search spends the same requested width on one
inferred formula: the true formula is selected for 231 queries instead of
268. Six concentrated-search queries return no candidate.

| Arm | NLL/molecule | Target in pool | Top 25, original order | Best Tanimoto in first 25 | Mean strain-flag fraction |
|---|---:|---:|---:|---:|---:|
| Original checkpoint, MIST | 111.688 | 0/300 | 0/300 | 0.131 | 14.49% |
| 6,000 steps, MIST | 32.150 | 0/300 | 0/300 | 0.172 | 2.28% |
| 6,000 steps, fingerprint removed | 31.971 | 0/300 | 0/300 | 0.144 | 1.92% |
| 6,000 steps, shuffled MIST | 35.225 | 0/300 | 0/300 | 0.145 | 2.11% |
| 1,000 steps, MIST | 29.118 | 0/300 | 0/300 | 0.159 | 1.78% |
| 1,000 steps, top inferred formula | 29.118 | 1/300 | 0/300 | 0.203 | 3.25% |

All generated candidates in these arms sanitize and are connected. The
strain flag is the previous audit's limited ring heuristic, not proof of
chemical plausibility. Fixed-length similarity improved: the primary adapted
arm's best top-25 Tanimoto is 0.172 versus 0.131 before adaptation. Against its
matched shuffled control the difference is +0.0269 (paired 95% interval
[0.0190, 0.0352]); a small query-specific effect exists despite zero recovery.
NLL with real MIST remains slightly worse than removing the fingerprint
(32.150 versus 31.971). Exact-fingerprint NLL worsened from 11.195 to 19.175.

The concentrated-beam follow-up generates `AMWPZASLDLLQFT`
(`spectrum_id=238088`, C12H17N7O7S), initially ranked 79th, or 63rd by decoder
log probability. It is in the audit's 272-molecule MIST-unseen subset.
Reranking the pool with the full stored MIST vector puts it **seventh**:
top-25 recovery 1/300, MRR@25 0.000476. The generator consumes probabilities
at or above 0.1, while the stored predictions retain values down to 0.02;
the reranker uses those additional available predictions. Bits absent from
the stored file use probability 0.01. Ranking receives no target argument,
and removing candidate target-match labels leaves its ordering unchanged.
`ranked_shortlists.jsonl` contains at most 25 distinct scoring identities per
query. No true fingerprints or target graphs are used to order candidates.

## Verification and review

- CPU: 28 completion-model, 2 example, 48 API and 8 spectrum tests passed.
- wgpu: 28 completion-model and 2 example tests passed.
- Python: all 26 API parity tests passed against the rebuilt CPU extension;
  the adapted checkpoint also passed the four-input generation smoke test.
  An older installed wheel initially shadowed the workspace extension;
  `PYTHONPATH=bindings/python/python` selects the verified build.
- Regression checks cover effective learning-rate updates and checkpoint
  persistence, invalid rates, conditioning-source defaults, absence of true-bit
  fallback, and rejection of validation-only predictions for training.
- **Claude Opus 5.5 (`claude-opus-5-5`) approved the core code repair.** Its
  diagnosis, review and model metadata are saved in the run directory. The
  later early-stop, search and reranking analyses are separate experiments.
- Source snapshots, hashes, exact commands, per-query outputs and test logs
  are saved. Analysis asserts the independent roster, unchanged conditioning
  inputs, unchanged targets and absence of oracle scaffolds.

## Reproduction and remaining limits

Artifacts are under `data/ms2/specgen/conditioning_repair_20261007/`:
`manifest.json`, `early_stop_manifest.json`, `supplemental_provenance.json`,
`repair_summary.json`, `verification.json`, `model.ckpt` (step 87,000),
`model.ckpt.best` (step 82,000), and `ranked_shortlists.jsonl`.
The executable experiment records are `run_experiment.py`,
`evaluate_early_stop.py` and `rerank_full_mist.py`. The latter uses the existing
RDKit scoring helpers. Use `data/ms2/specgen/venv/bin/python` for scoring.

Stored MIST predictions still use true precursor formulas even though the
generator enumerates its formulas from mass. Precursor masses are unusually
accurate; the original checkpoint was historically selected on validation;
there is one seed, one spectrum per molecule, a 32-heavy-atom limit, and no
competition-test evaluation. Training predictions may also be cleaner than
predictions for molecules unseen by MIST. Checkpoint loads restore weights,
not AdamW state. Missing-conditioning counts precede evidence dropout.

The default validation-source change is intentional: old recipes that omitted
`--fp-eval` may now select a different checkpoint. Synthetic `mist_like` draws
also use the trainer's absolute step count rather than replaying local draw
numbers on a resume; this does not affect the real-prediction experiment.
The validation-NLL path still lacks source counts in its own report; complete
coverage for this run was independently verified in supplemental provenance.

Do not replace a competition pipeline on the strength of one exploratory hit.
Further useful work needs a training/generalization experiment with predicted
conditioning of deployment quality, including the predictor's formula input;
more oracle-only gains would not establish that improvement.
