# Predicted-fingerprint conditioning: channel training — 2026-10-08

**Status: fixed for teacher-forced likelihood, not for generation.** Channel
training restores a signal of about one nat from real fingerprints (the
decoder used to gain nothing from them), but it does not measurably improve
the generated pools for molecules the predictor has never seen: 6 of 1,565
targets are generated at all, and on the roster generating without the
fingerprint does as well (2 in the pool against 1). Ranking figures below
are over all queries, so a target that was never generated counts as a miss;
a good rank therefore means little while so few targets are generated. This
is a MassSpecGym diagnostic, not a competition score.

Follows `MS2_CONDITIONING_REPAIR_20261008.md` (6,000 steps on real
predictions: 0/300, validation NLL rising after step 1,000).

## What was wrong

1. **The training fingerprints were cleaner than the ones met at
   evaluation.** MIST had trained on 3,949 of the 17,718 spectral training
   molecules (22%). At probability >= 0.1 its precision/recall there is
   0.72/0.83, against 0.42/0.50 on the 12,380 training molecules it never
   saw and 0.37/0.40 on unseen validation molecules.
2. **Only memorised molecules carried a predicted fingerprint.** The decoder
   had already seen the 17.7k spectral molecules for 81,000 steps, so a noisy
   fingerprint was enough to look one up; that does not transfer. The 163k
   structure-only molecules, which are what keeps the decoder from
   overfitting, received an empty fingerprint in `predicted` mode.
3. **The existing noise model was too coarse.** `mist_like` draws every bit
   from one pooled histogram, so it keeps how often the predictor is right
   but not which bits it is right about.

## What changed

- `FingerprintChannel` (`src/models/ms2/completion_fingerprint.rs`): a
  per-bit error channel `P(outcome | truth, bit, quality class)`. A sample
  draws one quality class for the molecule, then one outcome per bit ("no
  token" or one of the eight confidence buckets).
- `examples/ms2_spectral_completion.rs`: `--fp-train channel` and
  `--fp-train predicted_channel` (the real prediction when the spectrum has
  one, the channel otherwise), `--fp-channel FILE`.
- `tools/ms2/fit_fingerprint_channel.py`: fits the channel by EM (8 classes)
  on real predictions for predictor-unseen training molecules, then
  re-estimates the class weights and one miss rate on validation molecules
  outside the 300-molecule roster. `merge_fingerprint_channels.py` mixes
  channels of different quality.
- `tools/ms2/channel_rerank.py`: scores every candidate by the channel
  likelihood of the query's real prediction, under the competition's
  identity. The answer is looked up only after the order is fixed.

Fidelity of the fitted channel on the roster's own spectra (never fitted
on): 54.2 tokens, precision 0.380, recall 0.409 simulated against 55.4,
0.366, 0.399 real. Without the validation calibration it gives 0.470/0.513.

## Experiment

Start: `progress/model2.ckpt.best` (step 81,000). Both stages: 20,000 steps,
batch 32, learning rate 2e-4, evidence dropout 0.1, all 180k molecules.

- **Stage 1** (`model_channel.ckpt`, checkpoint step 101,000): real
  predictions for the 12,364 predictor-unseen training molecules, which is
  6.9% of the training rows; the unseen-quality channel for every other row.
- **Stage 2** (`model_mixed.ckpt`, checkpoint step 121,000, continues stage
  1): the real prediction of every training molecule that has one (all but
  six), and a channel that draws a quarter of its molecules from a
  good-predictor channel fitted on MIST's own training molecules.

Generation: beam 256 split over up to eight mass-derived formulas (6.5–7.1
searched per query on average), as in the audit. All queries are counted; an
empty list is a miss. Identity is the competition's (tautomer-canonical
InChIKey first block) everywhere unless a row says otherwise.

## Results

Teacher-forced NLL per molecule on the roster (validation molecules 0:300):

| Fingerprint input | Before (repair run) | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| Real MIST | 32.15 | 26.66 | 26.36 |
| Removed | 31.97 | 27.48 | 27.36 |
| Shuffled MIST | 35.23 | 28.09 | 28.38 |
| Channel sample of the true bits | — | 25.27 | 24.58 |
| Exact (oracle) | 19.18 (from the repair document) | 17.32 | 11.94 |

The real fingerprint now beats both controls (by 1.0 nat over removed and
2.0 over shuffled in stage 2). Most of the fall from 32 to 27 is the decoder
learning to work without a usable fingerprint, not the fingerprint itself.
On the monitor slice the NLL shows no sustained rise of the kind the repair
run had: 29.6 after 1,000 adaptation steps, 27.4 at the end of stage 1 and
27.0 at the end of stage 2, with rises of up to 0.4 between consecutive
evaluations.

Generation with real MIST fingerprints, every validation molecule, stage 2.
Top-1, top-25 and MRR@25 are in the `channel_plus_prior` order (channel
likelihood plus decoder log-probability, mixed channel):

| Group | Queries | Target in pool | Top-1 | Top-25 | MRR@25 |
|---|---:|---:|---:|---:|---:|
| MIST never saw the molecule | 1,565 | 6 | 2 | 6 | 0.0017 |
| — of which the roster | 272 | 1 | 1 | 1 | 0.0037 |
| Only in MIST's simulated augmentation | 88 | 0 | 0 | 0 | 0 |
| MIST trained on the molecule | 111 | 11 | 10 | 11 | 0.095 |

By channel likelihood alone the unseen group has 1 top-1 instead of 2. In
the model's own order the 111-molecule group has 1 top-1 and 7 top-25 (MRR
0.012). Stage 1 on the same queries generated 2 of 1,565 and 5 of 111;
re-ranked with its own channel all five of the latter are first. With exact
fingerprints stage 2 still generates 97 of 300 roster targets (the audited
checkpoint generated 107, per the audit summary).

Controls on the roster (300 queries, stage 2): real 1 in the pool, removed 2,
shuffled 0. At this fingerprint quality, conditioning the generator does not
measurably beat generating without the fingerprint and re-ranking with it.
A beam and trajectory count four times larger (1,024, stage 1, still at most
1,024 returned candidates) generates the same single target.

## Reading

- The share of targets generated is higher where the fingerprint is better:
  0.4% (6/1,565) at precision/recall 0.37/0.40, 10% (11/111) at about
  0.74/0.77, 32% (97/300) with exact bits. These are three different
  populations — unseen molecules, molecules MIST memorised, and an oracle
  input on the roster — and the first two counts are small, so this is a
  direction, not a curve a better predictor can be read off.
- For a predictor-unseen molecule the real fingerprint is worth about one nat
  to the decoder (2.8 nats for a channel sample), against about 15 for the
  exact one. The one wider search that was run did not help.
- The re-ranker does not create pools. Given that the target was generated,
  it ranks it first for 10 of 11 with good fingerprints and 2 of 6 with poor
  ones; it also ranks the two targets the no-fingerprint arm generated into
  the top 25.

## Limits

- MassSpecGym, molecules of at most 32 heavy atoms, `[M+H]+`/`[M+Na]+`, one
  spectrum per query, one seed, one run per stage. Not competition data.
- Stored MIST predictions were made with the true formula.
- The channel's class weights and miss rate (nine numbers) come from
  validation molecules 300 and up, which include the NLL monitor slice
  300:600; the monitor is therefore not independent of the channel. No roster
  structure entered training or calibration. The validation queries outside
  the roster come from the population the channel was calibrated on.
- Stage 2's full-validation ranks use the mixed channel, whose good-predictor
  part was fitted on MIST's training molecules without calibration; roster
  ranks use the stage-1 channel.
- The calibration likelihood is nearly flat in the miss rate (0.15–0.30), so
  its value is not identified; the weights do most of the work.
- Bits are independent given the class. Real predictions score 44 nats per
  spectrum lower under the tables than samples from them, and vary more
  (recall s.d. 0.102 against 0.071). The decoder's NLL on real predictions is
  1.4–1.8 nats worse than on channel samples.
- The 111-molecule group is a bracket, not a forecast: MIST memorised those
  molecules.
- No matched arms for `channel` alone, `mist_like` or `predicted` alone at
  this budget; the earlier runs with those sources used other settings.
- The hit counts on unseen molecules (1, 2, 6) are too small to rank the
  stages or the controls against each other.

## Verification and review

- CPU and wgpu: `ms2_completion_fingerprint` 14 passed, example unit tests
  3 passed (two new channel tests, one new source test). CPU regression:
  completion model 28, spectrum 8, beam 19 passed.
- Reviewer: opencode, `muse-spark-1.3-contributor-free`, read-only (codex had
  reached its usage limit). Two rounds. The code review found no defect in
  the sampler, the loader or the re-ranker and no target leakage; its three
  driver findings are applied. The results review (approve with changes)
  checked every number against the artifacts; its changes are made here: the
  status and reading no longer claim a generation gain, the stage-1 pools are
  re-ranked with the stage-1 channel, the exact arm is re-scored under the
  competition identity, and the wide-beam manifest entry is corrected.
  Reviews: `data/ms2/specgen/conditioning_fix_20261008/review/`.

## Reproduction

Everything is under `data/ms2/specgen/conditioning_fix_20261008/`:
`train_channel.sh`, `train_mixed.sh`, `evaluate_fix.py`, `evaluate_all.py`,
`chain.sh`, `channel*.json|npz|report.json`, both checkpoints,
`results_summary.json`, the probes that led to the diagnosis (`probes/`) and
every per-query output.

To use another fingerprint predictor: write its held-out predictions in the
`--predictions` format, run `fit_fingerprint_channel.py` on them, then
`train_channel.sh` with the new channel file.
