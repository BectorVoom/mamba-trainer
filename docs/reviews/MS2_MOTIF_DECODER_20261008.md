# Motif-level decoder and its Lean 4 verification — 2026-10-08/09

**Status: with the atom-level decoder's conditioning encoders and a similar
number of training steps, a decoder that writes whole ring systems and groups
generates about as many targets as the atom-level decoder when given exact
fingerprints. With real predicted fingerprints both stay near zero on
molecules the predictor never saw.** The first version, which read its
conditioning as a token prefix, was far behind; the revised conditioning
architecture more than doubled its recovery with the alphabet unchanged,
though these runs do not isolate which of the changes made with it explains
the gain. The decoder's stack machine is stated in Lean 4 with proofs of what
it guarantees; the Rust and Python implementations are tested against the
Lean executable.

These are exact-fingerprint diagnostics on MassSpecGym across different
training and search recipes, not a competition score: similar aggregate
recovery does not establish that the alphabets are equivalent, and nothing
here shows useful recovery from spectra of unseen molecules. "Unseen" refers
to MIST's training history; the error channel used validation molecules
outside the roster for calibration, and stored MIST predictions were made
with the true formula.

## What was built

- **Alphabet** (`tools/ms2/motif_tokens.py`). A kekulized molecule is cut
  into ring systems (the connected components of its ring bonds), acyclic
  groups (non-ring heteroatoms, non-ring carbons with a multiple bond or two
  heteroatom neighbours, merged when bonded) and single carbons. Every bond
  between two motifs is then a non-ring bond, so the motifs form a tree.
  A motif's identity is the canonical SMILES of the piece with cut bonds
  capped by hydrogens; `free[i]` is the hydrogens atom `i` carries there.
- **Sequence.** `MOTIF m (ATOM a BOND o MOTIF m' ATOM b … END)* END`: attach
  motif `m'` by a bond of order `o` from atom `a` of the open motif to its
  atom `b`, push it; `END` pops. Root at a centre of the motif tree, children
  in token order, symmetries resolved by the smallest result over the
  motif's automorphisms.
- **Machine and mask** (`src/models/ms2/motif.rs`). `MotifMachine::allowed`
  is the decoder's mask and `apply` the transition: an attachment needs
  `free >= o` at both ends; with a formula budget a motif must fit the heavy
  atoms still missing and the closing `END` needs the exact formula. The
  search is `motif::beam_search`.
- **Two decoders on that alphabet** (`examples/ms2_motif_decoder.rs`):
  - `--arch prefix`: a plain `Mamba3Lm` reads the formula and the
    fingerprint bits (three confidence levels) as a token prefix. It sees no
    spectrum.
  - `--arch encoders` (`src/models/ms2/motif_model.rs`): the completion
    model's fingerprint set encoder (eight confidence buckets) and spectrum
    encoder (peaks, adduct, neutral mass) form a memory; every decoder layer
    is a Mamba block followed by cross-attention into it, and the pooled
    vector is added to every input position. This is the atom-level
    decoder's conditioning path without its pattern encoder and progress
    features.
- **Scoring** (`tools/ms2/motif_score.py`): rebuilds each sequence and ranks
  under the competition's identity, with the target's identity taken from
  the same typed graph the atom-level scorer uses.

Data: 16,460 motifs in the 180k-molecule pool, 5,680 after dropping motifs
seen in one molecule only. The stored conversion rebuilt 180,610 of 180,614
pool molecules exactly. 169,668 molecules train: 10,769 hold a dropped
motif, 4 failed conversion and 173 exceed 128 tokens. 271 of the 300 roster
targets (1,564 of the 1,764 validation queries) have a sequence; that is
conversion coverage, not a ceiling, since another sequence of the same
identity can still be generated. A training molecule averages 8.5 motifs and
39 tokens for 20 heavy atoms.

## Lean 4 verification

`lean/MotifDecoder/` (Lean 4.34.1, core only, about 2,000 lines, no
`sorry`, no added axiom; `REPORT.md` has the statements). Proven for every
state the mask can reach, given a vocabulary of well-formed, connected motifs
(which the Rust loader checks):

1. the state stays well-formed (indices in range, no self-bond);
2. no atom spends more bond order than it has, and the free valences left
   are the motifs' total minus twice the attachment orders;
3. atoms and bonds add up: the bonds are the motifs' own plus one per
   attachment (a counting identity; cycles are not formalised);
4. with a budget, an accepted sequence has exactly the target's element
   counts and hydrogens;
5. mask facts (for a non-empty vocabulary and no budget, a next token always
   exists; an accepted sequence of `k` motifs has `5k − 3` tokens);
6. every atom is connected to the first;
7. the grammar is unambiguous: each accepted sequence is the serialization
   of exactly one motif tree (which is not uniqueness of a molecule's
   sequence).

What this does and does not cover:

- The proofs are about the Lean definitions. The Rust machine and an
  independent Python port of the specification print the Lean executable's
  output lines (accepted graph, or index of the first rejected token) on
  96,670 sequences (19,334 real, four random corruptions of each), with and
  without budgets. The converter's RDKit machine was compared without
  budgets only, on acceptance and on the accepted graph's atoms, free
  valences and bond endpoints. These are tests, not proofs.
- Not in Lean: the search and its pruning rule, the networks, the converter
  and its canonical form, the chemistry behind `free`, the Rust machine's
  bound of 256 atoms, and whether a state under a budget can still be
  completed.

## Experiments

All motif runs: batch 32, learning rate 3e-4 with warm-up and cosine decay,
from scratch, `d_model` 128, 4 layers; per training row 10% no fingerprint,
25% exact, the rest a real MIST prediction (half the time, when the molecule
has one) or a sample of the mixed error channel of
`MS2_CONDITIONING_CHANNEL_FIX_20261008.md`.

| Run | Conditioning | Steps | State size | Parameters | Where |
|---|---|---:|---:|---:|---|
| prefix | token prefix | 62,000 | 16 | 3.3M | local, wgpu |
| enc-A | encoders | 62,000 | 16 | 3.1M | local, wgpu |
| enc-B | encoders, second seed | 62,000 | 16 | 3.1M | Kaggle T4, CUDA |
| enc-C | encoders | 124,000 | 16 | 3.1M | Kaggle T4, CUDA |
| enc-D | encoders | 124,000 | 32 | 3.2M | Kaggle T4, CUDA |

The prefix run padded prefixes to the batch maximum for its first 22,000
steps and to one fixed length afterwards (continued from that checkpoint,
weights only). The encoder runs have no prefix.

The prefix and encoder runs differ in more than where the conditioning
enters: the encoder runs also read the spectrum, encode the fingerprint in
eight buckets instead of three levels, have learned position embeddings,
train a 5,717-way output instead of an 11,096-way one (the prefix model's
output includes its conditioning ids), and train without a restart.

**Reference: the atom-level decoder's stage-2 checkpoint** (2.0M parameters,
121,000 steps). The comparison shares the queries, the formula lists (taken
from the atom-level run's own formula search), the beam width (256, split
over up to eight formulas) and the identity. It does **not** share:

- the alphabet and the decoder around it: the atom-level decoder has
  factored action and pointer heads, an atom memory and per-step progress
  features, a state size of 32 (16 in prefix and enc-A/B/C, 32 in enc-D), and
  a substructure encoder that received no pattern in any compared query;
- the training objective: unmasked token cross-entropy here, grammar-masked
  factored action likelihood there;
- the training mixture: the atom-level stage 2 trained on real predictions
  and channel samples with no exact rows, having learned exact fingerprints
  in its first 81,000 steps;
- the starting point and schedule: scratch with warm-up and cosine decay at
  3e-4 against a warm start at a constant 2e-4;
- the training population: 169,668 against 180,416 molecules;
- the search's limits and pools: a position limit (128 tokens) against a
  ring-closure limit (six); sequences cut at 1,024 before identities are
  merged against graphs merged before the cut (no saved query reaches
  either cut);
- for the prefix run only, the spectrum and the fingerprint encoding.

## Results

Targets generated and rank by the decoder's log-probability, 300 roster
queries, exact fingerprints:

| Decoder | True formula: pool | top-1 / top-25 | MRR@25 | From mass: pool | top-1 / top-25 | MRR@25 |
|---|---:|---:|---:|---:|---:|---:|
| Motif, prefix | 78 | 8 / 46 | 0.052 | 33 | 5 / 32 | 0.038 |
| Motif, enc-A | 119 | 36 / 96 | 0.167 | 78 | 29 / 78 | 0.138 |
| Motif, enc-B | 119 | 32 / 98 | 0.162 | 76 | 27 / 74 | 0.138 |
| Motif, enc-C | 148 | 43 / 117 | 0.198 | 93 | 37 / 93 | 0.168 |
| Motif, enc-D | 138 | 57 / 117 | 0.240 | 97 | 50 / 96 | 0.211 |
| Atom-level | 141 | 41 / 113 | 0.193 | 97 | 35 / 94 | 0.160 |

Controls on the roster with mass-derived formulas (targets generated):

| Fingerprint | prefix | enc-A | enc-B | enc-C | enc-D | Atom-level |
|---|---:|---:|---:|---:|---:|---:|
| Removed | 0 | 0 | 0 | 0 | 0 | 2 |
| Shuffled MIST | 0 | 0 | 1 | 0 | 0 | 0 |

Real MIST fingerprints, the 1,764 validation queries of the atom-level run
(targets generated; in brackets how many rank first by channel likelihood
plus log-probability):

| Group | Queries | prefix | enc-A | enc-B | enc-C | enc-D | Atom-level |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIST never saw the molecule | 1,565 | 3 (1) | 4 (2) | 3 (1) | 2 (2) | 3 (2) | 6 (2) |
| Only in MIST's augmentation | 88 | 1 (0) | 0 | 0 | 0 | 0 | 0 |
| MIST trained on the molecule | 111 | 8 (8) | 8 (8) | 10 (10) | 12 (12) | 12 (12) | 11 (10) |

Teacher-forced NLL per molecule of the motif runs, given the true formula,
on the 271 roster targets with a sequence (not comparable with the
atom-level decoder's, which is over all 300 targets, another sequence and a
masked normalisation):

| Fingerprint | prefix | enc-A | enc-B | enc-C | enc-D |
|---|---:|---:|---:|---:|---:|
| Exact | 15.81 | 13.35 | 13.66 | 12.42 | 12.58 |
| Real MIST | 24.95 | 27.12 | 27.30 | 26.51 | 26.86 |
| Removed | 24.78 | 28.34 | 28.59 | 27.41 | 27.80 |
| Shuffled MIST | 26.91 | 30.36 | 30.76 | 29.36 | 29.54 |

What the pools contain (roster, mass formulas; one entry per identity; share
with a ring of that size, and best Tanimoto to the target among the 25 best
by log-probability; targets: 6.7% and 2.0%):

| Arm | Decoder | 3-ring | 4-ring | best of 25 |
|---|---|---:|---:|---:|
| Exact | prefix | 6.2% | 1.1% | 0.469 |
| Exact | enc-A / B / C / D | 7.1 / 7.1 / 6.7 / 7.0% | 2.2 / 1.6 / 1.7 / 2.3% | 0.575 / 0.575 / 0.631 / 0.627 |
| Exact | atom-level | 6.8% | 7.2% | 0.615 |
| Real MIST | prefix | 1.9% | 0.1% | 0.194 |
| Real MIST | enc-A / B / C / D | 1.1 / 1.2 / 1.5 / 1.6% | 0.0 / 0.1 / 0.1 / 0.2% | 0.191 / 0.188 / 0.209 / 0.196 |
| Real MIST | atom-level | 5.5% | 5.5% | 0.152 |

## Reading

- **The revised conditioning architecture substantially improves
  exact-fingerprint recovery with the motif alphabet unchanged.** Same
  alphabet, steps and block settings: 78 against 33 targets with
  mass-derived formulas (119 against 78 with the true formula), and the
  second seed gives 76 and 119. The alphabet was therefore not what held the
  first decoder back; which of the changes listed under "Experiments"
  (encoders and cross-attention, the spectrum, the fingerprint encoding, the
  output size, the uninterrupted training) accounts for the gain is not
  isolated.
- **With about the atom-level decoder's number of steps the motif decoder
  generates a similar number of targets with exact fingerprints**: 93 and 97
  against 97 from mass, 148 and 138 against 141 with the true formula. One
  run each; this is similar aggregate recovery in these runs, not shown
  equivalence. The two decoders succeed on different queries: of the 97
  targets each generates from mass (enc-D and atom-level), 58 are shared, so
  136 of the 300 are generated by at least one of them.
- **enc-D has the higher observed top-1 count**: the target is first by the
  decoder's own score for 50 queries against 35 (57 against 41 with the
  true formula). Paired on the same queries this is 36 against 21 discordant
  (exact McNemar p = 0.06; 39 against 23, p = 0.06, with the true formula),
  and enc-C at the same steps gives 37 and 43, so an advantage of the
  configuration is not established.
- **Real fingerprints: no resolved difference, and no useful recovery on
  unseen molecules for any decoder** (2-4 against 6 of 1,565). Where MIST
  trained on the molecule, the motif runs generate 8-12 of 111 against 11;
  every one the motif runs generate is first after re-ranking (10 of the
  atom-level decoder's 11). Given the true formula, the real fingerprint
  lowers the encoder runs' mean teacher-forced NLL by 0.9-1.3 nats against
  no fingerprint; the prefix run's is 0.17 nats worse with it.
- **Ring content.** With exact fingerprints the encoder runs' pools have
  about the targets' share of three- and four-membered rings (6.7-7.1% and
  1.6-2.3% against 6.7% and 2.0%); the prefix run has 1.1% with a
  four-membered ring and the atom-level decoder 7.2%. With weak fingerprints
  every motif pool has too few small rings (four-membered 0-0.2% against
  2.0%, three-membered 1.1-1.9% against 6.7%). Shares are over candidates,
  the reference over targets; these are descriptive frequencies, not a
  statement about plausibility.

## Limits

- One run per configuration except enc-A/enc-B; one model width; no
  atom-level run from scratch under the motif recipe, so the comparison with
  the atom-level decoder is across the differences listed under
  "Experiments".
- 29 roster targets (200 validation targets) have no sequence in the cut
  vocabulary. They are counted as queries.
- Five highly symmetric molecules of the stored conversion would now be
  left out (2,000 or more automorphisms in one motif); none is in the
  vocabulary used.
- Charged atoms, radicals, isotopes and stereo are outside both alphabets;
  molecules above 32 heavy atoms are outside the data.
- Stored MIST predictions were made with the true formula. The local RDKit
  is 2026.03.6, not the competition's 2026.03.3.
- The motif search steps about twice the decoder rows per query of the
  atom-level search (about 16-17k against 9.8k with the true formula, 19-22k
  against 8.4k with mass-derived formulas), and its mask is applied on the
  host.
- The Kaggle runs used the CUDA backend and the local runs wgpu; enc-A and
  enc-B differ in seed and backend, so they do not separate the two, and no
  test compares stepping with the teacher pass on CUDA. The source archive
  the Kaggle runs built from matches the workspace in 397 of 400 files; the
  three that differ are outside the motif driver and model. enc-C and enc-D
  validate every 4,000 steps instead of 2,000 and spread their schedule over
  the longer run; no build or input hashes were recorded on Kaggle.
- The prefix run's corrected full-validation score is from
  `final62k/mist_mass_all1766.jsonl` filtered to the shared query set; the
  older `mist_mass_all.jsonl` beside it holds the earlier 1,764 rows.
- The encoder search's length limit was off by one (a model of `P` positions
  could return at most `P - 1` tokens). It is fixed and tested; the reported
  runs could not reach it (accepted sequences have `5k - 3` tokens and the
  limit was 128).

## Verification and review

- Tests: `tests/ms2_motif.rs` 12 passed on CPU and on wgpu (the machine against the Lean
  executable's stored output with and without budgets, the mask, the bound
  on graph size, the prefix layout, the beam's selection, ties, dead ends and
  non-finite logits, the cache reorder, and for the encoder model: stepping
  equals the teacher-forced pass, reordered rows continue their parents,
  every conditioning input changes the output, a query's logits do not
  depend on its batch, every parameter receives a gradient). The fingerprint
  (14) and completion-model (28) suites still pass on CPU. `lake build`
  clean; 43 theorems report only Lean's three standard axioms.
- Reviewer: codex, read-only. Two rounds on the code and proofs: the proofs
  are not vacuous; outside them it found a per-parent beam cut that could
  drop the only valid completion, training/search padding that differed,
  unchecked integer bounds, a symmetry cap that broke the canonical form,
  fragile id parsing, negative indices accepted by the Python machine, and
  overstated sentences. All fixed. One round on the first results: counts
  confirmed; the comparison's description, the query set (two queries
  differed), one target identity, the NLL reading and several conclusions
  were corrected as it asked. One round on these final results and the
  encoder model (approve with changes): tables, query sets and target
  identities verified and no defect found in the model's conditioning,
  teacher forcing or stepping; it asked for the narrower causal statement,
  the fuller list of differences, the length-limit fix and the numeric and
  provenance corrections now in this document. Reviews:
  `data/ms2/specgen/motif_run/review/`.

## Reproduction

`data/ms2/specgen/motif/` (vocabulary and sequences);
`data/ms2/specgen/motif_run/` (`train_motif*.sh`, `evaluate_motif*.sh`,
checkpoints `motif.ckpt` and `motif_enc.ckpt`, `final62k/` and `enc62k/`
with every search output and score, `keys_*.json` query sets,
`results_summary.json`, `lean_cross/`);
`data/ms2/specgen/kaggle_motif/` (`make_kernel.py`, `score_run.sh`, and
`out_<kernel>/` with each Kaggle run's log, checkpoint, search outputs and
scores).

```sh
python tools/ms2/motif_tokens.py build --export <train.json> --export <extra.json> --apply <validation.json> --out <dir>
python tools/ms2/motif_tokens.py prepare --motif-dir <dir> --min-count 2 --export … --fp … --out <dir>
cargo run --release --no-default-features --features wgpu --example ms2_motif_decoder -- --arch encoders --vocab <dir>/lm_vocab.json …
cd lean/MotifDecoder && ~/.elan/bin/lake build
```
