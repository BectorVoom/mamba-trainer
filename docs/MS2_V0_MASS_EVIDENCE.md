# MS2 V0.1: exact mass evidence for the decoder

Status: proposal, not implemented; on hold. A codex review ([ms2_work review, 2026-10-03]) found that the
pilot does not yet establish the cause (the shuffled control was evaluated on sibling spectra; peak sensitivity,
per-field NLL and train/validation NLL are unmeasured), that teacher-forced evidence needs a per-step
composition snapshot from replay, that the evidence is a permissive mass hint rather than the recipe (top-128
peaks versus uncapped, open-valence versus boundary-bond shift bound), that the oracle check's expected value is
~100% of targets, and that an evidence-only ablation is needed. The diagnostics come first (task V0-G). Motivation: the V0 pilot
([tasks, V0.5](MS2_SUBSTRUCTURE_TASKS.md)) found that real spectra do not lower the held-out teacher-forced NLL
below the shuffled-spectrum, metadata-only and structure-prior controls, while the label diagnostic
(`tools/ms2/label_specificity.py`) shows the targets are strongly spectrum-specific: own peaks explain 51.7% of
the intensity against 5.7% for another molecule's peaks, and the retained `q` overlaps a sibling spectrum's by
0.56 against 0.054. The recipe chooses about 10 of about 52 candidate graphs per molecule **by exact mass**.
The V0 model sees peak masses only through Fourier features of the integer masses
([architecture §3.2](MS2_V0_ARCHITECTURE.md)) and must learn exact-formula matching from 3,500 spectra; it does
not. This change hands it the exact match, computed deterministically on the device from the integer sidecar.

## 1. What is computed

For a partial graph `P` (the atoms emitted so far), let `c(P)` be its composition including the fixed hydrogens
of its atom types (contracts §4.2), exactly as `chem::Composition` counts it. For each hydrogen shift
`s in {-2, ..., 2}` the ion m/z `mz(P, s)` is the contracts §4.3 ion of `c(P)` under the request adduct and
shift `s`, and its arithmetic bound is the §5 bound plus the spectrum's `mz_uncertainty`. A shift is
**admissible** at `|s| <= min(open_valence(P), 2)` — the recipe allows `|s| <= min(c(g), 2)` with `c(g)` the
boundary bonds of the final embedding, which a partial graph does not know; the residual valence is its upper
bound. The match of `P` at `s` is the §5 decision rule against the spectrum's kept peaks (the integer sidecar,
sorted by m/z, binary search):
`accept` → the largest linear relative intensity among accepted peaks; `ambiguous` → flagged; otherwise none.

Three evidence groups, all integers or exact flags turned into features in `f32` only at the end:

- **E_now** at every decoder position (the state before the next token): for each of the 5 shifts, `matched`
  (0/1), `ambiguous` (0/1), `log(1e-3 + intensity)` of the accepted peak (0 when none), and `admissible` (0/1):
  20 features. This tells the decoder whether stopping here would explain a peak.
- **E_add[t]** for each atom type `t` (18 rows): the same 20 features for `c(P) + t`'s ion — i.e. whether adding
  one atom of type `t` (budget permitting; zero row otherwise) would land on a peak. Hydrogen of the bond: an
  added atom bonded with order `b` consumes `b` of the parent's residual valence and `b` of its own, which does
  not change `c` (hydrogens are fixed per type, contracts §4.2); the shift range is evaluated on the new
  residual valence, conservatively at its maximum over the legal bonds.
- **E_spec**: the same features for the full parent formula of the conditioning formula row (one row per
  trajectory), for completeness of the "whole molecule" target.

## 2. How the model uses it

- `E_now` → `Linear(20 → d)` added to the decoder input embedding of the next position (teacher pass: position
  `i` gets the evidence of the state before token `i + 1`, exactly where the masks come from; sampler: the state
  the sampler holds before drawing).
- `E_add` → a per-type scalar bias `w · phi(E_add[t])` (`Linear(20 → 1)`) added to the atom-type logits before
  masking, and the same evidence through `Linear(20 → 1)` added to the pointer-free kind logit of ADD_ATOM via
  `max_t`; `E_now` → a scalar added to the STOP kind logit. The model can then learn "stop when the fragment
  explains a peak" and "prefer atoms that move towards a peak" directly.
- Nothing is a hard constraint: noise peaks and unexplained fragments stay possible.

## 3. Where it is computed

One kernel, `ms2_mass_evidence`, lane per (row, position) for the teacher pass and lane per row for the sampler,
reading the grammar state row (atom types, used element counts, residual valences: all already there,
architecture §3.5), the spectrum's sorted integer masses and intensities (`DeviceSpectra`), the request adduct
and uncertainty, and the constant atom table. The teacher pass needs the state at every position: either
`grammar_replay` writes the composition (10 words) per step into its replay row, or the evidence kernel is
fused into the replay kernel. The sampler calls it once per step on the live state. All integer arithmetic is
`u32` within the P2.5 bounds (a 16-atom fragment's ion is below the precursor bound + 2 Da). Host twin in
`twin.rs` against the P1 reference (`chem.rs` ion, tolerance, decision rule), poisoned outputs, CPU and wgpu.

## 4. Experiment, declared before any result

Same pilot protocol as V0.5 (same exports, step budget 6,000, batch 16, lr 1e-3, seed 1, evaluation every 1,500
steps), four models: real spectra + evidence, shuffled spectra + evidence (the evidence computed from the
shuffled peaks), metadata-only, structure prior (the last two are unchanged and need not be rerun, but are rerun
if any shared code path changed their outputs). Primary: validation teacher NLL per token, real versus shuffled.
Secondary: coverage and precision at K = 8. Additionally reported: the oracle-label check — the fraction of
targets whose `E_now` at STOP is `matched`, which must equal the fraction explained by the recipe (a test of the
kernel against the labels, not a result).

Overfitting: every V0 model's validation NLL was best at the first evaluation (step 1,500). The protocol keeps
the fixed budget for the primary comparison and additionally reports the best evaluation of each model; weight
decay and dropout are not changed in this experiment, so that the only difference from V0.5 is the evidence.

## 5. Risks and alternatives

- The evidence leaks the label recipe into the input: the labels are mass matches, so a model that learns
  "stop when matched" fits them by construction. This is intended (the recipe is the target definition), but it
  means a lower NLL is partly the recipe's own structure. The shuffled control, whose evidence comes from another
  spectrum, measures how much is spectrum-specific; the containment precision, which does not use the labels,
  measures whether generation improves.
- Admissible shifts use the residual valence as the bound on boundary bonds: it over-admits for fragments whose
  open valence will later be closed by ring bonds; this only adds false `matched` flags.
- Alternatives considered: peak-level subformula annotation in the encoder (each peak tagged with the parent
  subformulas it can be, as in MIST/SIRIUS) — richer but its size grows with the formula enumeration and it does
  not tell the decoder where it is; contrastive spectrum–fragment pretraining — more machinery than the question
  needs. The decoder-side evidence is the smallest change that answers whether exact mass evidence is the
  missing piece.
