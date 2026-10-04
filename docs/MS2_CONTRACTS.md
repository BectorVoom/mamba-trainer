# MS2-to-substructure contracts (P0)

Status: V0 contracts, revision 3 (2026-10-03), after the two codex reviews recorded in
[reviews/MS2_CONTRACTS_CODEX_REVIEW.md](reviews/MS2_CONTRACTS_CODEX_REVIEW.md) and
[reviews/MS2_CONTRACTS_CODEX_REVIEW_2.md](reviews/MS2_CONTRACTS_CODEX_REVIEW_2.md).
Design: [MS2_SUBSTRUCTURE_DESIGN.md](MS2_SUBSTRUCTURE_DESIGN.md). Tasks: [MS2_SUBSTRUCTURE_TASKS.md](MS2_SUBSTRUCTURE_TASKS.md).

Every number marked *measured* comes from one of the reports below, produced by scripts in this repository on
the files named in §1. Nothing here is a model result.

| Report | Script | What it measures |
|---|---|---|
| [casmi_audit.json](../bench/results/ms2/casmi_audit.json) | [audit_casmi.py](../tools/ms2/audit_casmi.py) | All 2,539,608 training spectra (metadata), 394,400 spectra (peak lists, 4 of 21 row groups), all 277,566 structures, the element-mass comparison with RDKit |
| [target_pilot.json](../bench/results/ms2/target_pilot.json) | [pilot_targets.py](../tools/ms2/pilot_targets.py) | The target recipe of §7 on 300 in-domain training-fold molecules, one spectrum each, 14,046 peaks after the adapter's 512-peak pre-selection and the filter of §2 |
| [formula_table.json](../bench/results/ms2/formula_table.json) | [formula_table.py](../tools/ms2/formula_table.py) | The V0 formula table of §9: rows, bytes, window occupancy, validation-fold coverage |

All three use [ms2_reference.py](../tools/ms2/ms2_reference.py), a Python implementation of §4, §5 and §7 that
is also the source of the Rust test fixture ([make_fixtures.py](../tools/ms2/make_fixtures.py)). They run under
the RDKit 2026.03.3 environment of the CASMI project
(`uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/<script>.py …`).

## 1. Data inventory and provenance (P0.5)

| Item | Location | Notes |
|---|---|---|
| Paired spectra and structures | `Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data/raw/train.parquet` | 2,539,608 spectra, SHA-256 `9423f90e…bcd98` (*measured*, equal to the hash the folds were built on) |
| Structures and folds | `…/data/folds/structures.parquet`, `spectra_folds.parquet` | 277,566 structures, 274,184 identity groups, 140,160 scaffold groups; 5 folds each |
| Hidden-test spectra | `…/data/raw/test.parquet` | 1,213 spectra, no structures. Not used for training, calibration or model selection |
| FPNet reference code | `…/data/v4g/v4b__code__casmi__fpnet6.py`, `…__chem.py`, `…__engine.py`, `…__traindata.py` | The `prep_peaks` contract of §2 |

Terms: the spectra and structures are the Enveda CASMI 2026 competition data; `…/data/v4g/SOURCE-LICENSES.md`
lists component-specific terms, including CC BY-NC components, and no blanket licence. This project treats the
data as non-commercial research material and commits none of it: the committed test fixture holds only
well-known public structures written as SMILES and synthetic spectra. Reports on real spectra are produced
locally by the scripts above.

**Fragment labels.** The data has no experimental fragment or peak annotations. Every fragment-level label in
this project is a pseudo-label from the recipe of §7. Claims about experimental ion-assignment accuracy are out
of reach of this data and are not made.

**Splits (frozen).** Every spectrum of an identity group stays in one subset.

| Subset | Identity split | Use |
|---|---|---|
| Train | `fold_identity` in {2, 3, 4} | Model training; also, with the validation fold, the only data the vocabulary, the recipe parameters and the formula table are chosen from |
| Validation | `fold_identity` = 1 and `identity_group % 3` = 0 | Model selection and early stopping |
| Ranking | `fold_identity` = 1 and `identity_group % 3` = 1 | Reranker fitting (P6.3) |
| Calibration | `fold_identity` = 1 and `identity_group % 3` = 2 | Calibration fitting, evaluated by cross-fitting within this subset (P7.9) |
| Test | `fold_identity` = 0 | Untouched until P9 |

The scaffold-disjoint evaluation uses `fold_scaffold` with the same numbers. In-domain structures per identity
fold 0–4: 54,532 / 54,506 / 54,521 / 54,519 / 54,531; the three parts of fold 1 hold 18,055 / 18,186 / 18,265
in-domain structures (*measured*).

**Instrument holdout (frozen protocol).** Classes by case-insensitive match on `instrument_type`: `timstof`
(45.5% of spectra), `orbitrap` (`orbitrap`, `qft` or `itft`; 37.0%), `qtof` (any other `tof`; 13.3%), `other`
(the rest and missing; 4.2%), all *measured*. Leave-one-class-out over the first three: train on the train
subset without the held-out class, evaluate on the test subset's spectra of the held-out class. P9.1 runs it.

**Structural pretraining.** V0 does none. Any later structure-only pretraining must drop the validation,
ranking, calibration and test identity groups first.

## 2. The `prep_peaks` contract and the FPNet inputs (P0.1)

Source: `prep_peaks(mz, it, prec_mz, max_peaks=160, floor=1e-3)` at `fpnet6.py:28`, `collate` at
`fpnet6.py:142`, callers at `engine.py:341-359` and `traindata.py:65-95`.

| Step | Behaviour |
|---|---|
| Input dtype | `float64` m/z and intensity |
| Mass unit | m/z as stored in the data file. *Measured*: 99.7% of spectra store 4 decimals |
| Filter 1 | Keep `0 < mz <= prec_mz + 2.0`. *Measured*: 16.5% of spectra hold a peak above that bound |
| Empty | No peak left, or maximum intensity `<= 0`: two empty arrays |
| Intensity | Divide by the maximum of the kept peaks, then keep `>= 1e-3` |
| Cap | Above 160 peaks: the 160 most intense (`argsort(-it)`, ties in unspecified order). *Measured*: 5.3% of spectra are truncated; the kept peaks then hold 98.4% of the intensity at the median, 88.6% at the 5th percentile |
| Sort | Ascending m/z (`argsort`, ties in unspecified order). *Measured*: 58 of 394,400 spectra hold two peaks of one m/z |
| Output | `float32` m/z, `float32` **square root** of the relative intensity, in (0, 1] |
| Precision loss | The `float32` cast. *Measured*: at most 0.060 ppm per spectrum, median 0.050 ppm |
| Padding | `collate` pads with zeros and a boolean mask; a spectrum with no peak gets slot 0 **unmasked** with `mz = it = 0` |

FPNet's per-spectrum metadata: precursor m/z (`float32`), adduct index into `chem.ADDUCT_LIST` (the ten test
adducts, then the rest, then `<unk>`), collision energy in eV (`0` when unknown), a known flag, the polarity as
`+1` or `-1`, and a merged count divided by 4. The count differs by caller: training uses `max(1, count)` for a
single spectrum and the sum of `max(1, count)` over merged spectra, capped at 8 (`traindata.py:76-95`);
inference uses `min(count, 8)` for a single spectrum, where a missing count defaults to 1, and the capped sum
of `max(1, count)` for a merged one (`engine.py:345-359`).

Consequences for this project:

1. **Original-mass sidecar is required.** Exact-mass decisions never see a `float32` m/z. The host converts the
   stored decimal or `float64` value once to an integer (§5) before any cast and states its precision
   (`mz_uncertainty_udalton`, §3.1). A caller that has only `float32` m/z still fills the integer field, with
   the uncertainty sentinel that disables exact-mass decisions for that spectrum.
2. **No double transformation.** The model takes untransformed non-negative intensities and applies the
   normalisation itself. An adapter fed `prep_peaks` output (already square-rooted) says so
   (`intensity_scale = 1`), and the model squares it back before its own features.
3. **Deterministic ties.** This project breaks ties by `peak_id`, in both the cap and the sort.
4. **Empty spectra abstain** (`empty_spectrum`); the unmasked zero peak is not reproduced.
5. **Peaks above the precursor.** V0 is singly charged (§4), so it keeps FPNet's `mz <= prec + 2.0` rule and
   counts the dropped peaks. A multiply charged domain must lift it.
6. **Energy count.** This project's `energy_count` is the number of collision energies behind the spectrum,
   `0` when unknown; it does not reproduce FPNet's `max(1, ·)` substitution, which makes unknown and one
   indistinguishable.

## 3. Schemas (P0.2)

Host-side, `serde`-serializable, row-major arrays. Each of the five schemas below starts with
`schema_version: u32`, omitted from the field tables; its `validate` rejects a version it does not
know (`Error::Config` naming both versions) and never guesses. `SpectrumBatch` and `ChemistryDomain`
stay at version 1; `ModelConfig`, `GenerationConfig` and `CandidateBatch` move to version 2 (V1 §1.2).
`B` is spectra per batch.

### 3.1 `SpectrumBatch`

| Field | Type / shape | Unit and meaning | Missing or invalid |
|---|---|---|---|
| `n_raw` | `u32` | Raw peak capacity of this batch's shape bucket: one of 64, 128, 256, 512 | Anything else: `Error::Config` |
| `spectrum_id` | `u64 [B]` | Stable identity: keys the RNG and the provenance of every candidate | Required; duplicates within a batch are `Error::Config` |
| `raw_peak_count` | `u32 [B]` | Peaks the caller had before any host pre-selection | `> peak_count` sets `raw_truncated`; `< peak_count` is `Error::Config` |
| `peak_count` | `u32 [B]` | Peaks supplied in this batch, `<= n_raw` | `0`: `empty_spectrum`; `> n_raw`: `over_capacity` |
| `peak_id` | `u32 [B, n_raw]` | Index of the peak in the caller's original list, strictly increasing within a spectrum; returned with every piece of evidence | Padding is `u32::MAX`; a non-increasing id, or a valid peak's id at or above `raw_peak_count`, is `Error::Config` |
| `mz_udalton` | `u32 [B, n_raw]` | m/z in integer units of 10⁻⁶ (`round(mz * 1e6)`) | Padding `0`, never read; a valid peak with `0` is `invalid_peak` |
| `intensity` | `f32 [B, n_raw]` | Non-negative, finite, in the scale `intensity_scale` names | Padding `0`, never read; NaN, infinite, or at or above `3e38` (`1.7e19` with scale 1, whose square must stay below `3e38`): `nonfinite_input`; negative: `negative_intensity`; all zero: `empty_spectrum` |
| `intensity_scale` | `u8` | `0` linear (default), `1` square root of relative intensity | Other: `Error::Config` |
| `mz_uncertainty_udalton` | `u32 [B]` | Half-width of the rounding interval of the stored m/z (4 decimals: 50) | `u32::MAX`: precision unknown (a `float32` source); sets `exact_mass_unavailable` and disables exact-mass decisions |
| `precursor_mz_udalton` | `u32 [B]` | Precursor m/z | Outside 50 to 2000 Da: `precursor_out_of_range` |
| `precursor_uncertainty_udalton` | `u32 [B]` | As for peaks | `u32::MAX`: formula search is skipped, `exact_mass_unavailable` |
| `adduct` | `u16 [B]` | Id in the chemistry domain | `0`: `insufficient_metadata`; an id the domain lacks: `unsupported_adduct` |
| `polarity` | `i8 [B]` | `+1` or `-1` | Other: `invalid_polarity`; sign differs from the adduct's charge: `polarity_adduct_conflict` |
| `collision_energy_ev` | `f32 [B]` | Mean of the energies, eV | Read only when the known flag is `1`; then NaN, infinite or negative is `nonfinite_input` |
| `collision_energy_known` | `u8 [B]` | `1` when the eV value is a measurement | `0`: a learned missing-energy embedding is used and a stored `0.0` is not read as 0 eV; other values: `Error::Config` |
| `energy_count` | `u8 [B]` | Collision energies behind the spectrum | `0` unknown; `1..=7` exact; `8` means **8 or more**; above 8: `Error::Config` |
| `fragment_tolerance_ppm_tenths` | `u16 [B]` | Tolerance for fragment-ion decisions | `0`: the default 100 (10 ppm); above 1000: `Error::Config` |
| `precursor_tolerance_ppm_tenths` | `u16 [B]` | Tolerance for the precursor formula window | `0`: the default 200 (20 ppm); above 1000: `Error::Config` |
| `instrument_class` | `u8 [B]` | `0` unknown, `1` timstof, `2` orbitrap, `3` qtof, `4` other | Recorded for evaluation; not a model input in V0 |

A host adapter that receives more than `n_raw` peaks keeps the `n_raw` most intense (ties by original index)
and reports both counts (*measured*: 8.7% of spectra exceed 512 peaks). The device then applies §2's filter and
keeps `N` peaks (V0: 128, most intense, ties by `peak_id`; *measured*: 8.4% of spectra are cut at 128) and
reports the kept count and the retained intensity fraction.

`Error::Config` marks a malformed batch (the whole call fails before any upload). A status marks one spectrum
(§8); the other spectra of the batch are unaffected.

### 3.2 `ChemistryDomain`

| Field | Type | V0 value |
|---|---|---|
| `version` | string | `ms2-chem-v0.1` |
| `mass_scale` | `u32` | `1_000_000` |
| `elements` | list of `{symbol, exact (decimal string), mass: u32, residual_nda: u32}` | §4.1 |
| `electron_mass`, `electron_residual_nda` | `u32` | `549`, `421` |
| `atom_types` | list of `{id: u8, element: index, hydrogens: u8, valence: u8}` | §4.2 |
| `bond_orders` | list of `u8` | `[1, 2, 3]` |
| `adducts` | list of `{id: u16, name, hydrogens: i32, charge: i32}` | §4.3 |
| `max_hydrogen_shift` | `u8` | `2` |
| `grammar`, `traversal`, `recipe` | strings | `grammar-bfs-v1`, `bfs-canon-v1`, `q-cut-v1` |

### 3.3 `ModelConfig`

| Field | Type | V0 value |
|---|---|---|
| `version` | string | `ms2-model-v0` |
| `chemistry` | string | The domain version the weights were trained on; a mismatch is `Error::Config` |
| `n_peaks` (N) | `u32` | 128 |
| `d_model` | `u32` | 128 |
| `encoder`, `decoder` | `SsmConfig` | `d_model 128, n_heads 4, head_dim 64, d_state 32, n_groups 4, Siso, Rotational, LearnedTrapezoid, conv_kernel None`; so `d_inner = n_heads * head_dim = 256` |
| `encoder_blocks`, `decoder_blocks` | `u32` | 2, 2 |
| `attention_heads` | `u32` | 4 |
| `fourier_features` | `u32` | 16 frequencies per scalar |
| `max_atoms` (A), `max_ring_closures` (R_max) | `u32` | 16, 4 |
| `formula_table` | `{version, rows: u32, sha256}` | §9; V1 schema 2 names a table only when `assignment` is absent (assignment disabled) |
| `formula_artifacts` | `{domain_version, domain_sha256, bounds_version, bounds_sha256}` or absent | V1 §1.4; `None` for table-only; a mismatch at load is `Error::Config` |
| `assignment` | `{hypotheses: u32 (J, 1..=8, default 4), work_max: u32 (default 4096), labels: u32 (L, default 64)}` or absent | Architecture §2; `None` (or version 1) means assignment disabled: exactly V0 behaviour and results; parameters travel with the model (visited/saved/loaded); a head-less checkpoint into an assignment config is `Error::Config` naming the missing parameters |

V1 §3.1 ranges (enforced by `ModelConfig::validate`): `1 <= max_atoms <=
32`, `max_ring_closures <= 8`, `1 <= decoder_blocks <= 4`; the decoder's
`n_heads * head_dim` may differ from the encoder's (only `d_model` is
shared). The V1 candidate shape for tests is `A = 32`, `R_max = 8` with 4
decoder blocks (`ModelConfig::v1_candidate`, tests only); `T = 42` belongs
to `GenerationConfig::max_steps` (`2 + A + R_max`, at most 64). The
vocabulary widths (5 kinds, 18 atom-type rows, 4 bond rows) are frozen with
the chemistry domain and stay constants.
| `energy_scale_ev`, `energy_clip_ev` | `f32` | 100, 400: the feature is `min(ce, 400) / 100`. Fixed constants, so no statistic is fitted on data |
| `dtype` | `DType` | `F32` | Validated set (contracts §3.3), independent of hardware capability: `F32` on every backend; `BF16` on the CPU backend only; `F16` nowhere — f16 is not validated for the MS2 model (NaN on the CPU runtime, kernel compilation failure on wgpu). The one shared `Ms2Capabilities::check_dtype` function (used by production, tests and `examples/ms2_dtype_report.rs`) refuses any other dtype with `Error::Unsupported` saying it is "not validated for this backend". `Ms2Model::init`, `Ms2Trainer::new` and the generation/training preflights additionally require the actual neural element type to equal the configured dtype (`E::DTYPE == config.dtype`, else `Error::Config`), and apply the policy to `E::DTYPE` — all before any allocation, upload or launch. Exact-mass decisions stay independent of the neural dtype |

Schema version 2 (V1 §1.2). A version-1 document still loads (table only); any other version is `Error::Config`.

### 3.4 `GenerationConfig`

| Field | Type | Default | Range and behaviour |
|---|---|---|---|
| `trajectories` (K) | `u32` | 8 | 1 to 64; the **total** per spectrum across formula hypotheses |
| `formulas` (F) | `u32` | 4 | 1 to 8 |
| `seed` | `u64` | 0 | Any |
| `temperature` | `f32` | 1.0 | `(0, 4]` |
| `max_steps` (T) | `u32` | 22 | At least `2 + A + R_max`, at most 64; smaller is `Error::Config` |
| `max_device_bytes` | `u64` | 2 GiB | A configuration whose estimate exceeds it is refused before allocation, never silently reduced |
| `formula_rows_visited_max` | `u32` | `u32::MAX` (no limit) | A search that needs more reads than this sets `formula_search_exhausted` (§9) |
| `formula_rows_scored_max` | `u32` | 4096 | More joined rows than this sets `formula_search_exhausted` (§9) |
| `mode` | enum | `Sampling` | `Beam` is P5 and is `Error::Unsupported` until then |
| `oracle_formula` | `bool` | `false` | Diagnostic only; every candidate carries `formula_source_oracle` |
| `control` | enum | `None` | `ShuffledSpectrum`, `MetadataOnly` (§10) |
| `formula_source` | enum | `Table` | `Table` or `Enumerate` (V1 §1.4) |
| `formula_window` (M) | `u32` | 32 | One of 32, 128, 512, 2048; anything else is `Error::Config` |
| `enum_lanes_max` | `u32` | 262144 | `Enumerate` only: `B * P` above this is refused before any launch; must be non-zero |
| `enum_lane_visits_max` | `u32` | 65536 | `Enumerate` only: per-lane visit budget; `formula_rows_visited_max` is not used here; must be non-zero |
| `allocation` | enum | `RoundRobin` | `RoundRobin` (V0: trajectory `k` uses formula `k mod top_count`) or `Proportional` (V1 §3.2) |
| `identity` | enum | `TraceOnly` | `TraceOnly` (no identity kernel, `identity_resolution` 0) or `Graph` (`graph_hash` then `graph_identity`, V1 §4.2) |
| `identity_work_max` | `u32` | 4096 | Per-pair exact-comparison budget; must be non-zero. The request bound `B * K * (K - 1) / 2 * identity_work_max <= identity_request_work_max` (2^28) is checked before dispatch |
| `returned` (R) | `u32` | `min(10, K)` (`0` in documents means the default) | Packed slots per spectrum (V1 §4.4); `1 <= R <= K` |
| `evidence` | `bool` | `false` | Emit fragment-ion evidence (§2.4); requires `ModelConfig::assignment`, else `Error::Config`; with `false` every evidence field is the V0 constant and no extra launch happens |
| `ion_request_work_max` | `u32` | `2^28` | `B * F * N * ion_work_max` above this is refused before dispatch (§2.1); must be non-zero |

Schema version 2 (V1 §1.2). A version-1 document still loads with `formula_source = Table`, `formula_window = 32`, `enum_lanes_max = 262144`, `enum_lane_visits_max = 65536`, `allocation = RoundRobin`, `identity = TraceOnly`, `identity_work_max = 4096`, `returned = 0` (the default `min(10, K)`), `evidence = false`, `ion_request_work_max = 2^28`; any other version is `Error::Config`. `TrainConfig` gains `formula_source` (`Table`, default), `formula_window` (32, one of 32, 128, 512, 2048) and `lambda_assign` (`0.0` = off; `0.1` with `--assign`).

### 3.5 `CandidateBatch`

Exactly `B * K` records, in `(spectrum, trajectory)` order, read in one batched read. V0 does no compaction: a
failed request still has its K records, with `request_failed` set and `length = 0`.

| Field | Type / shape | Meaning |
|---|---|---|
| `spectrum_id` | `u64 [B*K]` | Provenance |
| `trajectory` | `u32 [B*K]` | `0..K` |
| `actions` | `u32 [B*K, T, 4]` | `(kind, atom_type, bond_order, pointer)` per step (§4.4); fields a kind does not use are `0`; steps at or after `length` are PAD (all `0`) |
| `length` | `u32 [B*K]` | Tokens emitted: START, the actions, and STOP only when `finished`. A `truncated` trace has `length = T` and no STOP |
| `formula_row` | `u32 [B*K]` | Row of the conditioning formula; `u32::MAX` when there is none, and always `u32::MAX` for an enumerated formula |
| `formula_log_prob` | `f32 [B*K]` | `log p(formula | spectrum)` over the scored support; `0` with `formula_source_oracle`; NaN is never emitted (a missing value is `0` with the status that explains it) |
| `trace_log_prob` | `f32 [B*K]` | `sum_t log p(a_t | a_<t, spectrum, formula)` over legal support, up to `length` |
| `open_valence` | `u8 [B*K, A]` | Residual valence per atom (§4.5); meaningful only when `finished` |
| `attachment_partition` | `u8 [B*K]` | `0` = unknown, the only V0 value |
| `status` | `u32 [B*K]` | Candidate bits of §8 |
| `evidence_status` | `u8 [B*K]` | `0` = unassigned, the only V0 value |
| `identity_resolution` | `u8 [B*K]` | `0` = trace only (`identity = TraceOnly`); `1` = exact (every comparison decided); `2` = unresolved (some comparison ran out of budget; not a duplicate flag, the candidate stays eligible) |
| `request_status` | `u32 [B]` | Request bits of §8 |
| `rows_visited`, `rows_joined`, `rows_scored` | `u32 [B]` | §9; `rows_scored = min(rows_joined, formula_rows_scored_max, M)` |
| `formula_support_complete` | `u8 [B]` | `1` when every joined row was scored: only then are the formula probabilities over the whole window (§9) |
| `formula_mass_retained` | `f32 [B]` | Probability mass of the retained `F` formulas within the scored window; `0` and not meaningful unless `formula_support_complete` is `1` |
| `peaks_kept` | `u32 [B]` | Peaks after device selection |
| `intensity_retained` | `f32 [B]` | Fraction of filtered intensity the kept peaks hold |
| `formula_counts` | `u16 [B*K, 10]` | Composition of the conditioning formula in `ELEMENTS` order; all `0` when there is none |
| `formula_source` | `u8 [B]` | `0` table, `1` enumeration |
| `formula_rank` | `u32 [B*K]` | Rank of the formula in the scored support (its window slot); `u32::MAX` when there is none |
| `evidence_count` | `u8 [B*K]` | Evidence records returned (`0..=4`); `0` when evidence is disabled (the V0 constant) |
| `evidence_peak_id` | `u32 [B*K, E]` | ORIGINAL peak ids per record (`E = 4`); `0` beyond the count |
| `evidence_hypothesis` | `u8 [B*K, E]` | Hypothesis index among the kept `J` per record; `0` beyond the count |
| `evidence_shift` | `i8 [B*K, E]` | Hydrogen shift `s` per record; `0` beyond the count |
| `evidence_residual` | `i32 [B*K, E]` | Signed residual in integer mass units per record; `0` beyond the count |
| `evidence_log_prob` | `f32 [B*K, E]` | Assignment log-probability per record; `0` beyond the count |

Schema version 2 (V1 §1.2). A version-1 document has the three new fields absent (empty arrays, accepted by `validate` for version 1 only); any other version is `Error::Config`. `validate` checks lengths, `formula_source` in `{0, 1}`, `formula_row == u32::MAX` whenever `formula_source == 1`, and counts all zero exactly when there is no formula. No trajectory starts without a formula, so a finished record is never formula-less. Every evidence number is a pseudo-label: mass-consistency statement only (no experimental fragment confidence, no atom mapping).

### 3.6 Packed records and the validated score domain

Ranked, compacted candidates: `B * R` records in `(spectrum, rank)` order (`R` the configured `returned`
count). Per record the integer fields of §3.5 plus the ranking `score`; per spectrum the §3.5 counters and
`formula_source`.

**Validated score domain** (by specification, not by code): a ranking term or score is "in the validated
domain" when it lies strictly inside (−3e38, 3e38); anything else (NaN, infinities, finite extremes at or
beyond the bound, overflowing sums) is treated as invalid and excluded from ranking. Finiteness tests are
not portable across shader backends, so kernels, host twins, host `pack` and validation all classify with
the same range test — never exact IEEE classification — against ONE shared bound
(`crate::tensor::ops::ms2::FINITE_MAX`; `pack::SCORE_FINITE_MAX` and `allocate::ALLOC_FINITE_MAX` are
aliases of it). The ranking score is the f32 sum of the f32-widened terms on every neural dtype, and the
packed score buffer is f32, so device words equal host `pack` exactly.

**Packed validation** (`PackedCandidateBatch::validate`): shapes; `1 <= R <= K`; rank order with every
filled score independently in the validated domain; the unfilled pattern; per-spectrum rules carried over
from `CandidateBatch` through the SAME source-specific counter function (`validate_search_counters`):
table source requires `joined <= visited`, enumeration allows `joined <= 4 * visited` (up to 4 hydrogen
counts per visited heavy vector — e.g. visited 1, joined 2, scored 2 is legal), saturated counters carry
`formula_search_exhausted`. Every FILLED slot (finished or not) has real formula provenance: a source, a
rank below `rows_scored`, counts that are not all zero and that replay (a finished graph with zero counts
and MAX row/rank is corrupt, not formula-less).

## 4. Chemistry domain V0 (P0.3)

### 4.1 Elements and masses

Monoisotopic masses of the most abundant isotope, rounded once to integer units of 10⁻⁶ u. The decimal strings
below are the definition. They are meant to be those of the NIST "Atomic Weights and Isotopic Compositions"
table, an attribution made from memory and not verified here. RDKit 2026.03.3's periodic table holds values
that differ by up to 1.1 µDa per atom (*measured*: I +1.1, Br −0.5, P −0.37, S −0.17, F +0.06, the rest under
0.01), so a cross-check against RDKit masses allows that much per atom and is not an equality test.

| Symbol | Exact mass (u) | Integer | Residual, nano-dalton, rounded up |
|---|---:|---:|---:|
| C | 12 | 12,000,000 | 0 |
| H | 1.00782503223 | 1,007,825 | 33 |
| N | 14.00307400443 | 14,003,074 | 5 |
| O | 15.99491461957 | 15,994,915 | 381 |
| F | 18.99840316273 | 18,998,403 | 163 |
| P | 30.97376199842 | 30,973,762 | 2 |
| S | 31.9720711744 | 31,972,071 | 175 |
| Cl | 34.968852682 | 34,968,853 | 318 |
| Br | 78.9183376 | 78,918,338 | 400 |
| I | 126.9044719 | 126,904,472 | 100 |
| electron | 0.000548579909065 | 549 | 421 |

The residual is `|exact * 10⁶ − integer|` in units of 10⁻⁹ u, rounded **up**, so a sum of residuals is a true
upper bound. No isotope labels and no isotope-peak modelling: an isotope-labelled structure is out of domain,
and isotope peaks in a spectrum are ordinary unassigned peaks.

### 4.2 Atom types

An atom type is `(element, parent hydrogen count, valence)`; all V0 atoms are neutral. The hydrogen count is
the one the atom has **in the parent molecule** and never changes; the valence is that count plus the atom's
bond orders in the kekulized parent. Hydrogens are not graph vertices: a structure with an explicit hydrogen
vertex is out of domain (*measured*: 18 structures). V0 keeps the 17 types found in at least 100 structures of
the train and validation folds (*measured*: the rarest kept is iodine with 306 structures there, the commonest
dropped has 10):

| Id | Type | Id | Type | Id | Type |
|---:|---|---:|---|---:|---|
| 1 | C H0 v4 | 7 | N H2 v3 | 13 | S H0 v2 |
| 2 | C H1 v4 | 8 | O H0 v2 | 14 | S H1 v2 |
| 3 | C H2 v4 | 9 | O H1 v2 | 15 | S H0 v6 |
| 4 | C H3 v4 | 10 | F H0 v1 | 16 | P H0 v5 |
| 5 | N H0 v3 | 11 | Cl H0 v1 | 17 | I H0 v1 |
| 6 | N H1 v3 | 12 | Br H0 v1 | | |

Id 0 is padding. Bonds are the kekulized orders 1, 2 and 3 (*measured*: 77.1%, 22.8% and 0.14% of bonds); there
is no aromatic bond type. The kekulization is RDKit 2026.03.3's `Kekulize(clearAromaticFlags=True)` of the
stored `normalized_smiles`; it is one choice among resonance forms, so two subgraphs that differ only by that
choice are different labeled graphs here (identity policy of §7.4).

### 4.3 Adducts and ions

| Id | Adduct | Charge | Molecules | Hydrogens added | Parent mass from precursor |
|---:|---|---:|---:|---:|---|
| 0 | unknown | — | — | — | `insufficient_metadata` |
| 1 | `[M+H]+` | +1 | 1 | +1 | `M = mz − m_H + m_e` |
| 2 | `[M-H]-` | −1 | 1 | −1 | `M = mz + m_H − m_e` |

General rule, for later domains: `M = (|z| * mz − delta) / n` with `delta = mass(atoms gained) − mass(atoms lost)
− z * m_e` for signed charge `z`. V0 has `|z| = n = 1` only, so no multiplication or division by a charge
occurs. The parent `M` is a neutral molecule; an intrinsically charged parent is out of domain. The parent
mass carries the precursor's uncertainty plus the adduct's arithmetic bound, 33 + 421 nano-dalton rounded up
to 1 unit.

A **fragment ion** hypothesis for a parent subgraph `g` under adduct `a` is `(g, s)` with integer hydrogen shift
`s`, and its m/z is

```text
mz(g, s) = mass(g with its parent hydrogens) + (h_a + s) * m_H − z_a * m_e,     |s| <= min(c(g), 2)
```

where `h_a` is the adduct's hydrogen count, `z_a` its signed charge and `c(g)` the number of parent bonds
between `g` and the rest of the molecule. A hypothesis whose ion would have a negative hydrogen count does not
exist. This is the only supported parent-to-ion mapping in V0; it states a mass relation and says nothing about
which atoms carry the shifted hydrogens. Rearrangements, charge-remote losses and non-hydrogen transfers are
unsupported.

### 4.4 Graph-action grammar (`grammar-bfs-v1`)

A token is `(kind, atom_type, bond_order, pointer)`. Kinds: `0` PAD, `1` START, `2` ADD_ATOM, `3` CLOSE_RING,
`4` STOP. Atoms are numbered `0, 1, …` in the order they are added; a pointer is such a number.

```text
START
ADD_ATOM(type)                       # the root: no bond, no pointer
ADD_ATOM(type, parent, bond)         # a new atom bonded to an existing one
CLOSE_RING(previous, bond)           # a bond between the newest atom and an earlier one
STOP
```

Factor order, each conditioned on the earlier ones: kind, atom type (ADD only), bond order (non-root ADD and
CLOSE), pointer (non-root ADD and CLOSE). A field a kind does not use is `0` and contributes no probability;
a token with a non-zero unused field is illegal.

The traversal is breadth-first, which makes these prefix rules hold for every target and lets the decoder mask
on them:

1. The first token is START and the second is the root ADD_ATOM.
2. A non-root ADD_ATOM's parent pointer is not smaller than the previous ADD_ATOM's parent pointer.
3. A CLOSE_RING's pointer is larger than the newest atom's parent pointer, larger than the previous CLOSE_RING
   pointer of the same atom, and smaller than the newest atom's own number.
4. No CLOSE_RING follows the root.

Chemical legality, checked on the same prefix:

5. A bond of order `b` needs residual valence `>= b` on both ends, where residual valence is
   `valence − parent hydrogens − sum of retained bond orders`. No bond is added twice.
6. ADD_ATOM is illegal at `A` atoms; CLOSE_RING is illegal at `R_max` closures.
7. With a formula hypothesis, ADD_ATOM is illegal for a type whose element count, or whose hydrogens, would
   exceed the parent formula's.
8. STOP is legal once at least one atom exists. Nothing is legal after STOP; START and PAD are never legal
   after position 0.

A kind is legal only if some completion of its fields is; that includes the root, whose ADD_ATOM is legal only
if some atom type fits the formula budget. A prefix with no legal action other than STOP emits STOP, and a
prefix with none at all (possible only at the root) sets `no_valid_action`. With `A = 16` and `R_max = 4` the longest trace is `T = 2 + A + R_max = 22` tokens.

### 4.5 Partial and final validity, hydrogens and attachments (P1.8 inputs)

- **Partial validity**: every atom's residual valence is non-negative and the rules above held at every step.
- **Final validity** (at STOP): partial validity, at least one atom, at most `A` atoms and `R_max` closures,
  connected (guaranteed by the grammar), composition within the formula hypothesis when one is given.
- A molecule or subgraph is limited to 4,096 atoms on the host (larger inputs are `Error::Config`), which keeps
  every element count inside `u16`.
- **Hydrogens** are part of the atom type and are counted once, in the composition and in the mass. They are
  never added to cap an open valence.
- **Open attachment valence** of an atom at STOP is its residual valence. It records missing bonds to the rest
  of the parent and is not an atom. A residual of 2 is reported as the number 2 with
  `attachment_partition = unknown`: one double bond and two single bonds are both possible and neither is
  chosen. The sum of open valences is an upper bound on `c(g)` and equals it only if every missing bond is single.

### 4.6 Unsupported in V0, and the V1 domain

Out of the V0 structure domain (*measured*, structures; one structure can have several reasons): formal
charges including nitro and quaternary nitrogen 4,826; elements outside §4.1 80; explicit hydrogen vertices 18;
atom types outside §4.2 30; isotope labels 22; radicals 9; disconnected structures 0. In domain: 272,609 of
277,566 structures (98.2%).

Out of the V0 request domain, first reason in the order adduct, polarity conflict, structure, precursor range
(*measured*, spectra): adduct other than §4.3 749,335 (29.5%); polarity conflict 5,933 (0.23%); structure 34,226
(1.35%); precursor range 63. In domain: 1,750,051 of 2,539,608 spectra (68.9%). A spectrum's structure is the one of its own row
(`normalized_smiles`), not of its InChIKey: 1,612 keys cover more than one stored tautomer. Reports keep all 2,539,608 in
the full-dataset denominator.

Also unsupported: multiply charged ions, multimers, in-source losses, stereochemistry, aromatic or resonance
equivalence.

**V1 domain (provisional target, not frozen and not implemented).** Adducts: the ten of the hidden test set (`[M+H]+`,
`[M+NH4]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+Na]+`, `[M+K]+`, `[M-H]-`, `[M-H2O-H]-`, `[M+CH2O2-H]-`,
`[M+Cl]-`), each with its composition delta in the general rule of §4.3. Structures: the same elements, formal
charges −1, 0, +1 with their own atom types, every atom type found in the train and validation folds.
`N = 256`, `A = 32`, `R_max = 8`, `T = 42`. *Measured* request coverage of that broader domain, counting every atom type whether or not a later
vocabulary keeps it: 2,169,722 spectra (85.4%), an upper estimate; the rest is adducts outside the ten (14.3%).
The V1 vocabulary and the ion rules for sodium, potassium, ammonium, formate and chloride adducts are frozen
when V1 starts, with their own audit; V0 code must not assume
`|z| = n = 1` anywhere but in the two adduct rows.

## 5. Exact-mass arithmetic

- Scale: `1_000_000` integer units per dalton, unsigned 32 bit. Range `0 … 4294.967295` Da; the V0 request
  range is 50 to 2000 Da. Every sum is checked; overflow is an error on the host and `mass_overflow` on the
  device.
- Tolerance: `tol(mz, t) = floor(mz * t / 10^7)` for `t` in tenths of a ppm, `t <= 1000`, taken at the observed
  m/z. In 32 bits: `hi = mz / 10^4`, `lo = mz % 10^4`, `a = hi * t`, `tol = a / 1000 + ((a % 1000) * 10^4 + lo * t)
  / 10^7`; `a <= 429,496,000` and the second numerator is at most `19,989,000`.
- Arithmetic bound of an ion with composition `n_e` (its own hydrogen count, after adduct and shift) and one
  elementary charge: `E_arith = ceil((sum_e n_e * residual_nda_e + 421) / 1000)` integer units. For a 16-atom
  subgraph that is under 9, against a 10 ppm tolerance of at least 500 at m/z 50.
- Observation bound: `U = mz_uncertainty_udalton` of the spectrum, the half-width of the stored value's
  rounding interval (50 for 4 decimals). The integer field itself adds nothing for inputs with at most 6
  decimals. `E = E_arith + U`.
- Decision rule: with `r = |mz_observed − mz_computed|`, accept when `r + E <= tol`, reject when `r > tol + E`,
  otherwise `mass_boundary_ambiguous`. All quantities are integers. An ambiguous hypothesis is neither a label
  nor a rejection; it is counted. V0 implements no higher-resolution retry.
- The unknown-precision sentinel (`U = u32::MAX`) disables exact-mass decisions for that spectrum: no
  hypothesis is evaluated, so none is accepted and none is counted as ambiguous.
- The tolerance is the caller's statement of instrument accuracy and is not widened. `U` covers only the
  storage rounding of the value the caller supplied; calibration error beyond the stated tolerance is the
  caller's, and is why the tolerance is a per-request field.
- A spectrum whose `U` exceeds its tolerance at every peak can have no accepted hypothesis (*measured*: 19 of
  the 300 pilot spectra store 3 or fewer decimals; a 3-decimal `U` of 500 equals the 10 ppm tolerance at m/z 50).
- Neural features use `f32(mz_udalton) * 1e-6` and never feed an accept/reject decision.

## 6. Dataset-level conventions (P0.4)

| Quantity | Convention | Evidence (*measured*) |
|---|---|---|
| Fragment tolerance | 10 ppm default (`fragment_tolerance_ppm_tenths`) | §7 pilot grid |
| Precursor tolerance | 20 ppm default (`precursor_tolerance_ppm_tenths`) | In-domain precursors within 5 / 10 / 20 / 50 ppm of the structure: 93.4% / 95.9% / 97.5% / 98.2% |
| Collision energy unit | eV, from `collision_energy_ev`; the mean when several | Original units: eV 53.9%, NCE 31.1%, unknown 12.8%, V 2.2%. The file holds an eV value for 98.5% of the NCE rows and 98.4% of the V rows |
| Unknown energy | `collision_energy_known = 0`; the value is not read | 13.3% of spectra have no eV value |
| Energy count | `len(collision_energy_ev)`; `0` unknown; capped at 8 = "8 or more" | Per spectrum: 1 (67.2%), 3 (15.5%), 0 (13.3%), 4 (3.3%), 2, 5, 6. Only merged inputs can reach the cap |
| Merge policy | V0 trains, evaluates and accepts single spectra only; merged inputs are unsupported. FPNet's own merge is recorded here as what FPNet does, not as a contract of this project: per-spectrum normalisation, peaks within 0.005 Da merged keeping the stronger, `float32` output, precursor the median, adduct the mode, energy the mean of known values, count the capped sum. A future merged adapter must define its own clustering, tie rules and precision before it exists | `spectra.py:307-334`, `engine.py:346-359` |
| Precursor sanity | Finite and within 50 to 2000 Da | The file holds values from 2.0 to 2,430,333 |

The eV value of an NCE-derived row depends on the data provider's conversion, which this project cannot
verify. It is used as given; the instrument holdout of §1 is the check on that.

## 7. Targets (P0.7)

### 7.1 Domains and denominators

- **Full dataset**: every spectrum of the subset. Out-of-domain and unlabeled spectra count as misses.
- **V0 domain**: §4.6. Conditional metrics are reported next to the full-dataset ones, never instead of them.
- **Labeled**: in-domain spectra with at least one target under §7.2 (*measured* 82.3% in the pilot, a
  spectrum-weighted sample of one row group). *Measured* 2026-10-03 on the molecule-weighted V0 exports
  (`export_casmi.py`, `uniform-in-domain-v2`: molecules drawn uniformly, then each molecule's spectra drawn
  uniformly from its in-domain rows): 95.6% of 3,685 pilot-train and 96.4% of 740 pilot-validation spectra are
  labeled, with about 10 targets per spectrum. The difference is the weighting, not the recipe: the median
  spectrum has 42 peaks over all rows, 55 over the two V0 adducts and 141 when each molecule counts once
  (molecules with few spectra have rich ones). More peaks also mean more chance matches, so the unrelated-
  molecule diagnostic below, measured on the spectrum-weighted sample, is a lower bound for the V0 exports and
  the shuffled-spectrum control is the operative check. The first exporter took each molecule's first rows in
  file order (median 174 peaks); its exports were replaced.

### 7.2 Pseudo-label recipe `q-cut-v1`

1. **Input.** The kekulized parent graph (§4.2); the spectrum's peaks filtered as in §2 without the cap
   (`0 < mz <= precursor + 2`, relative intensity `>= 1e-3`); weights use the **linear** relative intensity.
2. **Candidates.** Every set of at most 2 parent bonds whose removal leaves each removed bond between two
   different components; each component with 3 to 16 atoms and at most 4 ring closures (`bonds − atoms + 1`)
   is a candidate **embedding** `g`. The empty cut set makes the whole molecule a candidate when it fits. Every
   candidate is an induced connected subgraph, and `c(g)`, the number of parent bonds leaving it, is a property
   of its atom set.
3. **Identity.** Embeddings are grouped into **graphs** by canonical trace (§7.4). An embedding whose
   canonicalization exceeds the work limit is dropped before any matching and counted
   (`canonicalization_budget_exceeded`).
4. **Matching.** A peak is explained by an embedding `g` if some shift `|s| <= min(c(g), 2)` gives an ion
   `mz(g, s)` (§4.3) the decision rule of §5 accepts. Ambiguous hypotheses are counted and explain nothing.
   A peak is explained by a graph when any of its embeddings explains it.
5. **Weights**, in integers so that ties are exact. A peak of relative intensity `r` has
   `I = floor(r * 2^20 + 0.5)` units; explained by `n` graphs, it gives each `floor(I * M / n)` with
   `M = 2,952,069,120` (`4096 * lcm(1..16)`, so the split is exact up to 16 graphs). `w(G)` is the `u64` sum
   over peaks.
6. **Retention.** Keep the 16 graphs of largest `w`, comparing the integers, ties by canonical trace (smaller
   first); `q(G) = w(G) / sum of kept w` in `f64`; record the dropped fraction of `w`. *Measured*: the cut falls
   between two graphs of equal weight in 14 of the 300 pilot spectra, so the tie rule is not a corner case.
7. **Evidence anchors.** Each kept graph stores the sorted `(peak_id, s)` pairs for which at least one of its
   embeddings was accepted.

Provenance stored with every label file: recipe, chemistry, grammar and traversal versions, tolerance, RDKit
version, source file hash, source row.

*Measured* on the pilot sample (linear intensity; "unrelated" matches the same peaks against the subgraphs of
the next sampled molecule; "shared" is the part of the explained intensity that the unrelated molecule also
explains):

| Cuts | ppm | Intensity explained (mean) | Unrelated | Shared | Peaks explained | Spectra labeled | Targets (p50 / p95) | Over 16 targets | Weight dropped (mean / p95) |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 10 | 27.3% | 1.2% | 1.0% | 4.9% | 68.0% | 1 / 6 | 0% | 0 / 0 |
| 2 | 5 | 34.1% | 4.0% | 2.8% | 14.0% | 75.3% | 4 / 23 | 13.3% | 0.5% / 1.7% |
| **2** | **10** | **41.2%** | **5.0%** | **3.6%** | **17.0%** | **82.3%** | **5 / 27** | **15.3%** | **0.7% / 3.9%** |
| 2 | 20 | 43.4% | 5.9% | 4.2% | 19.2% | 85.7% | 6 / 30 | 16.7% | 0.9% / 4.8% |
| 3 | 10 | 49.0% | 9.5% | 8.2% | 28.0% | 85.7% | 16.5 / 91 | 50.0% | 6.5% / 29.2% |

Reading: at the frozen setting, of every 41 points of explained intensity about 3.6 are also explained by an
unrelated molecule's subgraphs, and 11.5% of the explained peaks are. That is a diagnostic of how unspecific a
mass match is, not a measured false-label rate: a peak both molecules explain may still be correctly labeled.
Three cuts explain 8 more points of intensity, more than double the shared part, and put half of the spectra
over the 16-target cap, so V0 uses two. Most peaks are unexplained: 17.0% of the peaks carry the 41% of intensity.

Retained targets at the frozen setting (1,984): sizes spread over 3 to 16 atoms; ring closures 0 (37.7%),
1 (46.2%), 2 (14.6%), 3 (1.5%), 4 (0.15%), the recipe's cap; boundary bonds 0 (2.2%), 1 (25.2%), 2 (72.6%).

The pilot's graph identity is RDKit canonical fragment SMILES with each atom tagged by its type id, not the
canonical trace; P1.9 reports the Rust reference's counts on the same rows (stored in the report).

### 7.3 Missing labels and the two objectives

- A spectrum with no target contributes no graph loss and is not a negative example. It stays in the
  denominators of §7.1.
- **Containment** (primary): a candidate graph is correct if it is an **induced** labeled subgraph of the true
  parent: an injective map of its atoms into the parent's that preserves atom type, and under which two
  candidate atoms are bonded with order `b` exactly when their images are. This needs only the parent
  structure, and is decided offline on the host.
- **Ion assignment**: agreement with the pseudo-label anchors of §7.2. Reported separately and always named a
  pseudo-label metric.

### 7.4 Identity and canonical traversal (`bfs-canon-v1`)

Two graphs are identical when a bijection of their atoms preserves atom type and bond order. Aromatic and
resonance equivalence are not applied.

The canonical trace of a graph is the lexicographically smallest token sequence over all breadth-first
traversals that satisfy §4.4: every choice of root, and every order in which the atom at the head of the queue
discovers its neighbours. Tokens compare as `(kind, atom_type, bond_order, pointer)` tuples. Each new atom emits
its ADD_ATOM, then one CLOSE_RING per bond to an earlier atom other than its parent, in increasing pointer
order. The search is branch-and-bound over the tied choices with a work limit of 200,000 expansions.

Two graphs are identical exactly when their canonical traces are equal, because the trace encodes the labeled
graph without loss and isomorphisms preserve the set of traversals. P1.2 checks the implementation against an
exhaustive search, against RDKit canonical fragment SMILES with every atom tagged by its type id (an untagged
SMILES cannot tell sulfur of valence 2 from valence 6), and against atom permutations.

## 8. Statuses

Request status, `u32` bit set per spectrum. Bits 0–15 are fatal: the spectrum yields no candidate. Bits 16–31
are warnings: generation proceeds.

| Bit | Name | Bit | Name |
|---:|---|---:|---|
| 0 | `empty_spectrum` | 8 | `precursor_out_of_range` |
| 1 | `nonfinite_input` | 9 | `over_capacity` |
| 2 | `negative_intensity` | 10 | `formula_absent` (no table row in the precursor window) |
| 3 | `invalid_peak` | 11 | `mass_overflow` |
| 4 | `invalid_polarity` | 16 | `raw_truncated` |
| 5 | `polarity_adduct_conflict` | 17 | `exact_mass_unavailable` |
| 6 | `insufficient_metadata` | 18 | `formula_search_exhausted` |
| 7 | `unsupported_adduct` | 19 | `peaks_truncated` (device cap `N`) |

`exact_mass_unavailable` on the precursor makes formula search impossible; the request then also carries
`formula_absent`. With `oracle_formula` the formula bits are not set.

Candidate status, `u32` bit set per trajectory: `0` `finished`, `1` `truncated` (T reached without STOP),
`2` `no_valid_action`, `3` `invalid_final`, `4` `duplicate_trace` (set on every trajectory after the first
with the same trace and conditioning formula composition — for enumeration the 10 formula counts, since
`formula_row` is `u32::MAX` for every enumerated formula), `5` `formula_source_oracle`, `6` `request_failed`
(no trajectory started; every record of a spectrum with no scored formula carries it with `length = 0`,
see §9 — this includes exhausted/overflow cases whose request status alone is not fatal), `7`
`duplicate_graph` (V1 §4.2: an exact comparison proved equality with an earlier trajectory of the same
spectrum; the candidate stays, ranked out), `8` `identity_unresolved` (V1 §4.2: some comparison of this
(the later) trajectory ran out of budget; not a duplicate flag, the candidate stays eligible for ranking).

Evidence status, `u8` per trajectory (§2.4): `0` unassigned (no such peak), `1` mass-consistent,
`2` mass-consistent for every boundary count, plus bit 7 (`128`, `evidence_support_incomplete`) when the
assignment support behind the status is incomplete. Every evidence number is a pseudo-label mass-consistency
statement (no experimental fragment confidence, no atom mapping).

Label preparation (offline, per embedding or spectrum): `canonicalization_budget_exceeded`,
`mass_boundary_ambiguous` (count of hypotheses), `no_target`.

A request error never yields an arbitrary structure.

## 9. Search work, capacities and feasibility (P0.8)

**Formula search** (`FormulaTable::window`; the device kernel reproduces it step for step), per spectrum:

1. `parent = precursor − h_a * m_H + z_a * m_e`. If that leaves the `u32` range the request is `mass_overflow`
   (fatal) and nothing is searched. An unknown precursor precision (`u32::MAX`) is `exact_mass_unavailable`
   and `formula_absent`, with nothing searched.
2. `tol = tol(precursor, precursor tolerance)`, `bound = precursor uncertainty + 1` (the adduct's arithmetic
   bound). The **superset window** is `parent ± (tol + bound + E_table)`, `E_table` the largest row bound of
   the table; its ends are found by two halving searches over the sorted masses (lower bound of the low end,
   upper bound of the high end), saturating at 0 and `u32::MAX`.
3. Every row of the superset window gets the verdict of §5 with `E = E_row + bound`. Accepted and ambiguous
   rows are **joined** (an ambiguous row is flagged); rejected rows are not.
4. `rows_visited` counts every mass read: one per halving step, one per row of the superset window.
   `rows_joined` counts joined rows. The first `min(rows_scored_max, M)` joined rows in table order are
   **scored**; `rows_scored` counts them.
5. A search that would need more than `rows_visited_max` reads stops there and is **exhausted**: it keeps the
   counters and joined rows it has, and the request carries `formula_search_exhausted`. More joined rows than
   can be scored is exhausted too.
6. `formula_absent` (fatal) means the search **completed** and joined no row. `formula_support_complete` is `1`
   exactly when the search completed and every joined row was scored; only then are the formula probabilities
   and `formula_mass_retained` statements about the whole window. Equal `rows_scored` and `rows_joined` alone
   do not show that.

**Enumeration source** (V1 §1.4): `rows_visited` counts visited heavy `(C, N, O)` vectors, `rows_joined`
counts joined candidates (each vector admits up to 4 hydrogen counts, so `rows_joined <= 4 * rows_visited`);
`rows_scored = min(rows_joined, formula_rows_scored_max, M)` with `rows_scored <= rows_joined` as for the
table source. `visited` and `joined` saturate at `u32::MAX - 1`; a saturated counter sets
`formula_search_exhausted`, clears `complete`, and is then a lower bound.

**Incomplete enumeration searches** (V1 §1.4): `formula_absent` is set only for a **completed** search that
joined nothing, plus the explicit unknown-precision rule (unknown precursor precision:
`exact_mass_unavailable` + `formula_absent`, `complete = 0`, nothing searched). A too-wide window
(`half > 1,511,737`) is `formula_search_exhausted` with `complete = 0` and no `formula_absent`; arithmetic
overflow is `mass_overflow` with `complete = 0` and no `formula_absent`. A spectrum with no scored formula
for any reason abstains: no trajectory starts and every record carries `request_failed` (`length = 0`);
such records are valid exactly when `rows_scored == 0` (completed absence carries the fatal `formula_absent`,
overflow carries fatal `mass_overflow`, wide-window and other exhausted-no-formula cases carry
`formula_search_exhausted` with `rows_scored == 0`).

**V0 formula table**: the distinct molecular formulas of the in-domain train-subset structures, sorted by
integer mass. A row is 10 element counts (`u16`) and a mass (`u32`): 24 bytes. *Measured*: 37,859 rows from
163,571 structures, 908,616 bytes. A 20 ppm window around a table mass holds 6 rows at the median, 16 at the
95th percentile and 24 at most. 86.7% of in-domain validation-fold structures have their formula in the table;
the other 13.3% are `formula_absent` gold by construction of a train-only table, counted and never hidden.

**Vocabulary and trace**: 17 atom types, 3 bond orders, 5 kinds, pointers below `A = 16`; `R_max = 4`;
`T = 22`; at most 16 targets per spectrum. V1 §3.1 raises the configuration
caps to `A <= 32`, `R_max <= 8`, `T <= 64` with 4 decoder blocks; the V0
shapes above stay the defaults and every V0 result is reproducible at them.

**V0 shapes**: `B = 8`, `N = 128`, `K = 8`, `F <= 4`, the `ModelConfig` of §3.3 (SISO, rotational, no
convolution, so the carries are `h`, `last_u` and `angle` only):

| Buffer | Elements | FP32 bytes |
|---|---:|---:|
| Decoder carry per trajectory and layer: `h` + `last_u` + `angle` | `2 * 4 * 64 * 32 + 4 * 16 = 16,448` | 65,792 |
| Decoder carries, `B * K * 2` layers | 2,105,344 | 8.03 MiB |
| Encoder output and per-layer K/V, `B * N * d * (1 + 2 * 2)` | 655,360 | 2.5 MiB |
| Formula table | 37,859 rows | 0.87 MiB |
| Graph state per trajectory: 16 atom records, 16×16 bond orders, counters | under 400 | under 0.1 MiB total |

These are byte counts of the listed buffers from their shapes. They omit weights, activations, scan
workspace, attention scores and training state, and are reconciled with measured allocations in P2.2; they
show only that the recurrent state does not decide `B` and `K` at V0 sizes.

## 10. Confidence target, metrics, baselines and devices (P0.6)

- **Confidence target**: containment of the candidate in the true parent (§7.3). V0 exposes raw scores
  (`formula_log_prob + trace_log_prob`) and no calibrated probability.
- **Metrics** at K returned candidates, each also in three size strata (3–5, 6–9, 10–16 atoms): precision
  (fraction of finished, distinct-trace candidates contained in the parent), target coverage (the `q` mass of
  the targets found among the candidates), validity, uniqueness by trace, and formula recall at F. A size-aware
  summary weights each correct candidate by its atom count over 16, so a trivial fragment cannot carry the
  score.
- **Aggregation**: per spectrum, then the mean over the spectra of a molecule, then the mean over molecules;
  intervals by bootstrap over molecules.
- **Abstention**: a spectrum may return fewer than K candidates or none. Abstentions count as misses in the
  full-dataset denominator and are reported as a rate.
- **Controls** (V0.5), run through `GenerationConfig::control` and the same flag at training time:
  `ShuffledSpectrum` gives each spectrum the peaks of another spectrum of the batch while keeping its own
  metadata and targets; `MetadataOnly` replaces the peak memory by the single metadata token, bypassing the
  `empty_spectrum` abstention, which exists for real requests only. The **structure prior** is the model
  trained with `MetadataOnly` whose encoder additionally sees the unknown rows of every metadata embedding
  and a zeroed energy and precursor feature. Only the encoder is blinded: the request still carries its real
  adduct and precursor, which the formula search and the grammar's budget need, so request validation is
  unchanged.
- **Baselines** for the release evaluation (P9.3): FPNet where its outputs apply, a set encoder, and a
  parameter-matched Transformer, at matched candidate budgets. V0 compares against the controls only.
- A success threshold is not chosen after seeing results; the comparison is reported with its intervals.
- **Devices**: the CubeCL CPU runtime and the Apple M1 through wgpu/Metal are the mandatory targets. A CUDA
  device (Colab T4) is optional and used for CUDA-specific measurements only.

## 11. Open items carried forward

- The eV conversion of NCE rows is the data provider's (§6).
- The pseudo-labels are mass matches; §7.2's unrelated-molecule diagnostic bounds what an ion-assignment
  metric can mean.
- The kekulization choice makes some chemically equal subgraphs distinct (§4.2); the duplicate rate is to be
  measured in P1.9.
- A train-only formula table caps formula recall on held-out molecules near 87% (§9); P4 replaces it with a
  search over compositions.
- The attribution of the element masses to a named table is unverified (§4.1).
