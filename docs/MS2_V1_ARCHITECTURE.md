# MS2 V1 architecture: production formula search, ion assignment, full decoding and outputs

Status: implementation specification for P4 to P7 of [MS2_SUBSTRUCTURE_TASKS.md](MS2_SUBSTRUCTURE_TASKS.md), revision 3
(2026-10-03). Revisions 1 and 2 were reviewed by codex ([reviews](reviews/MS2_V1_ARCHITECTURE_CODEX_REVIEW.md), both: not
ready); revision 3 answers the second review's list. Remaining disagreements are settled in code review of each
work package, not by further revisions of this text. It extends [the V0 architecture](MS2_V0_ARCHITECTURE.md) and uses the names, units and
statuses of [the contracts](MS2_CONTRACTS.md). A statement marked *measured* names its report; everything else is a
shape or a rule, not a result. Sections are added as their phase starts; a section that does not exist yet is not
specified.

V1 keeps V0's rules: a new kernel only where no primitive does the job, a host twin with the same arithmetic for
every kernel, every launch through `launch_1d` / `launch_1d_spans`, at most 6 array bindings per kernel, every
output element written, selection by comparison and never by multiplying with a mask, no device read between the
upload and the one final read.

## 1. Formula candidates (K03; P4.1, P4.3, P4.9)

### 1.1 Why the table is replaced

*Measured* on the MassSpecGym pilot export (714 validation spectra of 384 molecules held out by structure, 20 ppm
precursor tolerance; `examples/ms2_formula_report.rs`, reports under `bench/results/ms2/formula_sources_*.json` once
the pruned enumeration is measured): the V0 table of train-fold formulas contains the gold formula of 23.7% of
validation spectra, a table of train-fold plus ChEBI 3-star formulas 37.0%, and bounded enumeration with the exact
filters of §1.4 98.7%, at a median of 1,548 candidates per spectrum against 2. A formula source for
structure-disjoint data has to enumerate, and has to prune. §1.2 and §1.3 make everything after the search
independent of where a candidate came from; §1.4 adds the enumerating source.

### 1.2 Candidate compositions

Everything downstream of the search consumes **compositions**, not table rows.

**Schema version 2** (contracts §3 gains the same text). `GenerationConfig`, `CandidateBatch` and `ModelConfig`
move to `schema_version = 2`. A version-1 document still loads: a version-1 `GenerationConfig` takes the "version-1
value" column below; a version-1 `CandidateBatch` has the three new fields **absent** (empty arrays, accepted by
`validate` for version 1 only — they cannot be reconstructed without the table); a version-1 `ModelConfig` names a
formula table only and has no enumeration artifacts and no assignment head (assignment disabled, §2). Any other
version is `Error::Config`, as before. Device restrictions introduced by V1 apply whatever the document's version:
a table row with an element count above 1023 is refused at upload (*measured* maxima in the MassSpecGym train fold:
H 96, C 60, F 34).

| Schema | New field | Type | Version-1 value | Meaning |
|---|---|---|---|---|
| `GenerationConfig` | `formula_source` | enum | `Table` | `Table` or `Enumerate` (§1.4; `Error::Unsupported` until implemented) |
| `GenerationConfig` | `formula_window` (M) | `u32` | 32 | Scored-candidate capacity per spectrum: one of 32, 128, 512, 2048; anything else is `Error::Config` |
| `CandidateBatch` | `formula_counts` | `u16 [B*K, 10]` | — | Composition of the conditioning formula in `ELEMENTS` order; all `0` when there is none |
| `CandidateBatch` | `formula_source` | `u8 [B]` | — | `0` table, `1` enumeration |
| `CandidateBatch` | `formula_rank` | `u32 [B*K]` | — | Rank of the formula in the scored support of its spectrum (its window slot); `u32::MAX` when there is none |

`formula_row` keeps its meaning — a row of the resident table — and is `u32::MAX` for an enumerated formula, so a
version-1 consumer can never look an enumeration rank up in a table. `formula_rows_scored_max` keeps its meaning;
`rows_scored = min(rows_joined, formula_rows_scored_max, M)`.

**Device buffers** of one `(B, M, F)` bucket, allocated once in the workspace:

| Buffer | Shape | Contents |
|---|---|---|
| `window` | `u32 [B, M, 2]` | V0, table source only: row (`u32::MAX` padding), flag |
| `counters` | `u32 [B, 5]` | V0: visited, joined, scored, request status bits, complete |
| `cand` | `u32 [B, M, 13]` | Per scored candidate: 10 element counts in `ELEMENTS` order (C, H, N, O, F, P, S, Cl, Br, I), integer mass, flag (`0` none, `1` accept, `2` ambiguous), source id (table row; `u32::MAX` for an enumerated candidate). A padding slot is all `0` except source id `u32::MAX` |
| `cand_feat` | float `[B, M, 10]` | `ln(1 + count)`; exact `0` in padding |
| `top` | `u32 [B, F, 2]` | V0: source id, window slot (`u32::MAX` padding) |
| `top_counts` | `u32 [B, F, 10]` | Counts of the retained formulas; `0` in padding |
| `top_log_prob`, `top_count` | V0 | Unchanged; a padding entry's `top_log_prob` is `0` and is never summed (`formula_mass_retained` sums the first `top_count` entries only) |

**Kernels** (arrays bound, all at most 6; each has a host twin, poisoned-output tests and a launch-error check):

- `ms2_formula_window`: unchanged (table, meta, window, counters: 4).
- `ms2_formula_gather`, lane per `(b, m)`: `window`, `table [R, 2]`, `table_counts [R, 10]` → `cand` (4). Table
  source only.
- `ms2_count_features`, lane per `(record, e)`: a `u32` record buffer of any record width `w >= 10` (a scalar
  argument; the counts are a record's first 10 words: `cand` with `w = 13`, `gold_counts` with `w = 10`, `ion`
  with `w = 12`), `log_table [1024]` → float `[records, 10]` (3). `log_table[n]` is the
  host's `(1.0 + n as f32).ln()` uploaded once, so the feature of a count is the **same bits** as V0's uploaded
  table features on every backend; no logarithm is evaluated on the device. A count above 1023 cannot occur: the
  table upload and every enumeration artifact are validated against that bound on the host (`Error::Config`).
- `ms2_formula_top`: reads `log_prob` and `cand` instead of `window` (the source id is `cand[.., 12]`); outputs as
  V0 (log_prob, cand, top, top_log_prob, top_count: 5). The selection rule is V0's (architecture §3.4), unchanged.
- `ms2_formula_top_counts`, lane per `(b, f)`: `top`, `cand` → `top_counts` (3).
- `ms2_gold_slot`, lane per spectrum: `cand`, `gold_counts [B, 10]` → `gold_slot [B]` (3): the first slot below
  `rows_scored` whose 10 counts equal the gold composition, else `u32::MAX`. Training and teacher-forced
  evaluation only; replaces the host-side window search of V0, whose function stays as this kernel's twin.
- `ms2_init_trajectories`: reads the budgets from `top_counts` instead of looking `top` up in the table
  (top, spectra_meta, top_counts, traj_meta, state, actions: 6). Trajectory `k` still uses formula
  `k mod top_count` (P5.5 changes that rule, not this section).

The formula head embeds `cand_feat` with the V0 weights (`Linear(10 → d)`, SiLU, `Linear(d → d)`): a V0 checkpoint
loads unchanged and, with `formula_source = Table` and `M = 32`, every integer output (window, counters, top,
actions, statuses) equals V0's exactly and every float output is produced by the same operations on the same
bits. The V0 test suites are the regression gate for that statement.

**Inference boundary.** Generation conditions only on `top_counts` and the head embeddings of the retained
formulas. No gold composition, target or anchor is an input of `generate`; a test builds two requests that differ
only in their (unused) gold payload and requires identical candidate batches.

**Teacher forcing.** The decoder conditions on the head embedding of the **true parent composition**
(`gold_counts`, uploaded with the targets, through `ms2_count_features` and the head's row network), whether or
not the search scored it. V0 used the zero vector when the gold row was outside the window (76% of MassSpecGym
validation spectra with the train-only table); `TrainConfig::gold_formula_conditioning` selects
`Composition` or `ScoredRowOrZero` (V0). A serialized training configuration without the field means
`ScoredRowOrZero`, so a V0 run is reproduced; new runs choose `Composition` explicitly (the experiment driver's
default). This is conditional
training: it says nothing about generation with predicted formulas, which is evaluated separately and reported
next to the oracle-conditioned numbers, with search misses in the denominator.

**Formula loss.** `L_formula` is the cross-entropy to `gold_slot` over the scored support, for spectra whose
`gold_slot != u32::MAX`, divided by `max(1, their count)`; a spectrum whose gold is not in the scored support
contributes nothing and is counted as `gold_not_scored`, split by cause when the cause is known on the host
(outside the source's domain; rejected by a pruning stage; beyond the scored cap). `gold_not_scored` is a training
metric; the request status `formula_absent` keeps its contract meaning (a completed search that joined nothing)
and is never used for it. Accumulation over microbatches sums the numerator and the eligible count and divides
once.

### 1.3 Capacity instead of tiles

Revision 1 scored the support in tiles with a mutable running partition. Revision 2 does not: the support of one
spectrum is one window of capacity `M`, chosen per bucket from {32, 128, 512, 2048}, and the probabilities are a
masked `log_softmax` over that window, which **is** the log-partition of the scored support. The memory that
matters is the embedding activations, `B * M * d` floats per layer of the head (16 MiB at `B = 8`, `M = 2048`,
`d = 128`, FP32), and tiles would not reduce what training has to retain; the memory estimate gains the items
`cand`, `cand_feat`, the head activations (3 `[B, M, d]` tensors in generation; with their gradients in training)
and the scores. A request whose estimate exceeds `max_device_bytes` is refused before allocation.

More joined candidates than `min(formula_rows_scored_max, M)` sets `formula_search_exhausted` and
`formula_support_complete = 0`: the probabilities are then over a prefix of the support in the source's order and
are reported as such (contracts §9). Launches of the formula stage are constant per bucket.

### 1.4 Bounded enumeration with closed-form hydrogen (`Enumerate`)

Revision 2 (answers review findings 1, 2, 3, 6, 9, 18). Host reference: `models/ms2/formula_enum.rs`
(`EnumDomain`, `RatioBounds` version `ms2-ratio-v1`, `enumerate`), reviewed and tested against brute force.

**Artifacts.** `EnumDomain` (per-element caps, heavy-atom total, hydrogen bounds) and `RatioBounds` (per-element
caps per carbon bucket and per heavy-total bucket, six cross-multiplied element ratios, the rare-heteroatom total
and distinct-count ranges, a twice-DBE interval per heavy-total bucket; buckets of 4; a bucket the fit never saw
admits nothing) are fitted on **train** compositions only, serialised with their version and SHA-256, and named by
`ModelConfig::formula_artifacts` next to the formula table. Before upload the host validates them with checked
64-bit arithmetic and refuses (`Error::Config`) an artifact that breaks any bound the device arithmetic relies on:
every cap `<= 255`, hydrogen bounds `<= 1023`, every ratio numerator and denominator `<= 2^20`, bucket tables of at
most 64 rows, at most `P_max = 16,384` rare combinations.

The exact filters use the maximum valences C4 N3 O2 F1 P5 S6 Cl1 Br1 I1. With counts `n`, twice the ring-and-double-
bond equivalent is `2 + 2 n_C + n_N + 3 n_P + 4 n_S − (n_H + n_F + n_Cl + n_Br + n_I)`; the device keeps the
positive and the negative total apart and compares them, so no unsigned subtraction can wrap. The three exact
stages are `n_H <= 2 + 2 n_C + n_N + 3 n_P + 4 n_S − n_hal` (hydrogen ceiling), an even difference (parity) and
`positive >= negative` (DBE). They are necessary conditions for a complete, connected, neutral parent of the V0
domain, not feasibility certificates.

**Order.** The rare-element combinations `(F, P, S, Cl, Br, I)` allowed by the domain caps and the rare ranges are
serialised once on the host in increasing lexicographic order of that tuple, without duplicates: `rare [P, 8]`
(6 counts, their mass, their count sum). A combination whose mass does not fit `u32` cannot lie in any window
and is not a row; `P` counts the rows. Combination 0 is all zeros when the rare ranges allow it. The query's
tolerance is the contract's (`ppm_tenths <= 1000`, else `Error::Config`), so the tolerance never wraps. A spectrum has `P` lanes; lane `r` enumerates,
in this nesting and each ascending, `C`, then `N`, then `O`, then `H`. The **joined rank** of a candidate is the
number of joined candidates of the same spectrum in earlier lanes plus its index among its own lane's joined
candidates. The scored support is ranks `0 .. min(joined, formula_rows_scored_max, M)`; `cand` slot `m` holds rank
`m`. `enumerate_device_order` in `formula_enum.rs` — written first, before any kernel — is the host twin of exactly
this order, including the cut-offs below; a test shows that its joined **set** equals that of the reviewed
`enumerate` whenever neither is exhausted, and the formula report measures gold recall of its bounded prefix
(`M`, `enum_lane_visits_max`) before the device path is accepted.

**Window.** Per spectrum, as the table search: `parent` by the adduct rule (`mass_overflow` outside `u32`),
`half = tol + bound + E_domain` with `E_domain` the domain's largest composition bound, `lo = parent − half` and
`hi = parent + half`, saturating at `0` and `u32::MAX`. For a heavy mass `m`: nothing when `m > hi`; otherwise
`h_lo = max(h_min, ceil(max(lo − m, 0) / m_H))` and `h_hi = min(h_max, floor((hi − m) / m_H))`, the ceiling by
quotient and remainder, both differences taken only after the comparison that makes them non-negative. Every
integer of `[h_lo, h_hi]` gets the verdict of contracts §5 with its own composition's bound
`E = ceil((sum n_e * residual_e) / 1000) + bound`. A heavy mass is accumulated term by term. A product `count * element mass` can exceed `u32`
(`255 * 126,904,472`), so no product is formed first: a count is admitted only when
`count <= (hi − m) / element mass`, after which `m + count * element mass <= hi <= u32::MAX`.

**Work is bounded before dispatch.** `lanes = B * P` is known on the host and must not exceed `enum_lanes_max`
(default 262,144), else the request is refused before any launch. The visit unit is one `(C, N, O)` heavy vector
whose hydrogen range is examined. The limit of this source is **per lane**: `enum_lane_visits_max` (default
4,096; `formula_rows_visited_max` is the table source's limit and is not used here), so the request's logical
work is at most `B * P * enum_lane_visits_max` visits and its physical work twice that (count and fill
re-enumerate). A lane stops **before** the visit that would exceed its budget and raises its exhausted flag.

**A dispatch is bounded too.** A GPU job that runs for many seconds is reset by the driver (*measured* on the
Radeon 860M: `amdgpu: ring gfx_0.0.0 timeout`, after which the device is lost for the process), so no launch may
have an unbounded worst case. Lanes are dispatched in contiguous chunks of
`max(1, enum_dispatch_visits_max / enum_lane_visits_max)` lanes (defaults 4,000,000 and 4,096: 976 lanes per
launch), each launch submitted on its own, so the worst case of one GPU job is about `enum_dispatch_visits_max`
visits. The number of launches is `ceil(B * P / lanes per launch)` for count and for fill, a function of the bucket
and the configuration only. (The reset that prompted this rule was later traced by bisection to the window size,
not to the enumeration: at `M = 512` the same data trains without a reset. The rule is kept as a bound on the
worst case. The cause was the top-F selection, O(F·M²) inside one lane per spectrum; rewritten as F argmax passes,
O(F·M), `M = 2048` trains on the same data without a reset. No kernel may have a per-lane cost quadratic in a
window or peak count without a stated bound.)
Chunking changes no result: the lane function takes the absolute lane index. A scope restriction, not chemistry: a spectrum is searched only when
`half <= 1,511,737` (`floor(3 * m_H / 2)`, one comparison, no multiplication), which bounds the hydrogen range
of a heavy vector to 4 integers; a wider window is `formula_search_exhausted` with nothing joined. `half` is
formed with saturating additions, so an unknown or huge uncertainty lands on that branch. Count and fill run the same lane function with the same budget, so they stop at the
same vector. Any lane exhaustion, or more joined candidates than the scored cap, sets
`formula_search_exhausted` and clears `complete`.

**Kernels** (one shared `#[cube]` lane function, two modes):

1. `ms2_enum_count`, lane per `(b, r)`: `meta [B, 8]`, `rare [P, 8]`, `bounds` (one packed `u32` buffer: domain
   caps, the two bucket tables, ratio fractions, rare ranges, DBE intervals offset by a bias so they are
   unsigned; layout constants in `ops/ms2.rs`), → `lane_stats [B * P, 2]` (joined; visited with the exhausted
   flag in the top bit). 4 arrays.
2. `ms2_enum_offsets`, lane per spectrum: `lane_stats`, `meta` → `offsets [B * P]` and `counters [B, 5]`
   (visited, joined, scored, status bits, complete). 4 arrays. An offset is the exclusive prefix sum **clamped to
   the scored cap** `min(formula_rows_scored_max, M)`, so no saturated value is ever used as a rank. `visited` and
   `joined` saturate at `u32::MAX − 1`; a saturated counter sets `formula_search_exhausted` and is then a lower
   bound, which the contract text says.
3. `ms2_enum_fill`, lane per `(b, r)`: `meta`, `rare`, `bounds`, `offsets`, → `cand`. 5 arrays. A lane whose first
   rank is at or above the scored cap returns at once, and every lane writes only ranks below `rows_scored` and
   stops at the first rank that is not. Slots at or after `scored` are written as padding by a
   separate launch of `ms2_cand_pad`, lane per `(b, m)`: `counters` → `cand` (2 arrays), so every element of
   `cand` has exactly one writer.

Launches of the formula stage with this source: 5 (count, offsets, fill, pad, `ms2_count_features`), then the
head. Memory: `rare` `32 P` bytes, `bounds` under
64 KiB, `lane_stats` and `offsets` `12 B P` bytes, all in the estimate.

Per-stage reject counts are not kept on the device: the staged recall of P4.3 is the host reference's report,
and the device is tested for equality with the host twin on every joined candidate, flag and counter.

### 1.5 Selection (P4.9)

*Measured*, MassSpecGym pilot validation export (714 spectra, 384 molecules held out by structure), 20 ppm,
single host thread, [pilot fit](../bench/results/ms2/formula_sources_msgym_pilot.json),
[scale fit](../bench/results/ms2/formula_sources_msgym_scalefit.json),
[database table](../bench/results/ms2/formula_sources_msgym_db_table.json):

| Source | Fitted on | Gold recall in the scored support | Candidates p50 / p95 / max | Work p50 / p95 | Host latency p50 / p95 | Resident bytes |
|---|---|---:|---|---|---|---:|
| Table, train formulas | 24,171 train structures | 0.237 | 2 / 5 / 24 | 29 / 33 rows | under 0.1 ms | 307,344 |
| Table, train + ChEBI 3-star | + 31,545 ChEBI records | 0.370 | 2 / 6 / 34 | 31 / 36 rows | under 0.1 ms | 530,424 |
| Enumeration, exact filters | 1,921 molecules | 0.987 | 1,548 / 89,087 / 403,707 | 1.33 M / 51 M nodes | 20 / 793 ms | 111 |
| Enumeration + ratio bounds | 1,921 molecules | 0.870 | 368 / 1,237 / 8,371 | 185 k / 1.45 M nodes | 6.3 / 48 ms | 1,172 |
| Enumeration + ratio bounds | 23,235 molecules | 0.985 | 997 / 4,350 / 25,589 | 415 k / 6.2 M nodes | 11 / 158 ms | 1,279 |
| The same in device order, scored prefix `M = 2048`, 65,536 visits per lane | 23,235 molecules | 0.965 | 997 / 4,350 / 25,589 joined; 20.9% of spectra exhausted (133 with the gold in the prefix, 16 without) | 19 k / 381 k visits over `P = 7,993` lanes | 16 / 388 ms | 1,279 + 255,776 (`rare`) |

On train spectra every source has recall 0.997. Reading: on structure-disjoint data a table of known formulas
loses three quarters of the gold formulas whatever its size here, and enumeration recovers them at the price of a
support of about a thousand candidates that the head then has to rank. The bounds fitted on 1,921 molecules cost 12
points of recall; fitted on 23,235 they cost 0.5. Whether the head can rank the gold formula among a thousand is
not known from these numbers and is measured by training (formula recall at F with each source, P4.3).

Defaults: `formula_source = Table` and `M = 32` stay the V1 defaults, so every V0 result is reproducible.
`Enumerate` with `M = 2048` and bounds fitted on the full train set is the configuration for structure-disjoint
evaluation; at that capacity one spectrum in five is `formula_search_exhausted` (more than 2,048 candidates) and
the gold formula is in the scored prefix for 96.5% of spectra against 98.5% without the cap, which is reported
and counted, not hidden. The host numbers are the reference's; the device kernels' latency is measured
separately when they exist.

### 1.6 What the head ranks with: residual and explained-peak features (P4.1, P4.3)

*Measured* (tasks document, "MassSpecGym results"). With `Enumerate`, `M = 2048`, the head of §1.2 puts the true
formula in its top 4 for 48.2% of held-out spectra — and for 44.8% when the peaks are another spectrum's. Its
candidate features are `ln(1 + count)` only: it ranks by a composition prior. A host experiment on the same
candidates shows what is missing: exactly computable explained-peak features add 25 points of recall@4 to a
linear prior and vanish with shuffled peaks; the precursor mass residual alone gives 0.707 on the stored data and
0.116 once a 1 ppm error is added, because the stored precursor m/z of this dataset is the theoretical value
for almost half of the spectra.

**Feature layout.** `ModelConfig::formula_features` (schema 2; absent in a version-1 document) is `Counts`
(default: today's 10 features, every earlier result and checkpoint unchanged) or `Evidence` (`C = 16`):

| Index | Feature | Definition |
|---|---|---|
| 0..10 | `ln(1 + count)` | as §1.2 |
| 10 | `abs_res` | `min(|m_c − m_p|, 4 w) / w`, with `m_c` the candidate's integer mass (`cand[.., 10]`), `m_p` the request's neutral precursor mass and `w` its window half-width (tolerance plus uncertainty, the integers the window was built with); `u32` subtraction and comparison first, one float division last |
| 11 | `signed_res` | the same with sign, in `[−4, 4]` |
| 12 | `expl_count_frac` | explained evidence peaks / evidence peaks of the spectrum (`0` when it has none) |
| 13 | `expl_intensity` | sum of the evidence weights of the explained peaks |
| 14 | `ln(1 + expl_count)` | |
| 15 | `evidence_complete` | `1` when the candidate's sub-composition walk finished inside the work bound, else `0` |

Padding slots (`m >= rows_scored`) are exact `0` in every feature. Features 12 to 15 are constants of the
computation graph: no gradient flows through the evidence.

**Evidence peaks.** `ms2_evidence_peaks`, lane per spectrum: `kept`, the peak m/z and intensity buffers →
`ev_peaks u32 [B, P, 2]` (integer m/z, kept position; `u32::MAX` padding) and `ev_w float [B, P]`, the `P = 32`
most intense kept peaks by `P` argmax passes over `N` (ties by smaller kept position; cost `P · N`, at most
4,096 loads per lane), `ev_w` their intensities divided by the sum over the selected peaks (`0` in padding, all
`0` when the sum is not positive). Peaks whose ion window is out of scope (`half_p > m_H`, §2.1) are never
selected.

**Explained peaks.** A peak is explained by candidate `c` when `ion_assign` (§2.1) with parent `c`, the request's
adduct and tolerances and `work_max = W` has at least one accepted hypothesis for it — the same visit order, the
same closed-form hydrogen interval, the same verdict; an ambiguous verdict explains nothing.
`ms2_formula_evidence`, lane per `(b, m)`: `cand`, `ev_peaks`, `ev_w`, `meta`, `spec` → `cand_ev float [B, M, 4]`
(6 arrays). The lane walks the non-empty heavy sub-vectors of `c` once in the §2.1 order, at most `W = 2,048`
visits (`GenerationConfig::formula_evidence_work_max`), and for each visit tests the `P` evidence peaks with
unconditional loads, keeping the explained peaks in one `u32` bit mask; it stops early when every evidence peak
is explained. Cost per lane at most `W · P = 65,536` peak tests; a padding slot or a spectrum without evidence
peaks costs one comparison. A candidate whose radix product exceeds `W` has `evidence_complete = 0` and features
from the visited prefix. The host twin is `models/ms2/formula_evidence.rs` restricted to the same peaks and
bound, tested for equality with `ion_assign` and with the kernel on poisoned outputs.

`ms2_formula_features`, lane per `(b, m)`: `cand`, `cand_ev`, `meta`, `log_table` → `cand_feat [B, M, 16]`
(5 arrays) replaces `ms2_count_features` for this layout. Launches added to the search stage: 3, constant.
Memory: `B P 12 + B M 16` bytes of evidence buffers plus the wider `cand_feat`, in the estimate.

**Precursor error.** Because the stored precursor of this dataset is often exact, residual features are trained
and judged with an error added: `TrainConfig::precursor_jitter_ppm = σ` multiplies each training precursor m/z
by `1 + e · 10⁻⁶`, `e` normal with standard deviation `σ`, truncated at `3σ`, drawn per (seed, step, spectrum)
on the host before the batch is built; evaluation uses a fixed draw per (seed, spectrum). The driver reports
formula recall at the stored precursor and at `σ = 2` ppm; the `σ = 2` number is the one compared with the V0.7
target. With `σ = 0` nothing changes.

**Cost alternative, not built.** Scoring all `M` candidates with the count and residual features, keeping the
best 64 and computing evidence only for those would cut the evidence work by a factor of 32 at the price of a
second head pass and a selection kernel; it is taken up only if the measured step time requires it (P8).

## 2. Fragment-ion assignment (P4.2)

Revision 2 (answers review findings 3, 6, 7, 8, 11, 12, 13, 17, 19).

### 2.1 Hypotheses

Scope: the two V0 adducts and singly charged ions. The adduct and the charge are fixed by the request; they are
not enumerated. For a formula `f` (composition `c_f`) and a kept peak `p` of the same spectrum, an **ion
hypothesis** is a non-empty heavy sub-vector `u <= c_f` with a hydrogen count `h`,
`0 <= h <= c_f[H] + max(h_a, 0) + 2`. Its atomic mass is compared with `t = mz_p + z_a * m_e` (for `[M+H]+`,
`t = mz_p + 549`; for `[M-H]-`, `t = mz_p − 549`, a request whose `t` leaves `u32` being `mass_overflow`), which
is contracts §4.3 solved for the mass: `mz(g, s) = mass(g) + (h_a + s) m_H − z_a m_e` with `h = H(g) + h_a + s`.
The hypotheses of a peak are a **composition superset**: every supported graph-to-ion mapping `(g, s)` of a
subgraph of a parent with formula `f` is among them, and most hypotheses correspond to no supported mapping
(composition alone fixes neither connectivity nor boundary bonds).

`ms2_ion_assign`, lane per `(b, f, p)`. The heavy sub-vectors are visited in mixed-radix order, element `e` of
`ELEMENTS` (heavy only, C first) being digit `e` with radix `c_f[e] + 1`, carbon least significant, index 1
upward (index 0, the empty vector, is excluded). The visit unit is one heavy vector. The number of non-empty vectors is `total − 1` with
`total` the product of the radices, formed with a guard that stops multiplying once the partial product exceeds
`ion_work_max + 1` (so nothing overflows and `total − 1 > ion_work_max` is decided exactly);
`visits = min(total − 1, ion_work_max)`, and `ion_search_exhausted` is set exactly when `total − 1 > ion_work_max`. For each visited vector with mass `m` (accumulated with the guarded additions of §1.4):
the window is `t ± half_p`, `half_p = tol_p + U + E_ion` by saturating additions, with `tol_p` the fragment
tolerance at `mz_p`, `U` the spectrum's m/z uncertainty and
`E_ion = ceil((sum_e c_f[e] residual_e + 3 residual_H + 421) / 1000)`, an upper bound of every hypothesis's own
bound because residuals are non-negative; the hydrogen range is that of §1.4 intersected with
`[0, c_f[H] + max(h_a, 0) + 2]`, at most 3 integers because a peak is searched only when
`half_p <= 1,007,825` (`m_H`; one comparison), a scope restriction — a wider window makes that peak
`ion_unavailable`; each `(u, h)` gets the verdict of contracts §5 with its own
bound `ceil((sum n_e residual_e + 421) / 1000) + U`. Accepted hypotheses are counted, and the first `J` of them
in visit order are kept; ambiguous ones are counted only.

Request-level bound, checked on the host before dispatch: `B * F * N * ion_work_max <= ion_request_work_max`
(defaults `ion_work_max = 4,096`, `J = 4`, `ion_request_work_max = 2^28`), else `Error::Config`.

| Buffer | Shape | Contents |
|---|---|---|
| `ion` | `u32 [B, F, N, J, 12]` | Per kept hypothesis: 10 element counts (hydrogen is the ion's own count), integer mass, signed residual `mass − t` offset by `2^31`; all `0` in padding |
| `ion_meta` | `u32 [B, F, N, 4]` | Accepted, ambiguous, kept, status: bit 0 `ion_search_exhausted` (visits cut), bit 1 `ion_capacity_exceeded` (accepted `> J`), bit 2 `ion_unavailable` (unknown precision, a window wider than the hydrogen bound, a padding peak or an unstarted formula) |

Bindings: `top_counts`, `kept` (m/z of the kept peaks), `meta`, → `ion`, `ion_meta`: 5 arrays. Memory items:
`ion`, `ion_meta`, the hypothesis features `[B, F, N, J, 10]`, two row-network activations `[B, F, N, J, d]`, the
logits and log-probabilities `[B, F, N, J + 1]`, the label buffers of §2.3, and in training the gradients of the
float items. The support of
`(b, f, p)` is **complete** when no status bit is set.

### 2.2 Distribution

For each `(b, f, p)` with `ion_unavailable` clear, a distribution over the kept hypotheses and an explicit
**unassigned** class: `logit_j = (e(u_j, h_j) · W x_p) / sqrt(d)` with `e` the formula head's row network on the
`ln(1 + count)` features of the hypothesis (through `ms2_count_features`) and `x_p` the encoder output of the peak;
`logit_unassigned = w · x_p + b`; masked `log_softmax` over the classes that exist. A peak with no kept hypothesis
has probability 1 on unassigned. When the support is incomplete the probabilities are conditional on the retained
hypotheses and are reported with that status. The unassigned class is a modelled remainder under partial
pseudo-label supervision; it is not a measured noise probability. Activations: `B * F * N * J * d` floats per
layer of the row network (8 MiB at `B = 8`, `F = 4`, `N = 128`, `J = 4`, `d = 128`, FP32),
an item of the memory estimate; `J <= 8`.

### 2.3 Supervision

Training enumerates under the true parent composition only (`F = 1`, `top_counts = gold_counts`); the metric is
oracle-conditioned and named so. Labels come from the anchors of `q-cut-v1` (contracts §7.2 step 7). For spectrum
`b` and original peak id `i`,

`L_bi = unique { (heavy(g), H(g) + h_a + s) : g a kept target of b, (i, s) in anchors(g) }`.

The host knows where each original peak id sits in the uploaded batch, so labels are keyed by the peak's **raw
index** in the batch, which is what `kept[.., 0]` holds: no `peak_id` buffer is bound. The host deduplicates the
labels of a spectrum, sorts them by (raw index, the 10 counts lexicographically), keeps the first `L = 64` and
counts the rest as `assignment_label_overflow`, then uploads `ion_labels u32 [B, L, 12]` (raw index, the 10 counts
with the ion's hydrogen count, a valid flag). `ms2_ion_label_mask`, lane per `(b, p)`: `ion_labels`, `ion`,
`ion_meta`, `kept` → `label_mask [B, N, J + 1]` float 0/1 and `label_state [B, N]` (`0` no label, `1` every label
kept, `2` labels exist but none kept, `3` some labels kept and at least one not — decided by the label matching
itself, not by the mask sum). 6 arrays. For a peak not in state 1 or 3 the mask row is the one-hot of the
unassigned class, so its masked log-sum is finite; it is then multiplied by the 0 eligibility indicator
(the all-masked rule of V0 §3.8: no logarithm of zero is ever formed).

`L_assign = − (sum over peaks with state 1 or 3 of log sum_j label_mask_j p_j) / max(1, number of such peaks)`. A
peak in state 3 is trained on the kept labels and counted `assignment_label_partial`; a
peak in state 2 contributes nothing and is counted `assignment_label_dropped`; a peak without a label contributes
nothing: a missing annotation is unknown, not noise. Microbatch accumulation sums numerators and counts.
`L = L_graph + 0.2 L_formula + lambda_a L_assign`, `lambda_a = 0.1`, fixed before any run. Every number of this
head is a pseudo-label metric.

### 2.4 Evidence for candidates (consumed by §4)

Generation runs `ms2_ion_assign` for the `F` retained formulas; a trajectory uses the rows of its own formula
slot. A finished candidate graph `g` (heavy counts `u_g`, parent hydrogens `H(g)`, open valence per atom `o_a`)
does not know its boundary-bond count `c(g)`, and the parent is unknown at inference, so whether any boundary
partition exists in it cannot be decided. What can be stated is a range from the graph alone: with bond orders up
to 3, `c_lo = sum_a ceil(o_a / 3)` and `c_hi = sum_a o_a` bound the count of **any** partition of the open
valences into boundary bonds. A kept hypothesis `(u, h)` of peak `p` with `u = u_g` gives the shift
`s = h − H(g) − h_a`; the candidate is **mass-consistent** with `p` when `|s| <= min(c_hi, 2)` and
**mass-consistent for every boundary count** when `|s| <= min(c_lo, 2)`. Neither says that a compatible parent
exists.

`evidence_status`: `0` unassigned (no such peak), `1` mass-consistent, `2` mass-consistent for every boundary
count; with an incomplete assignment support the status additionally carries bit 7
(`evidence_support_incomplete`). `CandidateBatch` (schema 2) gains `evidence_count u8 [B*K]` and, per record,
`evidence_peak_id u32`, `evidence_hypothesis u8` (index among the kept `J`), `evidence_shift i8`,
`evidence_residual i32`, `evidence_log_prob f32`, each `[B*K, E]`, padding `0` beyond `evidence_count`. Each
candidate returns at most `E = 4` evidence records, the ones of largest assignment probability, ties by smaller
peak position: original `peak_id`, shift `s` (`i8`), signed residual in integer mass units (`i32`), assignment
log-probability (`f32`). The statuses and records are pseudo-label evidence: no experimental fragment confidence
is attached to a proposal (P6.8).

## 3. Graph decoding (P5)

V0 already implements the sampling decoder: bounded grammar state, legality masks shared by training, sampling and
validation, factorised heads in the frozen order, creation-state atom memory, counter-keyed draws, absorbing
finished trajectories, a fixed step count and one read. This section lists only what P5 adds.

### 3.1 Capacities are configuration (P5.1, P5.2, P5.8)

`max_atoms` (A, at most 32), `max_ring_closures` (R_max, at most 8), `T = 2 + A + R_max` (at most 64, the contract's
`max_steps` range), `decoder_blocks` (at most 4) and the decoder's `d_inner = n_heads * head_dim`, set apart from
`d_model`, are read from `ModelConfig`; no V0 **capacity** is compiled into a kernel or head (the vocabulary widths
5 kinds, 18 atom-type rows and 4 bond rows are frozen with the chemistry domain and stay constants). Buffer widths follow V0's
formulas (`S = 3A + 16`, record `4T + A + 4`, logits `5 + 18 + 4 + A + 19A + 4A`). Every V0 test that pins a shape
runs at `(A, R_max, T) = (16, 4, 22)` and at `(32, 8, 42)` with 4 decoder blocks: replay masks against
`TraceState`, teacher-forced against stepped logits with every carry compared, the sampler twin, validation.
The data's targets stay within the V0 caps (contracts §7.2); the larger shape is tested on synthetic traces and is
not a claim about labels.

### 3.2 Trajectory allocation and the candidate score (P5.5)

`GenerationConfig::allocation`: `RoundRobin` (V0: trajectory `k` uses formula `k mod top_count`) or
`Proportional`. `ms2_allocate`, lane per spectrum: `top`, `top_counts`, `top_log_prob`, `top_count` →
`traj_formula u32 [B, K, 12]` (per trajectory: formula slot, source id, the 10 counts; slot `u32::MAX` and zero
counts when `top_count == 0`). 5 arrays. Exactly `K` assignments are written per spectrum. Proportional: with `p_f` the retained
probabilities renormalised over the `top_count` retained formulas, every retained formula first receives one
trajectory while trajectories remain (in rank order), the remaining `K'` are shared as `floor(K' p_f)` and the
rest by largest fractional part, ties by smaller slot; trajectories are then assigned to formulas in slot order.
With fewer trajectories than retained formulas the first `K` formulas get one each. The arithmetic is `f32` in a
fixed order; repeatability is guaranteed on one backend, and the test compares the device with the host twin only
where no product `K' p_f` is within `1e-4` of an integer (a floor boundary) or of another product's fractional part
(a tie): `exp` may differ in the last bits between backends. `K <= 64` and `F <= 8` (contract ranges), so the lane
uses fixed local arrays; a non-finite `top_log_prob` among the retained entries makes the spectrum fall back to
round robin. `ms2_init_trajectories` then binds `traj_formula`,
`spectra_meta`, `traj_meta`, `state` and `actions` (5 arrays): it no longer binds `top` or `top_counts`.

The **candidate score** is `formula_log_prob + trace_log_prob` (design §3.6); it orders candidates wherever no
reranker is configured (§4.4). `CandidateBatch` already carries both terms, the formula's composition, rank and
source, and the trajectory index, so provenance is complete. Draws stay keyed by spectrum id, trajectory, step and
field only (V0 §3.6): the allocation changes which formula a trajectory conditions on, not its random stream.

### 3.3 Work per step (P5.7)

No device counter is added. From the one final read the host derives `GenerationWork`: for each step `t`, the
number of **active** invocations — started trajectories with `length > t`, plus those that failed at this step
(`no_valid_action` with `length == t`: the invocation that detected the failure ran and emitted no token) — and
the rest (finished, failed earlier, or never started, `length == 0`), the
submitted trajectory-steps `B * K * (T − 1)` and the fraction of them that were active. Fixed dispatch and zero
per-step reads are V0 properties kept by the footprint tests. Compaction of finished trajectories is the optional
experiment O9 and is not built here.

### 3.4 Failure outcomes on a tiny domain (P5.8)

The exhaustive-enumeration test of V0 is extended to absorbing failures: for a tiny budget every trace prefix is
enumerated with its legal continuations, including prefixes that end in `no_valid_action` (a failure leaf keeps
its tokens so far and adds no token probability). The probabilities of finished and failed outcomes sum to 1
within `1e-5`. Truncation cannot occur at the derived cap `T = 2 + A + R_max`, so it is tested separately with a
deliberately shorter kernel horizon; fixed-seed sample frequencies
of every outcome class lie within 4 standard errors; an unsatisfiable formula (a budget no root atom fits) fails
every trajectory with its status and no token.

### 3.5 In-place recurrent step (P5.9)

With P8.2 (O2): the decoder step writes `h`, `last_u` and `angle` into the trajectory-owned cache after every old
value the step needs has been read, and is compared with the functional step over a full generation, every carry
bit-equal on the CPU runtime and within `1e-6` on the GPU. Until that test exists the estimate budgets two banks.

### 3.6 Beam search (P5.4, P5.6, P5.10)

Optional by the plan and not specified here. It is built only if the sampling baseline's quality at a matched
work budget calls for it (O11); until then `GenerationMode::Beam` stays `Error::Unsupported`.

## 4. Validation, identity, evidence and outputs (P6)

### 4.1 Validation (K10; P6.1)

`ms2_validate` (V0) replays every trace under the shared grammar functions and applies the final-validity rules of
contracts §4.5. P6.1 adds no rule; it adds one negative test per rule, each a crafted trace that violates exactly
that rule and is flagged `invalid_final`: a pointer to a missing atom (declared connectivity), a second bond
between two atoms (bond uniqueness), a bond that exceeds residual valence, an atom type outside the formula budget
(composition), a closure beyond `R_max`, a closure to the parent, a record claimed `finished` whose last token is not STOP (a legal
truncated history stays `truncated`, not invalid), fewer than the minimum atoms. Charge and hydrogen are fixed by the atom type in the V0 domain, so they have no separate rule. Aromatic
and resonance equivalence are not applied (contracts §7.4).

### 4.2 Graph identity (P6.2)

Reference: two graphs are identical exactly when their canonical traces are equal (contracts §7.4; host
`canonical_trace`).

Identity is of the **graph alone**: two trajectories conditioned on different formulas that built the same graph
are duplicates (equal graphs have equal atom multisets, whatever formula was the budget), so no formula field
enters the hash, and formulas are never compared through `formula_row` (which is `u32::MAX` for every enumerated
formula).

`ms2_graph_hash`, lane per trajectory: `actions` → `graph_hash u32 [B*K]`, `graph_scratch u32 [B*K, G]`
(2 arrays written, 1 read). The lane rebuilds the bond list (`A − 1 + R_max` bonds at most, 39 at `A = 32`) and
runs 4 **synchronous** rounds of label refinement over two label banks: an atom starts with
`hash(atom type, degree)`, and a round computes every new label from the old bank only, as
`hash(old label, sum over its bonds of hash(bond order, neighbour's old label))` in wrapping `u32` arithmetic —
the sum is commutative, so no sort is needed and the result does not depend on atom order. The graph hash mixes
the wrapping sum of the final labels with the atom count and the bond count. A scalar `hash_mask` is ANDed onto
every graph hash; production passes `u32::MAX`, tests pass a few bits to force collisions. The scratch row
(`G = 3 (A − 1 + R_max) + 2A` words: bonds as (atom, atom, order), the two label banks; the final labels are in
bank 0) is kept for the next kernel.

`ms2_graph_identity`, lane per trajectory `k`: `actions` (status and length, read only), `graph_hash`,
`graph_scratch`, → `identity u32 [B*K, 2]` (status bits to OR in on the host side of the pack, resolution) and a
lane-owned `identity_scratch u32 [B*K, 3A]` (assignment, used flags, cursor per depth). 5 arrays; validation's
flags are read, never written here. Against every earlier trajectory `j < k` of the same spectrum that is
finished and valid with an equal graph hash, it decides exact labelled-graph equality: unequal atom counts or
bond counts are **different**; otherwise a depth-first search for a **bijection** of `k`'s atoms onto `j`'s
(injective by the used flags, complete at depth `A_k`) such that atom types and refined labels agree and, for
every pair of already assigned atoms, the bond order between them is the same in both graphs, **order 0 (no
bond) included**; iterative with the explicit stack, bounded by `identity_work_max` assignments per pair
(default 4,096; a spectrum does at most `K (K − 1) / 2` comparisons, so the request bound
`B * K (K − 1) / 2 * identity_work_max` is checked against `identity_request_work_max` before dispatch).
Outcomes per pair: equal, different, unresolved (budget).

Candidate status bits (contracts §8 gains them): `7` `duplicate_graph` — an exact comparison proved equality
with an earlier trajectory; `8` `identity_unresolved` — some comparison of this (the later) trajectory ran out of
budget; it is **not** a duplicate flag, and the candidate stays eligible for ranking. `duplicate_trace` (V0) is
unchanged and implies graph equality when the formulas' compositions are equal. `identity_resolution`: `0` trace
only (identity disabled), `1` exact (every comparison of this candidate was decided), `2` unresolved. A hash
mismatch decides "different", which is sound because equal graphs have equal hashes under synchronous refinement;
a hash match decides nothing by itself (refinement cannot tell, for example, a triangular prism from K3,3 with
uniform labels), so no candidate is ever removed without the exact comparison.

Tests: every fixture graph against its own atom permutations and alternative legal traces (equal: flagged, both
records present), same-formula isomer pairs (different), forced collisions through `hash_mask` (distinct graphs
stay distinct), a symmetric graph with `identity_work_max = 1` (unresolved, both kept), all against the canonical-
trace reference on CPU and GPU.

### 4.3 Evidence and the reranker (P6.3)

Per candidate the device forms 8 features: `trace_log_prob`, `formula_log_prob`, atom count over `A`, open valence
sum over `2A`, the evidence count of §2.4 over `E`, the largest evidence log-probability (`0` when none), the
smallest absolute evidence residual in units of the fragment tolerance (`1` when none), and the
`evidence_support_incomplete` flag. The reranker is `Linear(8 → 16)`, SiLU, `Linear(16 → 1)`; its output is a
logit of **containment in the true parent** (contracts §7.3), trained with binary cross-entropy on candidates
generated by a **frozen** generator for the `rank` split. Splits: for CASMI the frozen identity groups of contracts
§1 (validation, ranking, calibration); for MassSpecGym the versioned replacement `msgym-split-v1`, molecule-
disjoint and fixed by a hash of the InChIKey block: the train fold is divided into `fit` (80%) and `rank` (20%),
the validation fold into `calibration` (50%) and `report` (50%), the test fold is not read. The generator and
**every fitted artifact it uses** (formula table, enumeration domain and bounds, any structural pretraining) are
fitted on `fit` only and frozen before a candidate of `rank` is generated; the generation, deduplication and
sampling configuration is stored with the reranker artifact. A training example is a finished, valid,
non-duplicate candidate with a **resolved** containment label; a candidate whose containment check hit its work
limit is excluded and counted, never used as a negative. Calibration (P7.9) is fitted on `calibration` only and
judged on `report`. Feature values are finite by construction: with assignment disabled or no evidence the three
evidence features are `0`, `0` and `1` and the incomplete flag is `0`; the residual unit is the request's fragment
tolerance at the peak, which is at least 1 integer unit for every valid request. A partial substructure is not asked to explain
every peak: no feature measures unexplained intensity.

Without a reranker the score is §3.2's and is labelled `raw` in the output.

### 4.4 Ranking, compaction and the packed output (P6.4, P6.5)

`ms2_rank`, lane per trajectory: `actions` (status), `identity`, `scores` → `rank u32 [B*K]` (4 arrays): its rank
among the candidates of its spectrum that are finished, valid, not `duplicate_trace` and not `duplicate_graph`,
by decreasing score, ties by smaller trajectory; `u32::MAX` for the others, which are never selected.
The score is the f32 sum of the f32-widened ranking terms on every neural dtype, so device and host order agree.
A term or a score is in the **validated domain** when it lies strictly inside (−3e38, 3e38); anything else — NaN,
infinities, finite values beyond that bound, an overflowing sum — makes the candidate ineligible, exactly as in
the top-F selection (§1.2). The rule is a range comparison because a finiteness test is not portable across
shader backends.
Before packing, `ms2_record_pack` gathers a trajectory's integer fields into `record u32 [B*K, W]` and its float
fields into `record_f [B*K, Wf]` (widths and offsets are constants in `ops/ms2.rs`). `ms2_pack`, lane per output
slot `(b, r)`: `rank`, `record`, `record_f` → `packed u32 [B, R, W]`, `packed_f [B, R, Wf]`,
`returned_count u32 [B]` (6 arrays). `R = GenerationConfig::returned`, `1 <= R <= K` (default `min(10, K)`);
`returned_count = min(R, eligible candidates)`; an unfilled slot has trajectory `u32::MAX`, formula row and rank
`u32::MAX` and status `0`; a failed request has `returned_count = 0` and its request status.

`PackedCandidateBatch` (schema 2) is the host form: `B * R` records in rank order with their original trajectory
index, `returned_count [B]`, and the per-spectrum fields of `CandidateBatch`. It is a separate type:
`CandidateBatch` keeps its `B * K` trajectory-ordered contract. `CandidateBatch::pack(R)` on the host has the same
semantics and is the twin.

Readout modes: `generate` reads the uncompacted `B * K` records (V0, one read); `generate_packed` reads only the
packed buffers and the per-spectrum fields (one read); `generate_resident` performs **no** read and returns a
result that **owns** its device buffers — the workspace bucket is leased to it and a later call on the same
workspace allocates another bucket instead of overwriting a leased one — with `read()` (one read) and `release()`.
Each mode is tested for its read count, `generate_packed` for equality with `generate(..).pack(R)`, and the
resident result for surviving a second `generate` on the same workspace unchanged.

### 4.5 API parity and the audit (P6.6, P6.7)

The MS2 classes are registered in the existing extension module (`_mamba3_rl`) and exposed through a packaged
facade `mamba3_ms2` (as `mamba3_graph` is), with type stubs: the three configs (constructible, validated,
round-tripped through JSON), resident inputs (`DeviceSpectra` uploaded once and reused; the resident formula
artifacts), `Ms2Model` (load, `encode`, `generate`, `generate_packed`, `generate_resident`), `CandidateBatch` and
`PackedCandidateBatch` as NumPy arrays with the contract's dtypes, the trainer's `step` / `teacher_eval`,
checkpoints. Errors use the bindings' existing mapping of `Error` variants to Python exceptions
(`bindings/python/src/err.rs`) with the Rust message unchanged; the parity test asserts class and message for one
error of every variant. FP32 only; the backend is the one the wheel was built for. Input arrays may be C- or Fortran-ordered
or non-contiguous; they are copied to contiguous buffers before upload, and the test compares all three layouts.
The GIL is released around device waits only where no Python object is borrowed.

P6.7 is a test and a statement: generation's only device read is the final one (the footprint binaries), no
chemistry toolkit is linked into the crate, and containment, canonical traces and isomorphism references are
called only from label preparation, metrics and tests.
