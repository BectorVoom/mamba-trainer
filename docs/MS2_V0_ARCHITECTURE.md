# MS2 V0 architecture: the trainable vertical slice

Status: implementation specification for P2, P3 and V0 of [MS2_SUBSTRUCTURE_TASKS.md](MS2_SUBSTRUCTURE_TASKS.md),
revision 2 after the codex review in [reviews/MS2_V0_ARCHITECTURE_CODEX_REVIEW.md](reviews/MS2_V0_ARCHITECTURE_CODEX_REVIEW.md).
It fixes what [the design](MS2_SUBSTRUCTURE_DESIGN.md) leaves open for V0 and uses the names, units and
statuses of [the contracts](MS2_CONTRACTS.md). Nothing here is measured; every size is a shape, not a benchmark.

V0 is a composed model: existing autograd operations wherever one exists, and a new CubeCL kernel only where
no primitive does the job (integer arithmetic, sorting and selection, graph legality, sampling). Fusion is P8.

## 1. Modules

| Path | Contents |
|---|---|
| `src/models/ms2/{chem,graph,grammar,targets,formula,contract}.rs` | Host reference (P1). No tensors |
| `src/tensor/ops/ms2.rs` | The kernels of §3 and the adapters of §3.8, each with a host twin |
| `src/models/ms2/batch.rs` | Device batches: uploads, shape buckets, the resident formula table |
| `src/models/ms2/encoder.rs` | Peak features, metadata context, bidirectional encoder, spectrum memory |
| `src/models/ms2/decoder.rs` | Formula head, graph-action decoder, factor heads |
| `src/models/ms2/loss.rs` | Teacher-forced graph loss and formula loss |
| `src/models/ms2/generate.rs` | Sampling loop, validation, packed readout |
| `src/models/ms2/workspace.rs` | Capabilities, memory estimate, preallocated generation state |
| `tests/ms2_kernels.rs`, `ms2_encoder.rs`, `ms2_training.rs`, `ms2_generation.rs`, `ms2_footprint.rs` | CPU and GPU tests |
| `examples/profile_ms2_substructure.rs`, `examples/ms2_train.rs`, `examples/ms2_label_report.rs` | Profile, training and label-report drivers |

Integer device buffers are `IdTensor` (`u32`). Every launch goes through `launch_1d` or `launch_1d_spans`, so
the existing launch counter sees it.

**Binding budget.** wgpu's default limit is 8 storage buffers per shader stage, and CubeCL uses one for scalar
and shape metadata and reports `hardware.max_bindings = limit − 1`. Every MS2 kernel therefore binds **at most
6 arrays**. Related fields are packed into one buffer and addressed by offsets passed as scalars. The
capability probe (§6.1) refuses a device with `max_bindings < 7` before any launch.

## 2. Shapes

`B` spectra, `Nr` raw peak capacity (512), `N` kept peaks (128), `d` width (128), `G` target slots per spectrum
(16), `T` trace length (22), `A` atoms (16), `M` formula window capacity (32), `F` formulas (4), `K`
trajectories (8), `V` atom types (17, ids 1–17), `R` table rows.

Per-spectrum integer metadata is one buffer `meta [B, 8] u32`: `peak_count`, `precursor`,
`precursor_uncertainty`, `adduct`, `fragment_tolerance`, `precursor_tolerance`, `id_lo`, `id_hi`.

## 3. Kernels

Each kernel has one lane per output item unless stated, writes every element of its outputs (padding gets an
explicit value), reads no padding slot of its inputs, and has a host twin with the same arithmetic. A selection
is made by comparison, never by multiplying with a mask, so a NaN or a huge value in an unselected slot cannot
reach an output.

### 3.1 Peak selection (K01/K02)

The host validates the batch first (`SpectrumBatch::validate`): a spectrum with a fatal request status is
uploaded with `peak_count = 0`, so the device never sees a non-finite or negative intensity of a valid peak.
`peak_id` is strictly increasing within a spectrum (a malformed batch otherwise), so ties broken by array index
are ties broken by `peak_id`.

The **transformed intensity** `t` of a raw peak is its intensity for scale 0 and its square for scale 1; a
square that overflows to infinity makes the peak ineligible. A raw peak is **eligible** when its index is below
`peak_count`, `0 < mz <= precursor + 2_000_000` (evaluated as `mz − precursor <= 2_000_000` when `mz >
precursor`, so no sum overflows) and `0 < t < 3e38`. Finiteness is that range test on purpose:
Metal compiles WGSL with fast-math, which folds `t - t == 0` and `t != t` to constants (an overflowed square
slipped through that way on the M1).

1. `ms2_peak_stats`, lane per spectrum. `max = ` the largest eligible `t` (`0` when none). Writes
   `stats [B, 3]`: `max`, `total = sum of t / max` over eligible peaks with `t >= 1e-3 * max`, summed in index
   order (`0` when `max == 0`).
2. `ms2_peak_rank`, lane per raw peak. `keep = eligible && max > 0 && t >= 1e-3 * max`. Rank among the kept
   peaks of the spectrum by decreasing `t`, ties by smaller index: `rank [B, Nr] u32`, `u32::MAX` when not kept.
3. `ms2_peak_order`, lane per raw peak. `selected = rank < N`. Position among the selected by increasing `mz`,
   ties by smaller index: `position [B, Nr] u32`, `u32::MAX` when not selected.
4. `ms2_peak_gather`, lane per kept slot `(b, p)`: finds the raw index `i` with `position == p` and counts the
   selected peaks `len`. Writes `kept [B, N, 3] u32`: `raw index` (`u32::MAX` in padding), `mz` (`0`),
   `reverse` (`len − 1 − p` for `p < len`, else `p`); and `kept_f [B, N, 2]`: relative intensity `t / max` (`0`)
   and `valid` (`1` / `0`).
5. `ms2_peak_summary`, lane per spectrum: `summary [B, 2] u32` (`len`, and the request bit `peaks_truncated`
   when more than `N` peaks were kept) and the third column of `stats`, `retained` (`sum of selected t / max`
   over `total`, `0` when `total == 0`). A separate kernel keeps every kernel within the binding budget and
   gives each output one writer.

A spectrum with `len == 0` is `empty_spectrum`: the host sets that bit from `summary` at the readout, and its
trajectories never start (§3.6). Work per lane is `Nr`; the five launches do `O(B * Nr^2)` comparisons.

### 3.2 Peak features (K01): `ms2_peak_features`

Lane per `(b, p)`. Output `[B, N, 71]`, exact zeros in padding. With integers `m = mz` and `c = precursor`:

- Scalars (7): `f32(m) * 1e-9`; `(f32(c) − f32(m)) * 1e-9`; `f32(m) / f32(c)`; relative intensity `r`; `sqrt(r)`;
  `ln(1 + 100 r) / ln(101)`; `ln(1 + gap)` with `gap` the distance in Da to the previous kept peak (`0` for the
  first).
- Fourier (64): for `v` in `{m, |c − m|}` (the difference taken by ordered integer subtraction) and the 16
  frozen integer wavelengths `w_k`, `sin` and `cos` of `2π * f32(v mod w_k) / f32(w_k)`. `w_k` is the table
  `MS2_WAVELENGTHS` in `ops/ms2.rs`: `round(10^(4 + 5.3 k / 15))`, from 10⁴ to 2×10⁹ integer units (0.01 to 2000
  Da). The remainder is exact; the phase is its `f32` approximation (24 significant bits), which is all a
  neural feature needs. No feature feeds an accept/reject decision.

### 3.3 Formula window (K03, V0 form): `ms2_formula_window`

Lane per spectrum. Inputs: the resident table `table [R, 2] u32` (mass and per-row arithmetic bound), `meta`,
and scalars (`max_error`, the limits). Reproduces `FormulaTable::window` of the host reference step for step,
including the two halving loops and the visit counter. Writes `window [B, M, 2] u32` (row, `u32::MAX` padding;
flag `0` none, `1` accept, `2` ambiguous) and `counters [B, 4] u32` (visited, joined, scored, request status
bits `formula_absent`, `formula_search_exhausted`, `exact_mass_unavailable`). More joined rows than
`min(M, rows_scored_max)` sets `formula_search_exhausted`; the formula probabilities of such a spectrum are
over a partial support and are reported with that status (contracts §9).

### 3.4 Top-F: `ms2_formula_top`

Lane per spectrum. From `log_prob [B, M]` and `window`, `F` successive argmax
passes (`O(F * M)` per lane): a slot is a candidate when its flag is
non-zero, its score lies in the validated domain `-3e38 < s < 3e38` (never
selected otherwise), and it was not chosen by an earlier pick; ties break by
smaller slot via a strict `>` scan in increasing slot order:
`top [B, F, 2] u32` (row, window slot), `top_log_prob [B, F]`, `count [B] u32`. Trajectory `k` of spectrum `b`
uses formula `k mod count[b]`. With `count == 0` the request has failed and its trajectories never start.
With `oracle_formula` the caller supplies `top` with `count = 1` and `top_log_prob = 0`.

### 3.5 Graph legality and replay (K08, training): `ms2_replay`

Lane per target `(b, g)`. Inputs: `tokens [B*G, T, 4] u32`, `target_meta [B*G, 12] u32` (length, a budget flag,
the 10 budget counts). Replays the trace with the grammar of contracts §4.4 and writes one packed output
`replay [B*G, T, 4 + A] u32`: for step `t`, the four legality masks **before** token `t` conditioned on the
earlier fields of token `t`, exactly as `TraceState::masks`, then each atom's residual valence before token
`t`; and `replay_atoms [B*G, A + 1] u32`: the step at which each atom was added (`u32::MAX` for unused slots)
and the first illegal step (`u32::MAX` for a legal trace). Steps at or after the trace length get all-zero
masks.

The grammar state (atom types, residual valences, parents, counters, used composition; no adjacency is
needed, because the grammar cannot emit a duplicate bond: an ADD creates a fresh atom, and a closure from the
newest atom must point above its parent and above its previous closure)
lives in a lane-owned row of a scratch buffer `state [.., S]` with `S = 3A + 16` words, initialised by the
kernel before it is read. The legality rules are `#[cube]` functions over that row, shared with §3.6 and §3.7:
one implementation serves training, inference and validation. The root ADD_ATOM is legal only if some atom
type fits the budget.

### 3.6 Legal sampling step (K08, inference): `ms2_sample_step`

Lane per trajectory `(b, k)`. Bindings (6): `logits [B*K, L]` float, the head outputs of this step packed as
`kind (5) | atom_type (18) | bond_base (4) | pointer_base (A) | pointer_by_type (19 * A) | pointer_by_bond
(4 * A)`; `tables` float, the parameter `bond_by_type [19, 4]`; `traj_meta [B*K, 14] u32` (id halves,
trajectory, a start flag, the 10 budget counts); `state [B*K, S] u32` (in/out); `actions [B*K, T * 4 + A + 4]
u32` (in/out: the trace, `open_valence`, `length`, `status`, and two spare words); `scores [B*K, 1]` float
(in/out: `trace_log_prob`). In one launch it:

1. leaves a trajectory that is finished, failed or not started untouched (absorbing);
2. computes the legal kinds; none legal sets `no_valid_action` and stops the trajectory;
3. samples the kind from the masked softmax of `kind / temperature`; then, as the kind uses them, the atom
   type, the bond order from `bond_base + bond_by_type[c]` and the pointer from
   `pointer_base + pointer_by_type[c] + pointer_by_bond[bond]`, each over its own legal set, which is
   non-empty by construction of the kind mask. `c` is the sampled atom type for ADD_ATOM and `18` for
   CLOSE_RING. The root ADD_ATOM uses kind and type only, CLOSE_RING kind, bond and pointer, STOP kind only;
4. adds the log-probabilities of the fields used, normalised over their legal sets at temperature 1, to
   `trace_log_prob`;
5. applies the token to the state, writes it at position `length`, bumps `length`, and on STOP sets
   `finished` and writes `open_valence`. At step `T − 1` without STOP it sets `truncated`.

Step 0 is not sampled: the loop initialises every started trajectory with START. Draws are inverse-CDF in
`f32` with `u = hash_unit_f32(4 * step + field, base, 0)`, where `base = hash_u32(trajectory, key, 0)`,
`key = hash_u32(id_lo, s, id_hi)` and `s = hash_u32(seed_hi, seed_lo, 0)`: a function of the spectrum's stable
id, the trajectory, the step and the field only. Each input passes through a full mixing round before it meets
the next one. The first V0-C form, `hash_u32(trajectory, id_lo, id_hi) ^ seed_lo`, XORed the trajectory into
the raw id, so with the exported sequential ids (high half 0) trajectory `k` of spectrum `i` and trajectory
`l` of spectrum `j` drew identical streams whenever `k ^ i == l ^ j`: every eight consecutive spectra shared
eight streams. The seed's high half also entered only after the last multiplication, so seeds differing only
there gave XOR-related draws. The first legal index, in
increasing order, whose cumulative probability exceeds `u` is taken, and the last legal index if rounding
leaves none.

### 3.7 Validation and duplicates (K10, V0 form): `ms2_validate`

Lane per trajectory. Replays `actions` from scratch with the functions of §3.5 into its own scratch row and
sets `invalid_final` if any step is illegal or the final-validity rules of contracts §4.5 fail. Then compares
the trace and formula row with every earlier trajectory of the same spectrum and sets `duplicate_trace` on a
match. `K * T` comparisons per lane.

### 3.8 Adapters

Small kernels that make the integer outputs usable by the float operations, each differentiable where a
gradient flows:

- `ms2_select_valid(x [.., n, d], valid [.., n])`: `x` where `valid != 0`, exact zero elsewhere; the adjoint
  is the same selection of the gradient. Used on every encoder tensor instead of a multiplication.
- `ms2_bits_to_mask(bits [rows] u32, width)`: a `[rows, width]` float 0/1 mask from a bit set, for
  `mask_logits`. A formula-window flag of `1` or `2` both become `1`.
- `ms2_lookup(table [V, d], ids [rows] u32)`: embedding lookup that returns a zero row for `u32::MAX` and
  whose adjoint is a device-side gather (one lane per table element looping over `rows`), so neither pass
  reads ids back to the host. `autograd::embedding` is not used: its adjoint sorts ids on the host.
- `ms2_safe_ids(ids, fallback)`: replaces `u32::MAX` by `fallback`, for `take_along_last` on unused fields.

A row whose legal set is empty (an unused field, a padding step) is given the mask `1` at index 0 only, so its
`log_softmax` is `0` at that index and the gathered value is exactly `0`; it is additionally multiplied by the
field's 0/1 use indicator. No distribution is ever read from an all-masked row.

### 3.9 Position-batched teacher and fused sampler step (P8, O4)

Both hot loops were launch-bound, not work-bound, so each was rewritten to do the same arithmetic in fewer
launches. Neither changes a value beyond the summation order inside a dot product.

**Teacher pass.** The heads are position-wise, so `Ms2Decoder::teacher` scores every output position in one
pass over `rows * (T − 1)` flattened rows instead of looping over the positions:

- `ms2_teacher_plan`, lane per `(row, position)`: for output position `i` (predicting token `i + 1`) one row of
  `[rows * (T − 1), 9 + A]` with the five in-range conditioning ids of the per-position rule (kind, atom type,
  bond, pointer, conditioning row), the four legality bit fields `replay[r, i + 1, 0..4]` and the residuals
  `replay[r, i + 1, 4..]`.
- `ms2_effective_mask`, lane per mask element: `bit * use + idle * (1 − use)` for one field over every
  position, the arithmetic of the composed `bits_to_mask`, `mul`, `rsub_scalar`, `add` chain.

The atom memory does not depend on the position: it is projected once and broadcast over the positions by the
key sum. `nll` is `0 − sum` over the `(position, field)` terms, so an empty target is exactly `+0.0`.

**Sampler step.** `Ms2Decoder::step_packed` is `step_logits` with one kernel per stage, driven by a state from
`start_state_fused` (the parameter tables are concatenated once per call, so each stage binds one table):

- `ms2_step_embed`: the six input embeddings and their sum, in the composed order.
- `ms2_attn_scores` (lane per `(row, head, slot)`) and `ms2_attn_softmax` (lane per `(row, head)`), then
  `ms2_attn_context` (lane per output element): single-query cross-attention over the cached keys and values
  with the head split folded into the indexing, so no per-step permute of the memory.
- `ms2_atom_key`: the atom memory is kept **projected**. The projection of the previous output is a segment of
  the previous step's head row, so on ADD_ATOM that segment is copied into `atom_keys[row, count − 1]` and the
  pointer head never re-projects the whole memory; the same launch refreshes the clamped residual ids.
- One product for every head and projection (`[d, 27 + 2 d]`: kind, atom type, bond, pointer query, memory
  projection), then `ms2_step_logits`: the 27 head logits copied and the three pointer blocks computed
  straight into the packed sampler row.
- `ms2_freeze_rows`: the carry freeze in place, guarded per lane, so a live row costs one flag load and no
  traffic on the carries; `h` and `last_u` share one launch.

Dot products in these kernels load eight pairs per round: on the Radeon a lane waits on memory once per load
round, so the unrolled form is what makes one lane per output element cheap. The composed step
(`step_logits` plus the pack copies) stays as the reference: `GenerationWorkspace::composed_step` drives the
loop with it, and `tests/ms2_fused_step.rs` compares the two forms of the same call (identical actions,
lengths and statuses; log-probabilities and every post-freeze carry within `1e-4`) on CPU and wgpu. A decoder
whose heads carry a LoRA adapter, a quantizer or a bias on the two pointer projections falls back to the
composed step.

## 4. Model

### 4.1 Encoder

All SSM blocks use the contracted `SsmConfig` (SISO, rotational, learned trapezoid, no convolution, 4 heads of
64 channels, 4 groups, state 32) and are internally unidirectional `Mamba3Block`s.

- Peak embedding: `Linear(71 → d)`, SiLU, `Linear(d → d)`, then `select_valid`.
- Metadata context `g [B, d]`: `ms2_lookup` embeddings of adduct (3 rows), polarity (2), energy count (9) and
  the energy known flag (2), plus `Linear(34 → d)` of `[min(ce, 400) / 100 or 0 when unknown, precursor * 1e-9,
  32 Fourier features of the precursor]`.
- Conditioning: `x = select_valid(x + W g)`.
- Two bidirectional blocks. Block `l`: `f = Fwd_l(x)`, `r = gather(Bwd_l(gather(x, reverse)), reverse)`,
  `x = select_valid(f + r − x)`, with `gather` the existing `gather_tokens` over the per-spectrum `reverse`
  ids of §3.1 (a permutation that is its own inverse). The input of every block is exact zeros in padding, so
  with finite weights no padding position holds a non-finite value, and the masked products inside the chunked
  scan multiply finite numbers. Padding sits after the valid positions in both directions and the scan is
  causal, so padding cannot change a valid output.
- Output: `select_valid(RmsNorm(x))`. Spectrum memory `mem [B, 1 + N, d]` is `[Linear(g); x]` with mask
  `[1; valid]`. Pooled vector `pool [B, d]`: the sum of `x` over peaks divided by `max(len, 1)`, plus `g`.
- `control = MetadataOnly` sets `valid` to zero for the memory and the pool (the encoder still runs);
  `ShuffledSpectrum` rotates the peak buffers by one spectrum within the batch before §3.1.

### 4.2 Formula head

Resident table features `ln(1 + counts) [R, 10]`. Window rows are looked up with `ms2_lookup` (padding gives
zero rows): `e = Linear(SiLU(Linear(10 → d))) [B, M, d]`, `score = (e · Linear(pool)) / sqrt(d)`, `mask_logits`
with the window mask, `log_softmax` over `M`. A spectrum with an empty window uses the all-masked rule of §3.8
and contributes nothing. Training loss: cross-entropy to the window slot of the gold formula when it is among
the scored rows; otherwise no formula loss and the spectrum is counted `formula_absent`.
`L_formula = sum over spectra with a gold slot / max(1, their count)`. The formula embedding of a row is `e`.

### 4.3 Decoder

Sequences are `[B*G, T, d]` in training and `[B*K, 1, d]` per step in generation. For cross-attention only,
they are viewed as `[B, G*T, d]` or `[B, K, d]` (each query attends independently, so this changes no value)
and restored before the next scan; the memory is never replicated per target or trajectory.

- Input at position `i` (token `i`, predicting token `i + 1`), all static lookups, so the parallel pass has
  no dependence on its own output: `E_kind[kind] + E_type[type] + E_bond[bond] + E_pointer[pointer] +
  E_step[i] + e_formula`, with the unused fields of a token looking up index 0.
- Two layers, each `x = Mamba3Block(x)` then `x = x + CrossAttention(RmsNorm(x), mem, mask)` with 4 heads;
  keys and values of each layer are computed once per spectrum. Attention scores are masked with
  `mask_logits` before the softmax.
- **Atom memory** `[.., A, d]` is used by the pointer head only. The atom added by token `s` has memory
  `h[s − 1]`, the decoder output that predicted it. In training it is gathered after the parallel pass with
  `gather_tokens(h, atom_step − 1)`, where unused slots keep `u32::MAX` (the gather's zero row), and the head
  at position `i` may point only at atoms with `atom_step <= i`, which the legality mask of row `i + 1` of
  `replay` enforces: the mask at output `i` is the one before token `i + 1`. In generation the memory row is
  written when the atom is added.
- Heads on the decoder output `h`, each `mask_logits` then `log_softmax`:
  - kind: `Linear(d → 5)`;
  - atom type: `Linear(d → 18)`;
  - bond: `Linear(d → 4)` plus the learned row `bond_by_type[c]`, added before masking;
  - pointer: keys `k_j = Linear(atom_memory_j) + E_residual[residual_j]`, query parts `Linear(h)`,
    `E_ptr_type[c]` and `E_ptr_bond[b]`; the score of atom `j` is the sum of the three dot products with `k_j`
    over `sqrt(d)`, added before masking.
- In generation a finished trajectory's recurrent carries are not advanced: its new cache is selected back to
  the old one (`h`, `last_u` and `angle`) by the finished flag, with `ms2_select_valid` on the cache tensors.

### 4.4 Loss

For target `g` of spectrum `b`, `nll(g) = − sum_t [log p(kind_t) + log p(type_t) + log p(bond_t) +
log p(pointer_t)]` over the fields each token uses, START excluded, STOP included, padding excluded.
`L_graph = (1 / B) * sum_b sum_g q_b(g) * nll(g)`, the design's mean over the spectra of the batch with an
unlabeled spectrum contributing zero; accumulation over microbatches divides by the total `B`.
`L = L_graph + 0.2 * L_formula`. All of it stays on the device; one batched scalar read per reporting boundary.

## 5. Generation loop, reads and candidate construction

Upload once → §3.1 → §3.2 → encoder → formula window → formula head → top-F → initialise trajectories →
`T − 1` sampling steps of `[embed, 2 × (block step, attention), heads, pack logits, ms2_sample_step,
atom-memory write, carry freeze]` with a fixed step count → `ms2_validate` → **one** `read_all` of the packed
buffers. The step runs in its fused form (§3.9): the same stages, one kernel each. Launches per call are recorded as `L_preprocess + L_encoder + L_search + (T − 1) * L_step +
L_finalize`.

`CandidateBatch` is built on the host from that one read: `actions`, `length`, `open_valence`, `status` from
`actions`; `trace_log_prob` from `scores`; `formula_row` and `formula_log_prob` from `top` by `k mod count`;
`request_status` as the union of the host validation bits, `summary` and `counters`; a failed request keeps
its K records with `request_failed`, `length 0`, `formula_row = u32::MAX` and zero scores;
`evidence_status`, `identity_resolution` and `attachment_partition` are the V0 constants.

**Reads.** `read_count` skips the matmul tuner's one-time reads by design, so it cannot prove a zero-read
region alone. P2.7 adds `runtime_read_count`, incremented on every device read whatever its purpose. The
footprint test warms each shape bucket (first call: tuning and compilation), then asserts that both counters
advance by exactly one per warmed generation call and that the training step's counters advance only at the
reporting boundary. Cold setup, warmed production, backward and profiling are reported separately.

**Allocation.** The grammar state, action records, scores and atom memory are allocated once per `(B, K)`
bucket in `Ms2Workspace` and reused. The mixer step and the composed attention allocate their outputs
functionally in V0: that is a known departure from the design's "no request-sized allocation in the warmed
decoder loop", which stays an open acceptance item for P8.2 (O2) and is not claimed by V0. V0 measures it:
allocation calls per step and reserved bytes over 200 repeated requests and an alternating-bucket sequence.

## 6. Infrastructure (P2)

### 6.1 Capabilities (P2.4)

`Ms2Capabilities::probe(device)` reports and `check(config)` enforces before any launch: element type support
(existing `supports_dtype`; V0 is `F32`), `max_bindings >= 7`, whether `client.profile` returns device
timestamps or only system time, and whether the runtime reports reserved bytes. A failed check is
`Error::Unsupported` naming the capability. CPU and wgpu/Metal are the mandatory targets.

### 6.2 Memory estimate (P2.2)

`Ms2MemoryEstimate::{generation, training}` returns named byte counts, computed with checked arithmetic (an
overflow is an error): weights, formula table (`R * (8 + 40)`), encoder activations (`2 B N d` plus the
projection widths of the mixer), spectrum memory and per-layer keys/values (`B (1 + N) d (1 + 2 Ld)`), decoder
carries (`B K Ld (2 h p s + h s / 2)` per bank, **two banks** while the step is functional), graph state,
actions, atom memory (`B K A d`), head scratch, readout, and for training the gradients, optimizer moments and
teacher-forced activations (`B G T` positions). A unit test reproduces the design's figure: `B = 8, K = 32,
Ld = 4, h = 8, p = 32, s = 64` gives 135,266,304 bytes (129 MiB) per bank and 258 MiB for two.
`GenerationConfig::max_device_bytes` is compared with the estimate before allocation. P2.2's reconciliation
test compares the estimate with the reserved-bytes change of a real call and records the ratio.

### 6.3 Counters (P2.7)

Added to `backend.rs` next to the existing ones: `runtime_read_count`, `upload_bytes`, `download_bytes` and
`allocation_calls`, incremented in the tensor constructors and read paths. Logical live and peak bytes are not
observable without wrapping every handle; they are reported as `unavailable`, and reserved bytes come from the
existing `reserved_bytes`.

### 6.4 Single-`u32` arithmetic (P2.5)

Bounds for the V0 domain, each a test: precursor at most 2,000,000,000; parent mass after the adduct shift
within `[48,992,724, 2,001,007,276]`; table masses at most 3,399,572,650; window bounds computed with
saturating arithmetic; ion m/z of a 16-atom subgraph plus shift below the precursor bound plus 2 Da;
tolerance by the split of contracts §5 with its stated intermediate bounds; differences taken by ordered
subtraction. Nothing in V0 needs a second limb. The integer kernels are tested against the host reference on
boundary values, including `u32::MAX` sentinels.

### 6.5 Profile driver (P2.6)

`examples/profile_ms2_substructure.rs`, modelled on `examples/profile_entity_model.rs`: stage spans
(preprocess, encoder, formula, decoder init, decode steps, validate, readout; forward, backward, optimizer
for training). Stages are isolated with the generation/training stage hooks (`generate_with_hook`,
`step_with_boundaries`): one instrumented call runs exactly what production runs, and the hook snapshots
counters and synchronised wall time at each real boundary, after draining previously queued work. Each stage
records synchronised host wall-clock time under the timer `SynchronizedHostWallClock` with the runtime's
profiling capability (probed from what `client.profile` returns: `DeviceTimestamps` or `SystemTime`) in a
separate field. In host mode (default) `profile_ms` is `unavailable`; in device mode (`--profile-mode
device`) the session runs on the device runner thread through `backend::profile_session` and each stage
carries a real `client.profile` span with the `ProfileDuration`'s own timing method — subject to the
single-pass limitation below. Device-mode stages call
the same workspace-level production stage functions production calls (warmed buckets, the same `no_grad`
guard) — no replica workload. On a `DeviceTimestamps` runtime whole-stage device duration is
UNAVAILABLE whenever the stage spans more than one timestamped compute pass, and is reported as the
JSON string `"unavailable"` rather than as a number: verified against the pinned `cubecl-wgpu-0.10.0`
source, an ordinary `client.profile` span returns the first timestamped compute pass's begin-to-end
duration, not the stage's elapsed device time (`compute/stream.rs:239` flushes queued work and opens the
token; `compute/stream.rs:548` attaches timestamp writes only when a new pass opens;
`compute/timings.rs:321` drains newly initialised tokens so later passes get no timestamp writes;
`compute/stream.rs:457` ends the pass once `tasks_count >= tasks_max`; `compute/timings.rs:193`
resolves the token's end against its initial query set). One pass holds at most `device_pass_task_limit`
tasks (default 32, `cubecl-wgpu-0.10.0/src/runtime.rs:188`, overridable with `CUBECL_WGPU_MAX_TASKS` at
`runtime.rs:192`; recorded in every record and device stage), and any mid-stage upload or read forces a
flush too (`compute/stream.rs:105` write path; `read_resources` ends the pass). A device-timestamp stage
whose host-measured launches exceed the limit, or that uploads or reads mid-stage, therefore reports
`"profile_ms": "unavailable"` with a `profile_scope` saying exactly why; a stage that provably fits one
pass (launches within the limit, no mid-stage upload/read) keeps its number with the scope `"single
timestamped compute pass"`. On the CPU runtime (`SystemTime`) nothing changes. `device_span_plausible`
remains only as an extra self-check alongside the scope. A `--profile-mode device-sum` that would sum
per-launch-group spans is deliberately NOT built: the production stage functions expose no
launch-group decomposition (each `generate_*_ws` stage is one closure), so summing separately profiled
groups would change the measurement without becoming whole-stage elapsed time. With
`--formula-source enumerate` every estimate and refusal in the driver — the base record, the T+1
slope, the stability preflight and the device session preflight — uses the enumeration-inclusive
estimate (`Ms2MemoryEstimate::generation_with_enum` / `training_with_enum`): the domain and bounds are
fitted on the host compositions first and sized exactly as `DeviceEnumArtifacts::upload` sizes them,
before any allocation, so a limit between the table-only and the enumeration-inclusive estimate
REFUSES instead of panicking in the cold call. Reserved bytes are recorded as
the production endpoint (`reserved_bytes_after`) plus the high-water mark sampled at every hook boundary
(`peak_reserved_bytes_sampled`). Recorded per call:
warm-up count, cold first-call time, the counters of §6.3 and launches per stage, the launch budget
`L_preprocess + L_encoder + L_search + L_init + (T-1)*L_step + L_finalize` reconciled against the
warmed-call total exactly (the driver exits non-zero on any mismatch), and the `--stability`
repeated/alternating-bucket serving measurement, written as JSON. P3.6 runs it
over `N` in {64, 128, 256, 512} and `B` in {1, 8, 32}, recording configurations refused by the memory limit
(the stability measurement preflights the same estimate and refuses the same way).

The P2 acceptance item "workspace allocation happens outside the decoder hot loop" is NOT met in V0: the
warmed decode loop allocates device buffers every step (measured per-step slope, `decoder_loop_allocation_free:
false`), and closing it is P8.2/O2 work (§5).

## 7. Experiments (V0.5 to V0.7), declared before any result

- **Overfit fixture**: 128 labeled train-subset spectra of 128 molecules (`export_casmi.py --name overfit`).
  Success means the teacher-forced token NLL on those spectra falls below 10% of its initial value and the
  sampled candidates (K = 8) reach target coverage above the shuffled-spectrum control's on the same spectra.
- **Pilot**: 2,000 train-subset molecules and 400 validation-subset molecules, two spectra each at most.
  Three models with the same budget of optimizer steps and the same seed: real spectra, `ShuffledSpectrum`,
  `MetadataOnly`; plus the structure prior. Primary comparison: validation teacher-forced NLL per token;
  secondary: target coverage and containment precision at K = 8 (contracts §10), both with bootstrap intervals
  over molecules. The result is reported whichever way it falls; if real spectra do not beat both controls,
  the next step is the representation, the labels and the objective, not more kernels.
- **Profiles** (V0.7): forward, backward and generation at the V0 shapes on CPU and M1, with the counters of
  §6.3. Latency and memory targets for later optimisation are set from those baselines, after they exist.

## 8. Parity and test matrix

| Check | Reference | Backends |
|---|---|---|
| Each kernel of §3 against its host twin on fixtures and randomised inputs, with NaN and `u32` sentinel poison in every output before launch and a launch-error check after | Host twin | CPU, wgpu |
| Peak selection against `targets::filter_peaks` plus a host top-N; empty, all-zero, duplicate-mass, over-capacity and poisoned-padding spectra | P1 | CPU, wgpu |
| `ms2_formula_window` against `FormulaTable::window`, counters included | P1 | CPU, wgpu |
| `ms2_replay` masks against `TraceState::masks` for every fixture trace, the invalid traces and the root-budget cases | P1 | CPU, wgpu |
| Padding independence: poisoned padding, batch permutation, a spectrum alone versus in a batch, short and long spectra, missing metadata | Self-consistency | CPU, wgpu |
| `Mamba3Block` `apply` against `step` for the contracted config, every carry compared, both directions | `Mamba3Block::step` | CPU, wgpu |
| Gradients of every new operation and composed path by finite differences on tiny shapes | Finite differences | CPU, wgpu |
| Teacher-forced logits equal stepped logits for the same prefix, atom memory and attention included | Self-consistency | CPU, wgpu |
| Sampler: exact RNG words and masks against the host twin; log-probabilities within tolerance for a supplied prefix; exhaustive trace enumeration on a tiny domain with normalised probabilities; fixed-seed frequencies; same-backend repeatability and batch independence. Sampled tokens are compared across backends only where the draw is farther from every CDF boundary than the numerical error | Host twin and enumeration | CPU, wgpu |
| One read per warmed generation call by both counters; launches per step constant; reserved bytes over 200 calls and alternating buckets | Counters | CPU, wgpu |
| Estimate against reserved bytes; binding-limit and memory-limit preflight errors | Capabilities | CPU, wgpu |
