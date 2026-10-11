# Manual: optimising the speed of molecular graph generation

How to make the MS2 generators in this crate produce molecular graphs faster on the
CubeCL backends (wgpu/WGSL on the Radeon 860M here, cubecl-cuda on T4), without
changing what they produce. It collects what was learned in the speed passes of
2026-10-04 to 2026-10-10 (`docs/MS2_SUBSTRUCTURE_TASKS.md` "Device-bound kernels
on the Radeon", `data/ms2/runs/motif_speed_20261010/`), so that the next pass
starts from the current picture instead of re-finding it.

"Generation" covers three paths that share the same machinery:

| Path | Entry point | Driver |
|---|---|---|
| Motif decoder beam search | `motif::beam_search` + `MotifModel::step` / `gather_host` | `examples/ms2_motif_decoder.rs --arch encoders --queries N --beam W` |
| Completion model sampling / beam | `CompletionModel::generate_with_spectra`, `generate_beam_with_spectra` | `examples/ms2_completion.rs`, Python `MolecularCompletionModel.generate` |
| Substructure decoder (older) | `Ms2Decoder::step_packed`, `generate::generate_decode_step` | `examples/ms2_launch_tally.rs --mode generate` |

Every path is a loop of "one decoder step on the device, one small read to the host,
host-side grammar / beam bookkeeping". The rules below follow from that shape.

---

## 1. The rules in one page

1. **Measure before choosing a lever, and measure again on the current binary.** The
   per-kernel GPU profile and the launch tally are cheap; a wrong lever costs a day.
   A profile from an earlier binary once pointed at a kernel that had already been
   fixed (§2.5).
2. **Generation is GPU-bound, not launch-bound, and then host-read-bound.** On this
   GPU a pipelined launch costs ~10 µs. Cutting launches only pays when the fused
   kernel keeps short lanes; cutting *device reads* pays until there is one per step.
3. **One device read per step is the floor.** Each read waits for the whole queue
   (~1.4 ms fixed on wgpu) plus the GPU work of the step (~0.85 ms for the motif
   decoder at beam 256). When the host spends most of its samples in `to_f32`, the
   path is at the floor and only an on-device step would go further.
4. **Keep the matmul shapes few and friendly.** `n` or `k` not a multiple of 4 locks a
   product into the row-tiled kernel (5 ms instead of <1 ms for the motif head).
   Every distinct row count is a new shape that the tuner probes. Pad the output
   width to a multiple of 64 and the beam rows to a multiple of 32.
5. **Do not read ids back from the device.** Anything the host built (parents, token
   ids) it already has. `MotifModel::gather_host`, `IdTensor::from_host`.
6. **Stop when every trajectory has stopped.** Poll the stop record every few steps
   instead of running `max_steps` (`SAMPLING_EXIT_POLL` = 8).
7. **Plane-per-row kernels beat lane-per-row kernels on RADV.** Unit-per-row mixer
   kernels made generation 1.7x slower until given plane forms with butterfly reduces.
8. **No per-step allocation, update the state in place.** `mixer_step_in_place`,
   `MixerStepBuffers`, `Mamba3Block::step_in_place`.
9. **Compare only paired runs in the same power state.** This package boosts when idle
   and caps near 21 W under load; the same binary swings 2x between the two.
10. **Check results, not just speed.** Candidate sets on the 8 roster queries must be
    identical and log-probs within 1e-4; a beam cut that "obviously" cannot matter
    dropped the only valid completion once.

---

## 2. Measuring

### 2.1 Build what you measure from a snapshot

Benchmark binaries come from a copy of the tree so that later edits do not change the
binary under test, and from a separate target directory so that cargo locks do not
fight another session:

```bash
rsync -a --exclude target --exclude data ./ ../mamba-trainer-verify/
cd ../mamba-trainer-verify
CARGO_TARGET_DIR=target/verify-wgpu cargo build --release --no-default-features \
    --features wgpu --example ms2_motif_decoder
cp target/verify-wgpu/release/examples/ms2_motif_decoder $S/base/motif_T<n>
```

`--features vulkan` (SPIR-V) segfaults in RADV on the MS2 kernels; use `wgpu`.
A HEAD tree built into the *same* target directory as the working tree reuses the
working tree's test binaries, so HEAD "results" would be yours. Separate directories.

### 2.2 Paired, interleaved runs

Run base and candidate alternately, three pairs, and print the CPU clock next to each
so you can see the power state:

```bash
mhz() { awk '/MHz/ {s+=$4; n++} END {printf "%d", s/n}' /proc/cpuinfo; }
for i in 1 2 3; do
  for b in base cand; do
    echo "$b load $(cut -d' ' -f1 /proc/loadavg) MHz $(mhz): $($S/base/motif_$b ... 2>&1 | tail -1)"
  done
done
```

- Check `uptime` first; timed runs while cargo compiles are worthless (8x on
  launch-bound paths at load 30).
- Check GPU users: `cat /sys/class/drm/card1/device/gpu_busy_percent`, and
  `/proc/*/fd` entries holding `renderD128`.
- The first run after any shape change is slow: the tuner probes the new shapes and
  writes `~/.cache/mamba3/matmul-tune-v3-0.1.0.txt`. Compare second runs.
- Report ranges, not single numbers, when another job shares the box.

### 2.3 Per-kernel GPU time

```bash
CUBECL_DEBUG_OPTION=profile CUBECL_DEBUG_LOG=$S/prof.log $S/base/motif_T<n> ... 
```

The log has GPU timestamps per launch and a summary table per sync. Aggregate the
last steady-state steps with awk (kernel name, count, total). Under the profiler a
step is 2–3x slower and per-kernel times swing 2x between runs, so use it for the
*ranking* only. Very short kernels are inflated. One artefact seen: the completion
model's `Ms2ReplayKernel` showed 126 ms per launch, more than the whole unprofiled
step; the drained tally did not show it.

### 2.4 Launches and shapes

- `examples/ms2_motif_decoder.rs --tally`: one training step's launches by region,
  op and launch site (`count  label / op / site`). The API is
  `backend::{start_launch_tally, reset_launch_tally, stop_launch_tally,
  launch_tally_detailed}` with `tally_scope("...")` labels, usable from any driver.
- `MAMBA3_TRACE=1`: one line per shape-sensitive launch, in order.
- `MAMBA3_TUNE_CACHE=0 MAMBA3_TUNE_LOG=1 ... 2>&1 | grep ^tune`: tuned shape, plan
  and GFLOP/s. A product far below ~1000 GFLOP/s boosted is the lead; one stuck on
  `MatmulRowTiledKernel` has an `n` or `k` that is not a multiple of 4.
- `examples/ms2_launch_tally.rs --mode generate --b 8 --k 8 --timed --time 20 --rounds 5`
  for the substructure decoder; `examples/profile_mixer_step.rs` with
  `MAMBA3_STEP_STATE=32` for the decoder-shaped mixer step, `MAMBA3_FUSED_STEP=0`
  for the composed form.
- Launch pins live in `tests/ms2_footprint.rs`, `ms2_generation_footprint.rs`,
  `ms2_launch_budget.rs`; move them when a kernel is added or removed. The `"wgpu"`
  branches in those pins never run here: the backend name is `wgpu<wgsl>`.

### 2.5 Host side

There is no `perf` on this PC. Sample the main thread with gdb from the scratchpad
(stray output files land in the cwd):

```bash
for i in $(seq 100); do gdb -p $PID -batch -ex "thread apply all -q bt 16" 2>/dev/null \
    | grep -A16 'Thread 1 ' >> $S/stacks.txt; sleep 0.05; done
```

Classify the main thread: in `to_f32` / `read_handles` = waiting for the GPU (the
floor); in `DeviceClient::enqueue` = queueing; in `MotifMachine::allowed`, the beam
heap or `Vec` growth = real host work worth optimising. The `DSD-*` device thread in
`WgpuStorage::reserve` / `MemoryPage::coalesce` means per-step allocations.

### 2.6 Traps

- **Stale profile.** A profile of an older binary showed the row-tiled head product
  that the padded head had already removed. Re-profile the binary you will edit.
- **Stale kernels.** CubeCL's on-disk kernel cache is keyed by kernel type and
  comptime args, not source. The repo's `cubecl.toml` has it off; when A/B-ing kernel
  edits run from a cwd with no `cubecl.toml` above it, or delete
  `target/<backend>/0.10.0` first. A value read inside `comptime!` (an env var) is
  not in the cache key either; pass variants as comptime arguments.
- **Knockouts.** Skipping a launch to measure its cost is unreliable here: NaN or
  garbage downstream made steps slower.
- **Drained timers.** `synchronize()` alone returns before the wgpu queue has run;
  drain with a small read too, or sites get charged for their predecessors.
- **Device lost.** "Parent device is lost" after a long single lane (~10M guarded
  loads) is the amdgpu job timeout; `journalctl -k | grep "ring gfx"`. It kills every
  GPU process. Bisect by shape before touching the suspected kernel, and never run
  such a probe while another GPU job runs.

---

## 3. Where the time goes today

Motif decoder, `--arch encoders`, d_model 128, 4 layers, 8 roster queries, beam 256,
wgpu (2026-10-10):

| State | 8 queries |
|---|---|
| Before the passes (per-step expand of keys/values, host mask, row-tiled head, ids read back) | 17.9–18.4 s |
| Now | 5.7–5.9 s |

Per beam step now: ~0.85 ms GPU, ~1.4 ms fixed read wait, and the host has 33 of 53
samples waiting in `to_f32`. The remaining GPU time is `GatherRowsKernel` (cache
tensors per layer), `MixerStepStatePlaneKernel`, and small projection matmuls that
are near the tiled launch floor.

Completion model (`generate_with_spectra`), wgpu: small molecules 0.15 s -> 0.02 s
per request from the early exit; 20-atom roster queries at 256 trajectories ~7%,
since most steps are needed. The Python binding floor is ~10 ms per request on the
fixture model; batching requests buys ~10%.

Substructure decoder: B=8,K=8 37 -> 14.6 ms, B=8,K=32 149 -> 40 ms; 9 small matmuls
are 38% of kernel time; the mixer state update is 8 MB per layer at f32.

### 3.1 Is the GPU well used? (2026-10-11, training step at d_model 320)

Sampled every 0.25 s over a 60-step run (`gpu_util_samples_60steps.txt`):
`gpu_busy_percent` is 100 in every steady-state sample (min 99), the shader clock
sits at 2.6–2.8 GHz against a 3.0 GHz top level, the memory clock at its top
level (937 MHz, LPDDR5X-7500, 120 GB/s theoretical), package power flat at
25.0 W (the cap), and the host thread at 26% of one core. The queue never
starves; there is no scheduling, read-back or allocation slack left in the step.

What is left is inside the kernels, against measured ceilings
(`gpu_bandwidth_ceiling.txt`): a write-only stream reaches 88–92 GB/s, a
two-read-one-write stream over 64 MB 53–61 GB/s (80–86 when the 8 MB operands
sit in cache). The step's memory-bound kernels — `AddFlat` 62 GB/s, `SplitBands`
53, the vectorised strided copies 76 — are at that ceiling; only fewer passes
(fusion) can cut them. The matmuls at 1.2–1.4 TFLOP/s are 45–50% of the 2.75
TFLOP/s single-issue fp32 peak at the running clock, the usual reach of a
register-tiled WGSL kernel without subgroup matrix instructions; that is the one
place with theoretical headroom, and it needs matrix-core or subgroup-matrix
access that wgpu's WGSL does not expose here. One more thing the samples show:
the same kernel runs 30–40% slower once the package is warm and power-limited
(bench run 1 vs run 2 above), which is the whole-step swing seen all day in
pairs — compare only interleaved runs.

### 3.2 ROCm (HIP) on this PC (2026-10-11)

`--features hip` builds and runs here (`ROCM_PATH=HIP_PATH=~/rocm/opt/rocm`,
ROCm 7.1.1, `LD_LIBRARY_PATH=$ROCM_PATH/lib`; gfx1151 is GFX11, wave32). Logs:
`hip_matmul_summary.txt`. The f32 block matmuls run at 1.4–1.7 TFLOP/s, a notch
above wgpu. The matrix cores (16x16x16 f16/bf16 WMMA) are reachable only on HIP,
and two bugs stood between `MAMBA3_MATMUL_PRECISION=bf16` and them:

- `matmul_cmma_kernel` used `sync_plane`, which CubeCL's HIP dialect compiles to
  `#error`; a wavefront is lockstep, so the epilogue's plane-local hand-off needs
  no barrier there (`plane_sync_if`, comptime off on HIP).
- CubeCL's HIP WMMA helpers index the accumulator row with `threadIdx.x / 16`,
  right only for the first wave of a workgroup: every other plane of a 64x64 or
  128x128 block wrote its tile two or three rows off (`examples/probe_cmma.rs`
  shows the plane-0-only pattern). Fixed in `vendor/cubecl-cpp` via
  `[patch.crates-io]`; `tests/mixed_precision.rs` passes on HIP with it.

With both, the tuner picks the 128x128 and 64x64 matrix-core plans at 1.5–2.6
TFLOP/s kernel-only against 1.0–1.5 for f32. Casting both operands in separate
launches on every call (the original staged path) ate that: 3.0 ms f32 vs 6.5 ms
bf16 at 20480x128x648. The mode now casts *late*: under a narrow mode with f32
operands, `launch_plan` casts per plan, and a `CmmaFused` plan reads f32 and
rounds while staging (no cast launches, but no 8-wide vector loads either). The
tuner sees both matrix-core forms and the register kernels with their casts
priced in, and every plan computes on rounded operands, so the mode's numerics
do not depend on which plan won (`tests/mixed_precision.rs` checks exactly
that). On HIP the late cast is the default (`MAMBA3_CMMA_FUSED_CAST=0` restores
the cast-first path). Nine entity-model shapes end to end: f16 29.5 ms, f32 32.9,
cast-first f16 42.0; every shape lands on the matrix cores.

The training step does not follow: HIP f16 0.44 steps/s, HIP f32 0.45, wgpu f32
0.82. `SsdScanBackwardChunked` takes 76 ms per call on HIP against 11 on wgpu
from the same source, and the offline compile says why
(`hip_scan_backward_resource_usage.txt`): 337 VGPRs and 17 SGPRs spilled, 1216
bytes of scratch per lane, a dynamic stack. RADV's compiler fits the kernel;
LLVM's AMDGPU backend does not. Until that kernel is restructured for register
pressure, nothing f16 buys on ROCm here reaches the step. Loss at step 40: f32
4.1953, bf16 4.1997, f16 4.2701 — f16's range, not its precision, is what moves
the gradient operands. Next on ROCm: the scan backward's register pressure, then
4-wide f32 vector loads in the fused matrix-core staging, then a weight-cast
cache for the head's chunk loop.

---

## 4. Levers, in the order they paid

### 4.1 Shapes (2–5x, usually first)

- **Pad the output width.** `MotifModelConfig::n_out_padded()` rounds the head to a
  multiple of 64 (5717 -> 5760). Old checkpoints are padded on load (`pad_head`: zero
  weights, bias −1e4 so the padding never wins). Tests index logits with the padded
  width.
- **Pad live rows.** The beam closure in the example resizes `parents` and `tokens`
  to a multiple of 32 by repeating the last row, so the device sees a handful of
  matmul shapes instead of one per live-row count (40% of host samples were cold
  tuner probes before this).
- **Unsplit keys and values.** `MotifModel::step` runs cross-attention through
  `ms2::attn_weights` / `attn_context` on `[1, m, d]` keys and values with
  `rows_per_spectrum = rows`, no per-step `expand` to `[rows, h, hd, m]` and no
  host-built mask. Search 2.4x on its own.
- **Pad attention memory** so batched products use the vector block kernels (129 ->
  132 masked slots in `packed_hidden`).

### 4.2 Reads and host work (1.5–3x)

- **Validate parents on the host.** `gather_host` takes the parent ids the beam
  already has; the old `gather` read them back from the device every step (25% of
  host samples).
- **Keep host copies of uploaded ids.** `IdTensor::from_host` stores the ids beside
  the buffer; `host()` returns them. Use it only for ids no kernel ever writes, or
  the copy silently goes stale.
- **Early exit.** `generate_with_spectra` polls the action record every
  `SAMPLING_EXIT_POLL` steps and breaks once all trajectories stopped. The beam loop
  already had this.
- **No per-row top-width cut before the heap.** Not exact: dead and finished
  candidates consume pops. See §6.

### 4.3 Kernels (1.2–1.7x)

- **Fused step, in place.** `mixer_step_in_place` / `MixerStepBuffers` /
  `Mamba3Block::step_in_place`: `h` updated in place, `last_u` read from the previous
  step's scratch, no per-step allocation.
- **Plane forms.** `ms2_attn_weights_plane_kernel` (scores + softmax, plane per
  (row, head), unrolled key loads, masked slots skipped); plane-segment coef and
  state kernels in the mixer step (`fused::plane_segments_per_row`, butterfly
  reduce). Long per-lane scans go into groups plus `sum_dim`.
- **Fused logits.** `step_logits_pack` stores pointer scores with the head bias
  folded in; `masked_softmax` replaces the max/sub/exp/sum/div chain (34 launches a
  step in the substructure decoder; the chain cost 5x the products).
- **Unobserved start state.** `Ms2Decoder::start_state_unobserved`: no carry freeze
  unless the trace is captured, caches empty, carries in `FusedStep`.
- **Unconditional loads.** On RADV a load inside an `if` is waited for at the branch
  join. Clamp the index, load unconditionally, mask at first use
  (`buf[select(ok, i, 0)]`). 1.5x on the scan backward; nothing on the LDS-bound
  matmuls, so check the ISA (`RADV_DEBUG=shaders,nocache`) before assuming.

### 4.4 Matmul planner

- `MAMBA3_SPLIT_K`, `MAMBA3_SPLIT_K_SLICE`: split-K for tall products
  (`m*n <= 256*128`, `k >= 1024`); the planner's `[16,128,128]` + `sum_dim` splits are
  deliberate, not waste.
- `Plan::BlockFit`: one right-sized cube per small batched matrix (+10–30% on
  attention shapes).
- Tuner probes are pipelined (`probe_reps`, `MAMBA3_TUNE_REPS`); single-launch probes
  timed the sync and small-shape winners were chance.
- `MAMBA3_TUNE_CACHE=0` disables the disk cache for a diagnosis run;
  `MAMBA3_TUNE_CACHE_DIR` moves it.

---

## 5. Where to go next

Ranked by expected gain per effort; none is a narrow task for a weak coding agent.

1. **On-device beam step.** The only lever left on the motif search. Move the
   `MotifMachine` mask, the per-parent top-k and the heap onto the device so that one
   read per *several* steps suffices. The Lean model in `lean/MotifDecoder/` fixes
   the semantics; the Rust machine is tested against it (96,670 sequences), so a
   device version must pass the same parity test.
2. **Rank-1 `last_u`.** The mixer state update stores `last_u` full (8 MB per layer)
   although it is rank 1; a compact form halves the state traffic per step.
3. **Reduced precision for the head products.** The head is ~39% of a training step
   and the largest product in a beam step, already near this iGPU's fp32 peak.
   bf16 (`MAMBA3_CMMA_BF16`, `MAMBA3_MATMUL_PRECISION`) is the user's decision; pure
   f16 gives NaN in the mixer backward, bf16 is fine.
4. **Cache tensor gathers.** 16 `GatherRowsKernel` launches per step (4 tensors x 4
   layers); gathering all cache tensors of a layer in one launch, or gathering by
   writing the beam's permutation once and indexing through it, removes most.
5. **Batched requests in the binding** buy ~10% only; not worth an API change.

---

## 6. Dead ends (do not repeat)

| Tried | Result | Why |
|---|---|---|
| Per-row top-`width` cut before the beam heap | Reverted | Not exact: dead/finished candidates consume pops; `beam_keeps_a_completion_a_dead_end_outscores` fails. The host was not the bottleneck anyway. |
| Folding norm/bias/residual into a hand-written projection kernel | Removed | Lost at every shape to the tuned block matmul plus separate launches, on wgpu and CPU. Long lanes cost more than the launches saved. |
| Folding the attention o-projection into cached values | Dropped | Values become `heads` x larger; more bytes per step. |
| `CUBECL_WGPU_MAX_TASKS` above 32 | 2x slower | |
| Unit-per-row mixer step kernels | 1.7x slower until rewritten as plane forms | Lane-per-row serialises on memory latency on RADV. |
| Head bias folded into chunked products with a ones column (k = 132) | Reverted | 31 fewer launches, no speedup: row slices became strided copies and 132 is a worse k than 128. |
| Counting launches as the cost model | Misleading | A pipelined launch is ~10 µs here; GPU time per kernel is the real table. |
| Pooling the scan checkpoint allocation | Not a cost | A 52 MB allocation per call is ~0.1 ms; the timing swing it was blamed for is clock state on a short kernel. |
| Fused training attention as lane-per-output kernels | Slower (0.76–0.82 vs 0.95–1.06 steps/s) | ~80 GFLOP/s on 6k rows against ~600 for the tiled batched matmul; kept behind `MAMBA3_FUSED_ATTENTION=1`. |

---

## 7. Workflow for a pass

1. Build the current binary from a snapshot, run the 8-query search and a 200-step
   training run, save outputs as the base (§2.1, §2.2).
2. Profile the base binary (§2.3) and tally one step (§2.4). Write down the top five
   kernels and the read count per step. Decide whether the path is GPU-bound,
   read-bound or host-bound (§2.5).
3. Pick one lever from §4 or §5 that attacks the top item. Write it as one mechanical
   task: file, exact edit, the test that must pass. For opencode use
   `docs/opencode_manual.md`; the free tier does narrow tasks in 3–8 minutes but often
   exits without running the tests, and a later task may rewrite a file from git and
   drop an earlier edit, so re-check earlier edits in any file it touches. Launch it
   in a memory-capped scope (`systemd-run --user --scope -p MemoryMax=12G`) with
   `< /dev/null`.
4. Run the CPU tests yourself:
   `cargo test --features cpu --test ms2_motif --test ms2_kernels --test autograd`.
5. Build the candidate from a snapshot, run three interleaved pairs, and diff the
   candidate sets and log-probs against the base (identical sets, 1e-4).
6. Commit with the measured range in the message; copy task files, logs, profiles and
   outputs to `data/ms2/runs/<name>/` (the scratchpad is tmpfs and gone after a
   reboot).

## 8. Checklist before claiming a speedup

- [ ] Same power state in every pair (CPU MHz printed next to each run).
- [ ] Second run after a shape change, not the first (tuner probes).
- [ ] No cargo build, no other GPU job during the pairs.
- [ ] Candidates identical, log-probs within 1e-4, losses identical to 4 decimals.
- [ ] Launch pins updated if a kernel was added or removed.
- [ ] Profile taken from the binary that was edited, not an older one.

---

## 9. Training step at the SMILES-decoder size (2026-10-11)

`--arch encoders --no-spectrum`, d_model 320, 7 layers, d_state 32, batch 128,
wgpu, `data/ms2/runs/motif_speed_20261011/`. The step is GPU-bound: ~700 GFLOP of
f32 matmul per step (head 21%, the 7 layers' key/value projections over the 129
memory slots 21%, input projections 18%, fingerprint encoder 8%) already run at
1.2–1.4 TFLOP/s, this GPU's tiled-WGSL ceiling, and take half the step. The
other half is memory-bound glue at 50–60 GB/s, so only fewer passes, not faster
kernels, cut it. Reduced-precision products are not available on wgpu (the
`bf16`/`f16` modes need matrix cores). The 128 fingerprint slots are 84% full
(predicted bits at threshold 0.02), so packing the memory is not a lever either.

Done, 0.79–0.81 -> 0.82–0.85 steps/s (quiet GPU, three pairs), loss identical at
step 40:

- **Keys in `[b, h, m, hd]` and `matmul_nt` for the scores** (`MotifModel`,
  `Ms2Decoder::split_heads`). The old `[b, h, hd, m]` key copy, forward and
  back, was the one permute the strided copy cannot vectorise: 5 ms per 21 MB,
  14 launches a step. Strided-copy time 923 -> 272 ms per 12 steps.
- **`select_valid` a vector at a time** (fingerprint encoder, 6 launches a
  step): 3.5 -> 1.4 ms per launch.
- **Conv weight gradient in one pass**: a unit per (group, channel) slides a
  `taps`-wide register window, so the gradient and the input are read once per
  position instead of once per tap. 2.5 -> 1.0 ms per launch, bit-identical.

Tried the same day, kept switchable, off by default: **fused cross-attention for
training** (`Var::ms2_attention`: the step path's `attn_weights` +
`attn_context` forward, `attn_backward` = score gradient plane kernel, `dq`
through `attn_context` over the keys, `dk`/`dv` in one pass over a spectrum's
rows; no head permute anywhere; parity test in `tests/autograd.rs`). It removes
the permutes (476 ms per 12 steps), the softmax pair (197 ms) and the small
batched products, yet the step is *slower*: 0.76–0.82 against 0.95–1.06 steps/s
with `MAMBA3_FUSED_ATTENTION=0` in the same binary. The lane-per-output kernels
run at ~80 GFLOP/s on 6,016 rows (context 6.3 ms, `dk`/`dv` 11 ms, weights 7.2 ms
per launch) where the tiled batched matmul manages ~600; they only win at the
beam's 256 rows. A win needs a shared-memory-tiled form (query block x slot
block, flash-attention style), not another lane-per-output kernel.

**SSD scan backward**, examined the same day with `examples/bench_ssd_scan.rs`
at the training shape (`MAMBA3_SCAN_SHAPE=128,67,10,64,32`; logs
`scan_backward_knockouts.txt`): the chunked kernel takes 11 ms per call against
a ~3 ms bandwidth floor. Knocking out any one of its five phases saves 0.5–1.7
ms while knocking out all five leaves 3.4 ms, so no phase dominates: the cube is
a latency-bound chain of small shared-memory tiles and seven barriers per chunk,
and too few cubes are resident to overlap it. Inflating the shared memory by
10–20 KiB changes nothing at chunk 16 (residency is capped by registers) and
costs 35% at chunk 8 (there it is the 28 KiB of shared memory). The existing
`MAMBA3_SCAN_CHUNK=8` is 9–12% faster on the kernel at both shapes measured
(10.9 -> 9.9 ms, 9.5 -> 8.8 ms), which is under 1% of a step and doubles the
live checkpoints (7 layers x 94 MB), so the default stays 16. The recurrent
backward is slower (14.8 ms). A real gain would need fewer, larger phases per
chunk (fewer barriers) or a lower register footprint, which is a redesign of a
kernel that already went through several rounds. The forward with checkpoints
swings 2.4–7.5 ms between back-to-back rounds while its GPU time is 3.1 ms. That
is **not** the 52 MB checkpoint allocation: allocating it per call costs ~0.1 ms
over reusing one buffer (0.73 vs 0.61 ms with a fill), the scratch arena changes
nothing, and the forward *without* checkpoints swings the same way (1.8–4.0 ms).
It is the GPU's clock state on a kernel too short for 20-call rounds to settle
(`bench_alloc_checkpoint.out`). Trust the profiler's GPU time for short kernels.

Not done, in order of expected gain: the tiled attention (~4%); band-wise
input-projection products instead of `SplitBands`/`CatBands` (~2%, moves launch
pins in every mixer user); the scan backward redesign (≤4%). Dead end: widening the split-K
bound to `1024 x 1024` so the `[320, 2100]` input-projection weight gradient is
sliced — the tuner clocks the slices faster (1,175 vs 820 GFLOP/s) but the step
is not.
