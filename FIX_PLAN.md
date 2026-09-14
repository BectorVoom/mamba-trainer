# Defect-fix record

The record of every defect found by driving this crate's reinforcement-learning
stack from a real project (a Kaggriculture agent) and of how each was fixed. Every
"why" is a measurement, not a guess. The **status table is the source of truth**;
each section below it is an *as implemented* record — API, semantics, tests,
measured numbers — not an instruction. `PLAN.md` is the separate training-speed
plan and stays authoritative for kernel work.

Plans that fed this record, in order: `MAMBA_TRAINER_FIX_PLAN.md` (F1, F2,
A1–A7), `MAMBA_TRAINER_KAIZEN_PLAN.md` (R1, K1–K8), and
`MAMBA_TRAINER_OPEN_ISSUES_PLAN.md` (E1, W1a–e, F2b, A2b, A6, Q1–Q3, D1), and
`MAMBA_TRAINER_EMA_PLAN.md` (T1 of `MAMBA_TRAINER_STABILITY_KAIZEN_PLAN.md`), all
in `Kaggriculture/experiments/exp011_fused_ppo/`.

---

## Status

| | task | severity | state | record |
|---|---|---|---|---|
| F1 | PPO reference anchor | high | done | [F1, F2](#f1-f2) |
| F2 | precision capability check | high | done (F2b replaced the name match) | [F1, F2](#f1-f2) |
| A1 | legal-action masking | high | done (K4–K6, WGSL fix) | [A1](#a1-legal-action-masking) |
| A2 | optimizer and trainer state in checkpoints | high | done (A2a) | [A2, A5](#a2-a5-checkpoints) |
| A2b | exact RL continuation, environment included | high | done | [A2b](#a2b-exact-continuation) |
| A3 | Python LR schedules | medium | done | [A3](#a3-learning-rate-schedules) |
| A4 | completed-episode return accounting | medium | done | [A4](#a4-episode-returns) |
| A5 | versioned binary checkpoints | medium | done | [A2, A5](#a2-a5-checkpoints) |
| A6 | compiled device games from Python, routed fused | low | done | [A6](#a6-device-games-from-python) |
| A7 | checked `MAMBA3_MATMUL_PRECISION` | low | done | [A7](#a7-matmul-precision-from-the-environment) |
| R1 | the reference scores its own history | high | done (K7) | [kaizen](#kaizen-round-r1-k1k8) |
| K1–K8 | atomic restore, config, counters, masks, docs | — | done | [kaizen](#kaizen-round-r1-k1k8) |
| V1 | docs, examples, wgpu wheel run | required | done (K8) | [kaizen](#kaizen-round-r1-k1k8) |
| E1 | kernel compilation failures are loud | P0 | done | [E1](#e1-launch-failures-are-errors) |
| W1a | no non-finite literals in distribution kernels | P1 | done | [W1](#w1-wgsl-gaps) |
| W1b | launch-shape draw test on WGSL | P1 | done (passed unmodified after W1a) | [W1](#w1-wgsl-gaps) |
| W1c | mixed-precision tests honest on WGSL | P2 | done | [W1](#w1-wgsl-gaps) |
| W1d | host-twin bit-exactness gated on the libm probe | P2 | done; found and fixed N3 | [W1](#w1-wgsl-gaps) |
| W1e | the reductions claim corrected | P2 | done | [W1](#w1-wgsl-gaps) |
| F2b | capability query instead of device-name match | P2 | done | [F2b](#f2b-capability-query) |
| Q1 | clippy warnings | P3 | done: 0 with `-D warnings` | [Q1–Q3](#q1q3-tooling) |
| Q2 | one reproducible check entry point | P2 | done: `tools/check.sh` | [Q1–Q3](#q1q3-tooling) |
| Q3 | portable CPU wheel by default | P3 | done: `tools/build_wheel.sh` | [Q1–Q3](#q1q3-tooling) |
| D1 | this document consolidated | P3 | done | — |
| N1 | training not bit-reproducible run to run | medium | fixed: `Grads` iterates in id order | [findings](#findings-along-the-way) |
| N2 | CubeCL CPU reads use the caller's stream | medium | worked around | [findings](#findings-along-the-way) |
| N3 | `log1p` broken by Metal fast math | medium | fixed | [findings](#findings-along-the-way) |
| N4 | fused rollout adopts a stale observation after restore | medium | fixed | [findings](#findings-along-the-way) |
| N5 | matmul tuner timed failed candidates | low | fixed with E1 | [E1](#e1-launch-failures-are-errors) |
| N6 | matmul tuner replaced a plan already in use; kernel not pinnable from Python | medium | fixed | [findings](#findings-along-the-way) |
| T1 | a moving average of the weights (EMA), Rust and Python, in checkpoints | P1 | done | [T1](#t1-a-moving-average-of-the-weights) |
| N7 | reading an empty tensor back panicked | low | fixed with T1 | [findings](#findings-along-the-way) |

---

## Before you change anything

```bash
cd /Users/ods/Documents/mamba-trainer
tools/check.sh            # fmt, clippy -D warnings, CPU tests, CPU wheel + pytest
tools/check.sh --wgpu     # ...plus the wgpu Rust suite and wgpu wheel + pytest
```

`tools/check.sh` checks every step's exit code (never a pipeline's), keeps the full
logs, and prints a per-group table and every skip reason. Run the wgpu half for any
kernel change: CPU and WGSL accept different kernel source.

### Repo map

| path | what lives there |
|---|---|
| `src/backend.rs:272` | `check_launches` (E1); `supports_dtype` at `:215` (F2b); `read_handle` at `:287` (N2) |
| `src/rl/env.rs:62` | `VecEnv`; `save_state`/`load_state` at `:121`/`:130` |
| `src/rl/buffer.rs:98` | `TrajectoryBuffer` |
| `src/rl/collect.rs:516` | `Collector::episode_return`; `export_state`/`stage_state` at `:537`/`:573` |
| `src/rl/snapshot.rs:414` | `RolloutSnapshot`, `StateWriter`/`StateReader` |
| `src/rl/fused.rs` | the one-kernel rollout step |
| `src/rl/ppo.rs` | `PpoConfig`, `PpoBatch`, `ppo_objective`, `PpoTask`, `ReferencePolicy` |
| `src/rl/imitation.rs` | `ImitationBatch`, `behaviour_cloning_loss` |
| `src/tensor/ops/rl.rs:363` | `sample_categorical` |
| `src/train/optim.rs:65` | `Optimizer`; `AdamW` at `:254` |
| `src/train/checkpoint.rs:107` | `Checkpoint`; `save`/`load` at `:200`/`:219`; `stage_training` at `:478` |
| `src/train/ema.rs` | `EmaConfig`, `Ema` (T1); `Trainer::with_ema` in `src/train/trainer.rs`; kernel `ema_step` in `src/tensor/ops/fused.rs` |
| `src/distributions/univariate.rs:108` | `NonFinite` (W1a) |
| `bindings/python/src/lib.rs:169` | `#[pymodule] fn _mamba3_rl` |
| `bindings/python/src/learner.rs` | `PyPpoLearner`, `PyImitationLearner`, `evaluate` |
| `bindings/python/src/resume.rs` | learner checkpoints: levels, staging, config comparison |
| `bindings/python/src/game.rs` | compiled-in device games (A6) |
| `bindings/python/python/mamba3_rl/_mamba3_rl.pyi` | type stubs — update with every API change |

### Invariants every change keeps

1. **A replayed window reproduces the actor's log-probabilities.** Every first-epoch
   PPO ratio is 1, masks included (`tests/rl_learn.rs`, `tests/rl_masking.rs`).
2. **Fused and unfused rollouts produce byte-identical windows**, masks included
   (`tests/rl_fused.rs`; from Python, `tests/test_game.py`).
3. **Collection performs zero per-step host reads**; reserved bytes and per-step
   launches stay flat (`tests/rl_*footprint.rs`).
4. **Defaults do not change behaviour.** New knobs are off by default.
5. **`tests/rl_reference.rs::oracle` stays green on CPU and wgpu.**
6. **No silent fallback.** An unsupported request errors with an actionable message.

`PpoBatch` is built with a struct literal in `tests/rl_learn.rs:1090`; a field
added to it breaks that target only. After one: `cargo test --no-run`.

---

## E1 — launch failures are errors

**Why.** A kernel that failed to compile "ran" and left its output untouched —
zeros for a fresh buffer. Both the `-inf` masking bug (fixed in `01a5741`) and W1a
were invisible for that reason alone.

**Investigation (CubeCL 0.10).** `launch_unchecked` returns `()`. On wgpu,
`backend/base.rs` `create_module` returns `CompilationError` after a
`log::error!` (this crate installs no logger); `compute/server.rs` `launch` then
pushes `ServerError::Launch` onto the launching thread's stream and does not
dispatch. `read`/`read_one` flush with errors ignored, so a read never sees it;
`ComputeClient::flush()` and `sync()` flush with errors requested and return
`ServerError::ServerUnhealthy { errors }`, clearing them. The CPU runtime panics
at launch (`prepare_task(..).unwrap()`). So CubeCL does expose the failure, and
the **propagate** option was chosen; no fail-fast hook or CubeCL patch was needed.

**As implemented.**
- `backend::check_launches(&device) -> Result<()>`: flushes this thread's stream
  and maps parked errors to `Error::Backend`, de-duplicated, naming the kernel and
  quoting the WGSL validator. A flush is not a read: footprint read counts are
  unchanged.
- Checked reads: `Tensor::try_to_data`/`try_to_f32`, `IdTensor::try_to_vec`,
  `Var::try_to_f32`. The infallible reads panic with the same message.
  `Device::synchronize` panics on a failed launch (it used to discard the error,
  consuming it); `Device::try_synchronize` returns it.
- Checked boundaries: the collector's window end; `Trainer::step` before the
  optimizer update and at its reads; every `Result`-returning host read in
  `train/`, `rl/`, `nn/quant`, `infer`, `tensor/ops`.
- N5: the matmul tuner's syncs were `let _ = block_on(sync())`; a candidate that
  failed to compile would time as the fastest and win. They are checked now.
- Python: `Error::Backend` → `RuntimeError`; array conversions and
  `synchronize()` use the checked reads.

Errors are per stream, and a stream is per thread: a failure is reported by a check
on the thread that launched the kernel.

**Tests.** `tests/kernel_errors.rs`: an infinity-literal kernel is an
`Error::Backend` naming `infinity_literal_kernel` on wgpu<wgsl> (and runs, with
`inf` in its output, on CPU); the infallible read panics; sync and the explicit
check each report it once. **Mutation check** (local, not committed): reverting
`01a5741`'s `F::min_value()` in `mask_logits` made 16 `rl_masking` tests fail on
wgpu with `WGSL compilation failed for kernel mask_logits_flat_kernel_f_f32_n_4 …
unknown identifier inf` instead of assertions on zeros.

**Cost.** Measured by `examples/measure_reliability.rs` (medians of 1,000):
`check_launches` on an idle queue costs 8.0 µs on the CPU runtime and 19.8 µs on
wgpu<wgsl> (Apple M1); a one-element read including it 13.6 µs and 1.30 ms. It is
paid once per collected window, once per trainer step before the update, and once
per host read — never per collection step.

## W1 — WGSL gaps

**W1e, correcting the earlier record.** It said `reduce::max_dim`/`min_dim` "seed
with infinities too". They do not: `reduce_op!` seeds each accumulator from the
input buffer (`src/tensor/ops/reduce.rs`, the comment above `reduce_op!`); the
infinity identity is host-only, for an empty axis. `tests/tensor.rs` and the
`max_dim`/`softmax`/`log_softmax` cases in `tests/autograd.rs` pass on wgpu.

**W1a.** `f32::NEG_INFINITY`/`INFINITY`/`NAN` in `#[cube]` code of
`src/distributions/univariate.rs` compiled to `f32(inf)`/`f32(NaN)`, invalid WGSL;
the `NaN` default outputs sat in every kind's `cdf_of`/`icdf_of`/`entropy_of`/
`moment_of`/`entropy_grad_of`, so even Normal returned zeros. Now `NonFinite {
inf, nan }` is passed in, and the pointwise, parameter and parameter-gradient
kernels take `inf`/`nan` as scalar arguments, which no compiler folds into a
literal; the host twin passes `NonFinite::HOST`. Semantics unchanged (`-inf`
log-density outside the support, `+inf` divergent moments, `NaN` where undefined).
`icdf_of` wraps `closed_form_icdf_of`, which the samplers call without `NonFinite`.
`tests/kernel_literals.rs` scans every `#[cube]` body under `src/` for
`INFINITY`/`NEG_INFINITY`/`NAN` and `infinity()`/`neg_infinity()`/`nan()`
(mutation check: `out = f32::NAN` back in `cdf_of` →
`univariate.rs:661: NAN in #[cube] fn cdf_of`). wgpu: `tests/distributions` 21/21
(13 failed before).

**W1b.** `draws_do_not_depend_on_the_launch_shape` passed unmodified after W1a: its
failure ("only 4096 draws changed") was Normal draws being zeros, not an RNG defect.

**W1c.** `tests/mixed_precision.rs` gates every narrow case on
`backend::supports_dtype` and prints the skip reason; modes are set through
`try_set_matmul_precision`. The device-thread panic (`bf16 is not a valid
WgpuElement`) is unreachable from public APIs: `Tensor::from_data` returns
`Error::Unsupported` for an unsupported dtype, `Tensor::empty` (every tensor's
origin) panics on the calling thread with the same message, and `matmul_t` returns
an error when an unchecked `set_matmul_precision` stored an unsupported mode.
`the_capability_query_matches_what_the_kernels_do` checks all of it. wgpu: 6/6,
f16 runs, bf16 skips with its reason.

**W1d.** `libm_primitives_agree_with_the_host` reports its table instead of failing
(still asserting that the CPU runtime agrees exactly, and that every backend is
within a sane bound: 16 ulp, `tan` 1024). The two bit-exact twin tests skip with the
probe's summary where it disagrees. `special_device_matches_host_within_budget` and
`every_distribution_matches_its_host_twin_within_budget` run everywhere with
per-function and per-operation ulp budgets (2× the worst measured on wgpu<wgsl>,
Apple M1, to the next power of two; rejection samplers may disagree on 5% of draws).
Measured Metal libm: `exp` 4, `ln` 24, `sqrt` 2, `sin` 424, `tan` 426, `powf` 5 ulp
(`floor` exact). That tolerance test found N3.

## F2b — capability query

`supports_matmul_precision` is `backend::supports_dtype`: CubeCL's per-type table
(`properties().type_usage(StorageType::Scalar(Float(kind)))`) must grant `Buffer`,
`Arithmetic` and `Conversion`. Verified on the CPU runtime (f16 and bf16) and
wgpu<wgsl> on Apple M1 (f16 via `SHADER_F16`, not bf16), matching
`cubecl-wgpu` `backend/wgsl.rs`; no override table was needed. Python messages
unchanged (`test_module.py` WGSL refusal test still passes).

## A2b — exact continuation

**Why.** A learner checkpoint restored training exactly but not the run: a restored
learner's next window started from a fresh environment, zeroed recurrent state and a
restarted draw schedule, and `load_checkpoint` reported `"exact": true` for it.

**As implemented (Rust).**
- `VecEnv::save_state() -> Result<Option<Vec<u8>>>` (default `None`) and
  `load_state(&[u8])` (default `Error::Unsupported`), all or nothing by contract.
  `RecallEnv`, `GameWorld` and `ParallelEnvs` implement them with tagged, versioned
  layouts (`StateWriter`/`StateReader`); a pool restores every worker or puts the
  earlier ones back.
- `Collector::export_state` / `stage_state` → `StagedCollector::apply`: observation,
  last termination flags, running and completed episode returns, draw seed and
  counter, temperature, mask-column width, engine step counter, every layer's
  `h`, `last_u`, `angle`, `conv`.
- `RolloutSnapshot { collector, reference_cache, env }`: `capture`, `attach` into a
  `Checkpoint` (tensors in `Checkpoint::rollout`, environment bytes in
  `Checkpoint::blobs`, every integer exactly in `metadata.rollout`),
  `from_checkpoint`, `stage` (validates against a live collector and reference,
  uploads, changes nothing), `StagedRollout::apply` (`load_state` first, then an
  infallible swap).
- `RolloutSnapshot::reference_weights`: a captured reference carries its weights,
  written to `Checkpoint::reference`; `stage` refuses a live reference whose
  weights fingerprint differently (`StateDict::fingerprint`, FNV-1a, moved from
  the bindings). `ReferencePolicy::{weights, from_weights, fingerprint}` rebuild
  and compare references. A checkpoint from before the field stages unchecked.
- `MultiSyncCollector::{capture_rollout, restore_rollout, environments}`: the
  bundled pool snapshots and restores like `Collector` + `ParallelEnvs`.
- Binary checkpoint format v3 adds rollout slots, reference-weight slots and byte
  blobs; a checkpoint without them is still written as v2. JSON refuses rollout
  state and blobs, and carries reference weights.
- `Checkpoint::stage_training` validates weights and restores an optimizer without
  writing the model, so a learner can stage everything before changing anything.

**As implemented (Python).**
- Boundary rule: a full save between `collect()` and `update()` raises; the prepared
  batch is not serialized.
- `learner.save(path, level="optimizer" | "full")` (default `"optimizer"`: today's
  behaviour). `"full"` needs `.m3ck` and an environment with `save_state()` and
  `load_state()` (the optional protocol in `protocol.py`; bytes are opaque, never
  unpickled), else `NotImplementedError`.
- `load_checkpoint(path, strict, config, level=None)`: stages trainer, weights,
  rollout and reference, then calls the environment's `load_state`, then swaps.
  Seed and temperature follow `config` like the rest of the configuration.
- `metadata.contents` is what the file holds; `continuation = {"level": "full" |
  "optimizer" | "warm", "notes"}` is the learner's history (only ever decreasing).
  Old `{"exact": true}` reads as `"optimizer"`, `false` as `"warm"`. The load report
  carries this load's `level`.
- `from_checkpoint` takes `temperature`/`seed` from a full checkpoint by default.
- A PPO learner with a reference saves its weights (`Checkpoint::reference`) and
  architecture (`metadata.reference_policy`) at both levels; the recorded
  fingerprint must match them or the load is refused as damaged.
  `from_checkpoint` rebuilds the reference when none is passed;
  `config="checkpoint"` adopts it (over a different reference, or none);
  `config="live"` keeps a different live one as before.

**Tests.** `tests/rl_resume.rs` (CPU and wgpu): PPO with a reference, masks and an
LR schedule, and DAgger with a decaying schedule, over an environment with
asynchronous resets drawn from its own generator — N rounds, a file, fresh objects
with other weights, seeds and environment state, M rounds, against N + M. Every
observation acted on, next observation, sampled action, reward, mask,
completed-episode return total and count, running return, mean episode return,
reference score, learning rate and counter is identical, and so are the losses
and final weights, to the bit (after N1); the restored reference is rebuilt from
the file's weights. Also: the same for the fused game path and a worker pool;
PPO and DAgger through a `MultiSyncCollector` saved and restored with
`capture_rollout`/`restore_rollout` (other-weights reference, missing reference
and wrong pool width refused without changing the pool); reference weights round
trip in both formats and refuse another architecture; skipping the rollout state
diverges; refused staging and a refusing environment change nothing; `RecallEnv` and
`GameWorld` refuse foreign, truncated and mis-seeded bytes; format v3/v2 and JSON
refusal. Python `test_continuation.py`: the restore in a **subprocess**, PPO
(through `load_checkpoint` and through `from_checkpoint` with no reference) and
DAgger, actions, observations and episode-return totals compared exactly;
`test_resume.py`: a reference rebuilt, adopted over another or where there was
none, a passed reference still verified, damaged reference weights refused; without the rollout the run diverges; missing
protocol, pending window, JSON path and unknown level are refused; a refusing
environment and a corrupted built-in environment state leave the learner unchanged;
old checkpoints map to `"optimizer"`.

**Size and time.** At the exp010 shape (208 lanes × 120 steps, `d_model=256`, 4 layers, 988,054
parameters, a reference, `RecallEnv`), medians of 3:

| | CPU runtime | wgpu<wgsl>, Apple M1 |
|---|---|---|
| optimizer-level `.m3ck` | 3,956,061 bytes (3.77 MiB) | same |
| full `.m3ck` | 66,402,664 bytes (63.33 MiB): the rollout adds 59.55 MiB, almost all of it the actor's and the reference's recurrent state (29.76 MiB each) | same |
| `RolloutSnapshot::capture` | 6.5 ms | 62.1 ms |
| attach + save | 68.7 ms | 140.6 ms |
| load + parse | 17.2 ms | 16.8 ms |
| stage + restore | 13.8 ms | 18.5 ms |

Measured before reference weights were saved. With a reference of the policy's
architecture, both files at this shape now also carry its 988,054 parameters:
3,952,216 more payload bytes (3.77 MiB) plus their header descriptors, at either
level (computed, not re-measured).

## A6 — device games from Python

`mamba3_rl.game("recall", num_envs, symbols=4, horizon=8, seed=0, masked=False)`
builds a compiled-in `GameWorld` from a one-entry registry
(`bindings/python/src/game.rs`). `PpoLearner(policy, game)` collects through
`Collector::collect_fused` (`learner.collection_path == "fused"`; `fused=False`
drives it from the host; `fused=True` over anything else raises). Unknown names, bad
parameters (including a horizon other than the compiled 8) and `ImitationLearner`
over a game (no expert) raise `ValueError`. Games support masks and A2b.
`learner.window()` reads a window back; `launch_count()` is exposed. The bindings
README documents adding a game in Rust. Tests (`tests/test_game.py`): fused and host
windows byte-identical, masked and unmasked; fewer launches fused; refusals; a full
checkpoint continues a game run.

## T1 — a moving average of the weights

**Why.** exp011 Phase 2b evaluated arm A every 10 rounds on the same 16 seeds and
got margins from −208 to +6,618; its best checkpoints reproduced on 64 unseen games
(+5,829 … +8,450) while round-200 checkpoints ranged +1,062 … +7,034. A stronger
anchor and an lr warm-up did not narrow that. Good weights keep appearing and are
left behind; the trainer had no weight averaging. Plan:
`MAMBA_TRAINER_EMA_PLAN.md`.

**As implemented (Rust).**
- `ema_step(ema, param, one_minus_decay) -> Result<Tensor>`
  (`src/tensor/ops/fused.rs`): one launch of `ema + c·(param − ema)`, in that form
  and order, into a fresh buffer; none for an empty tensor; mismatched shapes
  refused.
- `mamba3::train::{Ema, EmaConfig, EmaWarmup, StagedEma}` (`src/train/ema.rs`, also
  in the prelude). `EmaConfig { decay, warmup: None | Tf }`, `validate` (finite, in
  `[0, 1]`), `decay_at(t)` (`min(decay, (1+t)/(10+t))` for `Tf`, the ratio in
  `f64` then rounded once). `Ema::new(source, shadow, config)` pairs two models by
  path and shape and refuses a shared parameter, an invalid config and `f16`/`bf16`
  weights (`Error::Unsupported`), changing neither model. `update(step)`,
  `reset`, `set_config`, `updates`, `state_dict`,
  `stage_state_dict`/`load_state_dict` (all or nothing; non-strict ignores unknown
  paths and restarts missing ones from the source).
- `Trainer::with_ema / ema / ema_mut / take_ema`; `step` queues the update right
  after `step_scaled`, with the trainer's 1-based counter.
- `Checkpoint::{ema, ema_updates}` (an exact integer), `with_ema`, `restore_ema`,
  `stage_ema`. Binary format **version 4**, written only when an average is
  present; v2/v3 files are written byte for byte as before, and a v≤3 header with an
  average is refused. JSON carries it too.

**Where it differs from the plan, and why.**
- *No host read at construction or reset.* The plan allowed one (as
  `ReferencePolicy::snapshot` pays); `Ema::new` and `reset` seed the shadow with a
  device copy (`Tensor::deep_clone`) of each trainable weight instead — one launch
  per trainable parameter, once.
- *`d_t = 0` and `d_t = 1` are defined, not computed.* `e + 1·(p − e)` is not
  always `p` in floating point, so R5 ("decay 0 equals the weights, decay 1 the
  first ones", exactly) could not hold through the kernel. At `d_t = 0` the shadow
  takes the weight's buffer, at `d_t = 1` it is left alone, with no launch.
- *The host-twin gate is the backend, not the libm probe.* The recurrence uses no
  libm; the realistic difference is a shader compiler contracting into a fused
  multiply-add. Twins are exact on the CPU runtime (asserted) and elsewhere exact
  or within `4·ε·max(1, |x|)`, one `skipped:` line per test saying which. Metal
  does contract: 18,492 of 508,505 kernel-test values differ, all within budget.
- *Recurrence twins are one-step* (from the device's previous average), so the
  wgpu budget is per step. On the CPU runtime every step is exact, which makes the
  five-step claim the same.
- *R2 (no launch when empty) is in the footprint binary*, which is alone in its
  process: the launch counter is process-wide.
- *Warm-up counts the trainer's step*, as the plan specifies; an average reset
  late in a run (after a critic warm-up) is already past it. Documented.
- `ema_step` returns `Result` (the plan wrote `Tensor`): a public kernel should not
  read out of bounds on mismatched shapes.

**As implemented (Python).**
- `mamba3_rl.EmaConfig(decay, warmup="none"|"tf")`, validated in the constructor
  (`ValueError`; a non-number `TypeError`), `decay`, `warmup`, `decay_at`, `repr`
  that evaluates back, `==`.
- `PpoLearner(..., ema=None)`, `ImitationLearner(..., ema=None)`;
  `ema_policy` (a `Policy` handle onto the shadow, never trained by the optimizer),
  `ema_updates`, `ema_config`, `reset_ema()` (refused without an average and, for
  PPO, between `collect()` and `update()` with the pending-window message).
- `Policy.fingerprint()` (T6's part the tests need): `StateDict::fingerprint`, one
  host read per parameter.
- Checkpoints carry the average's weights and counter at both levels and
  `trainer_config.ema` (`{"decay", "warmup"}` or `null`; older files read as
  `null`). A file whose weights and `trainer_config.ema` disagree is refused as
  damaged. `config="verify"` lists `ema`, `ema.decay`, `ema.warmup` differences;
  `"checkpoint"` adopts a saved average (configuration via `Ema::set_config` onto the
  learner's own shadow, weights, counter), and refuses adopting "none" over a live
  average; `"live"` re-seeds the learner's average from the loaded weights when the
  checkpoint has none (`"warm"`, note `"ema re-seeded from loaded weights"`) and
  ignores a checkpoint's average the learner does not keep (note). A warm start
  re-seeds too. `from_checkpoint` rebuilds the average. All staged before the
  environment's `load_state()`, applied after.
- README ("A moving average of the weights", saving rules, "What crosses the
  boundary": no reads for the average), `_mamba3_rl.pyi`, `__init__.py`.

**Tests.**
- `tests/train_ema.rs` (CPU and wgpu): R1 kernel against its host twin over 20
  lengths and 5 decays; R4 recurrence over five steps; R5 decay 0 and 1 exact; R6
  TF warm-up (with R4 repeated); R7 invalid configs and architectures refused,
  nothing changed; **R8 training identical with and without the average** (StepInfo,
  weights, AdamW moments and counter, to the bit); R9 shadow independent of
  reloads, rollouts and other trainers; R10 frozen parameters follow the source and a
  shared buffer is never written; R11 failed steps change nothing; identical runs
  agree; f16/bf16 refused.
- `tests/train_ema_footprint.rs`: R2; **F1** attaching holds exactly the trainable
  weights rounded to the measured buffer alignment — CPU 11,432 bytes for 29
  tensors of 2,857 scalars (8-byte alignment), wgpu 16,384 (256-byte), critic
  frozen 11,360 / 15,872; **F2** reserved bytes flat over 200 steps, 2 reads per
  step with or without; **F3** +29 launches per step for 29 trainable tensors, 27
  with the critic frozen, constant across steps.
- `tests/kernel_literals.rs`: the scan reaches `adamw_kernel` and `ema_kernel`.
- `tests/train.rs`: C1 both formats; **C2** golden v2 and v3 checkpoints in both
  formats (`tests/golden/checkpoint_*`, written while the checkpoint code was still
  `e7a77e2`'s) written identically; C3 version 4 and v3-with-average refused; C4
  malformed averages refused with the average unchanged; C5 non-strict; C6
  `2^24 + 1` counter.
- `tests/rl_resume.rs`: **R12/R13** PPO (TF warm-up, reference, masks, schedule)
  and DAgger continue exactly with the average's fingerprint and counter compared
  every round, and at the optimizer level on a given window; **R14** re-seeding the
  restored average leaves everything else exact and the average different every
  round; a malformed average is refused after everything else stages, and the next
  round equals a twin's.
- `tests/train_ema_parity.rs` + `tests/golden/ema_parity.json`: **P5** one PPO
  scenario recorded per backend (CPU, wgpu<wgsl> with `block_tiled`); regenerate
  with `MAMBA3_WRITE_EMA_PARITY=1`.
- Python `tests/test_ema.py` (both learners where both apply): P1 config, P2
  default, **P3** training identical, P4 NumPy recurrence per optimizer step (with
  and without warm-up), **P5** parity with the Rust record, P6 `ema_policy` usable
  and untrained, P7 `reset_ema`, P8 both levels (and JSON), P9 verify, P10 adopt,
  P11 live, P12 `from_checkpoint`, P14 freeze. `test_policy.py`: P0.
  `test_continuation.py`: **P13** the cross-process full continuation with an
  average (PPO through `load_checkpoint` and `from_checkpoint`, DAgger), plus the
  in-process negative control.

Found along the way: N7 (below).

**Cost.** `bench/ema_overhead.py`, 2026-09-14, Apple M1: two learners of one seed
alternating rounds at the exp010 shape over `RecallEnv` (208 lanes × 120 steps,
`d_model=256`, 4 layers, 971,125 parameters, 2 epochs, 1 minibatch, `row_tiled`),
`ema=EmaConfig(0.99)` on one, median `update()` over 20 rounds after 2 warm-up:

| | CPU runtime | wgpu<wgsl> |
|---|---|---|
| `update()`, EMA off | 7.297 s (min 7.234, max 7.746) | 1.752 s (1.741 / 1.785) |
| `update()`, EMA on | 7.305 s (7.235 / 7.920) | 1.755 s (1.743 / 1.799) |
| difference | +0.10%, within the spread | +0.17%, within the spread |
| launches per optimizer step | 1,239 → 1,290 (+51, one per trainable tensor) | same |
| added device bytes | 3,884,500 (3.70 MiB), computed from the trainable count; F1 measures it | same, in 256-byte buffers |
| host reads | none added (F2) | none added |

At exp010's own 988,054 parameters the copy is 3.77 MiB. The expectation was "under
3% of an update on wgpu"; the per-parameter launches were not fused.

## Q1–Q3 — tooling

- **Q1.** `cargo clippy --all-targets -- -D warnings` passes for the crate (CPU and
  wgpu features) and the bindings. One scoped `allow` with its reason
  (`tests/ssm.rs`, a reference implementation indexing three buffers), besides the
  pre-existing comptime `collapsible_if`.
- **Q2.** `tools/check.sh [--wgpu] [--no-python]` — see above. No CI workflow was
  added and nothing is run remotely.
- **Q3.** `tools/build_wheel.sh cpu|wgpu [out] [--smoke]`: CPU wheels always get
  `--auditwheel=repair`; `--smoke` installs into a fresh venv, fails on any `otool -L`
  link outside the system and the wheel, and imports with the library search paths
  cleared.

## Findings along the way

- **N1 — training was not bit-reproducible run to run.** Two identical PPO runs in
  one process agreed on every sampled action, reward and reference score but their
  weights differed by a few ulp after the first update, and the loss by one ulp later
  (imitation too, less often). Bisected: repeating one forward, loss and backward 30
  times gave identical bits, so no kernel was racy; the first divergence was the
  update. `Grads` kept parameter gradients in a `HashMap`, whose iteration order is
  randomised per instance, and `grad_sum_squares` lays each gradient's partial sums
  out in that order before one reduction — so the global norm, the clip factor and
  every weight came out a few ulp apart. `Grads` is a `BTreeMap` now (ids follow
  creation order); three identical runs agree to the bit at every round, and
  `tests/rl_resume.rs`, `test_continuation.py`, `test_game.py` and `test_resume.py`
  compare losses and weights exactly again (the first two across processes).
- **N6 — on a GPU, the matmul tuner broke the same reproducibility twice.** Within
  a process: two threads tuning one shape each inserted their winner, so the second
  replaced a plan the first had already computed with, and later calls ran another
  kernel — `rl_resume`'s imitation run ended with two `actor.bias` values apart on
  wgpu. The first recorded plan is now kept (`entry().or_insert`). Across processes:
  `auto` picks kernels by timing, so a restore in another process matched actions but
  not losses (`0.015133568` against `0.015133644`). Pinning a kernel fixes that, and
  Python could not: `mamba3_rl.set_matmul_kernel`/`matmul_kernel` and a checked
  `MAMBA3_MATMUL_KERNEL` now match Rust's `set_default_kernel`;
  `test_continuation.py` pins `block_tiled` on non-CPU backends and compares exactly.
- **N7 — reading an empty tensor back panicked.** `Tensor::try_to_data` handed the
  zero-length slice a read returns to `bytemuck`, which refuses it as unaligned for
  `f32`. Found by T1's kernel test at length 0; an empty tensor now reads back as an
  empty vector without a read (`tests/tensor.rs`).
- **N2 — CubeCL 0.10's CPU `read` looks memory up in the caller's stream**
  (`cubecl-cpu` `compute/server.rs` `read`: `self.scheduler.stream(&stream_id)`,
  where the wgpu server uses `desc.handle.stream`). A buffer allocated on one thread
  and read on another panicked with "Memory slice N doesn't exist" — every host
  environment on a `ParallelEnvs` worker that read its actions, and every worker
  save. Worked around in `backend::read_handle` (the read runs under the buffer's
  stream via `StreamId::executes`, after the caller's own flush). Worth reporting
  upstream (not done: needs approval).
- **N3 — Metal's reassociating math broke `log1p_f32`.** Kahan's `ln(u)·x/(u − 1)`
  relies on `(1 + x) − 1` not folding to `x`; on wgpu over Metal it did,
  `softplus(-16.6)` was `1.65e-7` instead of `6.1e-8` (14.6M ulp), and
  `log_sigmoid`/`log1mexp` with it. `log1p_f32` uses `2·atanh(x/(2 + x))` as a
  five-term series for `|x| < ¼`; device-vs-host error is now ≤ 27 ulp and the f64
  reference budgets still hold.
- **N4 — the fused rollout adopted a stale observation after a restore.**
  `collect_fused` writes the next observation into the collector's buffer and ends
  by adopting the world's, assuming one buffer; after a restore they were two. It now
  adopts the world's at the start of the window too (a no-op in a live run).

---

## Earlier records (as implemented)

### F1, F2

**F1 — PPO reference anchor.** `PpoConfig::reference_coeff`,
`PpoBatch::reference_log_probs`, `rl::reference_log_probs`,
`PpoLoss::reference_kl`, `PpoStats::reference_kl`; Python `PpoLearner(...,
reference=policy)`. Loss gains `coeff · E[exp(d) − d − 1]`, `d = log π_ref(a) −
log π_θ(a)`; `0.0` is byte-identical. *Why:* two runs drifted from a good clone to a
do-nothing policy (+707 → −756 over 70 rounds) at `approx_kl` ≈ 0.0002 a round.
Tests: `tests/rl_reference.rs`.

**F2 — precision capability check.** `supports_matmul_precision`,
`try_set_matmul_precision`; the Python setter uses the checked one. *Why:*
`set_matmul_precision("bf16")` on wgpu was accepted and then aborted in the WGSL
compiler 17,084 times. The capability is now a runtime query (F2b).

### A7 — matmul precision from the environment

`MAMBA3_MATMUL_PRECISION` is read at import through `try_set_precision_from_env`,
which **raises** for a value the backend cannot honour (it does not clear to F32 and
warn). *Why:* `f32`/`f16`/`bf16` measured 3.36/3.40/3.45 s — the variable was never
read. Tests: `bindings/python/tests/test_module.py`.

### A4 — episode returns

`Collector::episode_return() -> Result<(mean, count)>`, both `[1]` device tensors;
lanes carry their in-progress return across windows. Python `evaluate(...)` and
`PpoLearner.episode_return()` return `None` when no episode completed, in one read of
a packed two-element tensor. *Why:* a window with no completed episode reported
−31.5 where the environment's own bookkeeping said −1.3. Tests: `tests/rl_learn.rs`,
`test_ppo.py`.

### A3 — learning-rate schedules

`mamba3_rl.LrSchedule` (`constant`, `cosine`, `linear`, `inverse_sqrt`, `step`,
`rate_at`) and `lr_schedule=` on both learners; `None` is constant. A schedule
advances per optimizer step (epochs × minibatches). *Why:* at `5e-4` the gradient
norm spiked to 6.2 and agreement oscillated; at `1e-4` it was monotone. Tests:
`test_config.py`, `test_ppo.py`.

### A2, A5 — checkpoints

`Checkpoint { step, state, optimizer, optimizer_steps, metadata, rollout, blobs }`;
`with_optimizer`, `restore_optimizer`, `restore_training`, `stage_training`. AdamW
moments are keyed by parameter path; counters are JSON integers. `.json` writes the
legacy text encoding; anything else the binary container (magic `MAMBA3CK`, version,
JSON header, `f32` payload, v3 blobs), sniffed on load. Measured: a 988,054-parameter
policy is 12.54 MB as JSON and 3.96 MB as `.m3ck` (3.17×). Tests: `tests/train.rs`
(`resuming_is_indistinguishable_from_not_stopping`, `exact_restore`, format
round-trips and an inline pre-A5 JSON fixture, `a_legacy_json_fixture_still_loads`).

### A1 — legal-action masking

`VecEnv::action_mask() -> Result<Option<Tensor>>`; `None` means all legal for that
step. The masked draw, the recorded `[envs, steps, actions]` column, the PPO replay,
the entropy term and imitation's cross entropy apply the same mask. Masked logits are
`F::min_value()`, not `-inf` (WGSL). An all-zero row or a value other than 0/1 is
refused. Tests: `tests/rl_masking.rs` (23), `tests/rl_fused.rs`, `test_masking.py`.

### Kaizen round (R1, K1–K8)

**R1/K7.** `ReferencePolicy::score` never carried a cache; it now starts from an
explicit zero history and returns the end cache. `tests/rl_reference.rs::oracle`
checks it against an independently stepped reference (≤ 1e-5, measured ~1.2e-7) over
resets at window ends, inside windows and never, SISO/MIMO, and a masked window.
**K1** atomic restore (`stage_state_dict`, replacing optimizer state). **K2** training
configuration in learner checkpoints (`config="verify" | "checkpoint" | "live"`).
**K3** exact integer counters. **K4** optional per-step masks. **K5** masks in
`Rollout`. **K6** masks through the fused rollout (`GameLogic::legal`). **K8/V1**
docs, examples, measurements.

K8 measurements (`cargo run --release --example measure_reliability`; policy
`obs_dim=39`, 37 actions, `d_model=256`, 4 layers, 988,054 parameters; 208 lanes × 120
steps):

| | CPU runtime | wgpu<wgsl>, Apple M1 |
|---|---|---|
| `ReferencePolicy::score`, one window | 1.20 s | 268 ms |
| collect + batch, same window | 2.24 s | 1.33 s |
| reference cache carried | 29.76 MiB | 29.76 MiB |
| mask column `[208, 120, 37]` | 3.52 MiB | same |
| mask validation read | 1.04 ms | 1.81 ms |
| checkpoint `.json` / `.m3ck` | 12.54 MB / 3.96 MB | same |

## Performance

`PLAN.md` owns it. Two measurements from the driving project: 97% of a PPO round is
the policy's forward and backward passes (environment 3.1%, rollout 58.8%, update
38.1% at `d_model=256`, 208 environments, `wgpu<wgsl>`), so fusing buys at most a few
percent at that size; and `f16` bought nothing at that size (3.34 s against `f32`'s
3.36 s).

## Suites

Apple M1, 2026-09-14, `tools/check.sh --wgpu` at T1 step 6 (`9024dc2`; every step's
exit code 0). Rust counts are test results across result groups (24 binaries plus
doc-tests).

| | result |
|---|---|
| `cargo fmt --check` (root, bindings) | clean |
| `cargo clippy --all-targets -- -D warnings` (root, bindings) | clean |
| Rust, CPU runtime | 297 passed, 0 failed, 25 groups |
| Rust, wgpu<wgsl> | 297 passed, 0 failed, 25 groups. Skips, each printing its reason: 4 bf16 cases (`backend::supports_dtype` says no), 2 bit-exact twin tests (Metal libm differs; the within-budget twin tests run instead), 4 EMA host twins held to `4·ε·max(1, |x|)` instead of bits (Metal contracts into a fused multiply-add), bf16 EMA refusal (no bf16 buffers) |
| Python, CPU wheel (`--auditwheel=repair`, fresh venv) | 210 passed, 1 skipped (the WGSL-only bf16 refusal test) |
| Python, wgpu<wgsl> wheel (fresh venv) | 209 passed, 2 skipped (bf16 environment-variable cases WGSL cannot express) |

At T1 step 1 (`eb95a97`: `e7a77e2` plus `Policy.fingerprint()` and its one test):
Rust CPU 272 passed, 22 groups; Python CPU 178 passed / 1 skipped.

Baseline before the open-issues round (`198b3f3`): CPU 252 passed; wgpu 21 failed
(`distributions` 13, `distributions_bitexact` 4, `mixed_precision` 4); 46 clippy
warnings.
