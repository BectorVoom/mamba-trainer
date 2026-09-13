# Defect-fix plan

Execution document. Each task below is self-contained: goal, exact files and line
anchors, the API to add, the traps, the test, and the command that proves it. Work
tasks in the order given; each one lands on its own and leaves the suite green.

Issues were found by driving this crate's reinforcement-learning stack from a real
project (a Kaggriculture agent, four experiments) — every "why" is a measurement,
not a guess. `PLAN.md` is the separate training-speed plan and stays authoritative
for kernel work; §Performance here only records what this workload measured.

---

## 0. Before you start

```bash
cd /Users/ods/Documents/mamba-trainer

# Baseline: must pass before and after every task. Measured 2026-09-13 on
# kaizen-k1-k8 (Apple M1, CPU runtime): 252 passed, 0 failed across 19 result groups. Most of the wall time is
# tests/tensor.rs (~200 s).
cargo test --release --no-default-features --features cpu

# The same suite on the GPU. Not optional for kernel changes: a kernel that WGSL
# cannot compile does not fail loudly, it computes zeros (see W1).
CARGO_TARGET_DIR=/tmp/target-wgpu cargo test --release --no-default-features --features wgpu --no-fail-fast

# Python binding (only for tasks that touch bindings/python): build a wheel per
# backend into its own venv, then run the suite in each.
cd bindings/python
maturin build --release --no-default-features --features cpu --auditwheel=repair -o /tmp/wheels-cpu
maturin build --release --no-default-features --features wgpu -o /tmp/wheels-wgpu
python -m pytest tests -q
```

**Check the exit code, not the output.** `cargo test … | grep …` reports the
*grep's* status; a compile failure in one test target looks like success. Always:

```bash
cargo test --release --no-default-features --features cpu > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|^error" /tmp/t.log
```

### Repo map for these tasks

| path | what lives there |
|---|---|
| `src/rl/env.rs:62` | `VecEnv` trait |
| `src/rl/buffer.rs:89` | `TrajectoryBuffer` fields |
| `src/rl/collect.rs:420` | `Collector::episode_return` |
| `src/rl/fused.rs` | the one-kernel rollout step |
| `src/rl/ppo.rs` | `PpoConfig`, `PpoBatch`, `ppo_objective`, `PpoTask` |
| `src/rl/imitation.rs` | `ImitationBatch`, `behaviour_cloning_loss` |
| `src/tensor/ops/rl.rs:256` | `sample_categorical` |
| `src/train/optim.rs:20` | `Optimizer` trait; `AdamW` at `:149`, `Moments` at `:156` |
| `src/train/checkpoint.rs:18` | `Checkpoint`; `save` at `:55`, `load` at `:62` |
| `bindings/python/src/lib.rs:140` | `#[pymodule] fn _mamba3_rl` |
| `bindings/python/src/learner.rs` | `PyPpoLearner`, `PyImitationLearner`, `evaluate` at `:736` |
| `bindings/python/python/mamba3_rl/_mamba3_rl.pyi` | type stubs — **update with every API change** |

### Invariants that must survive every task

1. **A replayed window reproduces the actor's log-probabilities.** `tests/rl_learn.rs`
   asserts every first-epoch PPO ratio is 1. Anything applied when acting must be
   applied identically when replaying.
2. **The fused and unfused rollouts produce byte-identical windows.**
   `tests/rl_fused.rs`. A change to the draw goes in both paths or neither.
3. **Collection performs zero host reads.** `tests/rl_collect_footprint.rs`, and
   reserved bytes and per-step dispatch count stay flat.
4. **Defaults do not change behaviour.** New knobs are off by default and the
   objective stays byte-identical when they are.

### Trap that has already bitten

`PpoBatch` is constructed with a struct literal in `tests/rl_learn.rs:879`. Adding a
field breaks that target only, and `cargo check` on the lib will not show it. After
any field addition: `cargo test --no-default-features --features cpu --no-run`.

---

## Status

| | task | severity | effort | state |
|---|---|---|---|---|
| F1 | PPO reference anchor | high | — | **done** |
| F2 | precision capability check | high | — | **done** |
| A7 | checked `MAMBA3_MATMUL_PRECISION` from Python | low | 15 min | done |
| A4 | completed-episode return accounting | medium | 1 h | done |
| A3 | Python LR schedules | medium | 2 h | done |
| A2a | optimizer and trainer state in checkpoints | high | 4 h | done |
| A2b | exact learner continuation including environment state | high | — | open |
| A5 | versioned binary checkpoints with JSON loading | medium | 3 h | done |
| A1 | legal-action masking | high | 1–2 days | done (K4–K6, WGSL fix) |
| A6 | `GameLogic` reachable from Python and routed fused | low | 1 day | open |
| R1 | the reference scores its own recurrent history | high | — | **done** (K7: was not carried at all before) |
| K1 | atomic learner checkpoint restore | high | — | done |
| K2 | training configuration in learner checkpoints | high | — | done |
| K3 | exact integer counters | high | — | done |
| K4 | optional masks mean all-legal per step | medium | — | done |
| K5 | masks in `Rollout.step` / `Rollout.evaluate` | medium | — | done |
| K6 | masks through the fused rollout | low | — | done |
| V1 | docs, examples, wgpu wheel run, measurements | required | — | done (K8) |
| W1 | WGSL gaps outside the RL path (distributions, bf16/bit-exact tests) | medium | — | **open** |

A2 and A5 both change the checkpoint format — do them together, A2 first.
Every "done" row above links to its tests in the sections below; the kaizen
round (R1, K1–K8) is in [its own section](#kaizen-round-r1-k1k8).

---

## Done: F1, F2

**F1 — PPO reference anchor.** `PpoConfig::reference_coeff`,
`PpoBatch::reference_log_probs`, `rl::reference_log_probs(&reference, &batch)`,
`PpoLoss::reference_kl`, `PpoStats::reference_kl`; Python
`PpoLearner(..., reference=policy)` and `PpoConfig(reference_coeff=…)`.
Loss gains `coeff · E[exp(d) − d − 1]`, `d = log π_ref(a) − log π_θ(a)`.
Default `0.0` is byte-identical. Tests: `tests/rl_reference.rs` (11, including
the K7 oracle).

*Why:* two runs drifted from a good clone to a do-nothing policy (+707 → −756 over
70 rounds) with `approx_kl` ≈ 0.0002 a round. The clip bounds one update; nothing
bounded two hundred.

**F2 — precision capability check.** `supports_matmul_precision(&device, mode)`,
`try_set_matmul_precision(&device, mode)`; the Python setter goes through the latter.

*Why:* `set_matmul_precision("bf16")` on wgpu was accepted, then aborted in the WGSL
compiler on a worker thread — 17,084 panics, no result.

*Follow-up:* capability is a substring of `Device::name()`. Replace with a real
CubeCL capability query if one appears.

---

## A7 — `MAMBA3_MATMUL_PRECISION` is inert from Python

**Goal.** Setting the env var must either work or not exist.

**Why.** Measured 3.36 / 3.40 / 3.45 s for `f32` / `f16` / `bf16` — noise around one
number, because `set_precision_from_env()` (`src/tensor/ops/matmul.rs:153`) is never
called by the extension module.

**Files.** `bindings/python/src/lib.rs:141`.

**Steps.**
1. In `fn _mamba3_rl`, before registering classes, call
   `mamba3::tensor::ops::matmul::set_precision_from_env();`.
2. It is unchecked (F2's guard needs a device) — so follow it with a validation
   that clears the mode back to `F32` and emits a Python warning if the compiled
   backend cannot honour it:
   ```rust
   let device = mamba3::backend::Device::<R>::default();
   let wanted = mamba3::tensor::ops::matmul::matmul_precision();
   if !mamba3::tensor::ops::matmul::supports_matmul_precision(&device, wanted) { … }
   ```
3. Document the variable in the module docstring next to `set_matmul_precision`.

**Test.** `bindings/python/tests/test_config.py`: with
`MAMBA3_MATMUL_PRECISION=f16` in `os.environ` before import, `matmul_precision()`
returns `"f16"`; with `bf16` on a WGSL build it returns `"f32"` and warns.

**Verify.**
```bash
MAMBA3_MATMUL_PRECISION=f16 python -c "import mamba3_rl; print(mamba3_rl.matmul_precision())"   # f16
MAMBA3_MATMUL_PRECISION=bf16 python -c "import mamba3_rl; print(mamba3_rl.matmul_precision())"  # f32 + warning
```

**Done when** both lines print the values above and the suite is green.

---

## A4 — `evaluate` and `episode_return` change unit silently

**Goal.** A window in which no episode finished must not report a total where every
other window reports a mean.

**Why.** Both divide total reward by `max(dones, 1)`. Measured: `episode_return`
reported −31.5 where the environment's own bookkeeping said −1.3, a factor of 24,
because all environments had been reset together and so finished together, leaving
two windows in three with no episode boundary. The reading was correct and unusable.

**Files.** `src/rl/collect.rs:420` (`episode_return`), `src/rl/parallel.rs:517`
(delegates), `bindings/python/src/learner.rs:736` (`evaluate`) and the
`PyPpoLearner::episode_return` getter.

**API.**
```rust
// src/rl/collect.rs — replace the existing signature.
/// Mean reward per completed episode, and how many completed.
///
/// The count is returned because the mean is meaningless without it: a window in
/// which nothing finished has no episodes to average over.
pub fn episode_return(&self) -> Result<(Tensor<R, E>, Tensor<R, E>)>;
```
Python: `evaluate(...) -> float | None` and `PpoLearner.episode_return() -> float | None`,
returning `None` when the count is zero. `Stats.episode_return` is already
`Optional[float]` in the stubs, so no stub change is needed there.

**Steps.**
1. Return `(sum / max(count,1), count)` from the Rust method; do **not** clamp away
   the count.
2. `parallel.rs` delegates — update the signature only.
3. In the binding, read both, and return `None` when `count == 0.0`.
4. Update `README.md`'s "What crosses the boundary" table, which documents
   `episode_return()` as one of the two deliberate host reads — it is now two reads
   or one read of a two-element tensor; make it the latter to keep the count honest.

**Traps.** Keep it a single read. Two `to_f32()` calls double the synchronisation
this method exists to bound.

**Test.** `tests/rl_learn.rs`: collect a window shorter than one episode of
`RecallEnv`; assert the count is 0. Python: `evaluate(policy, env, steps=1)` returns
`None`.

**Verify.** `cargo test --release --no-default-features --features cpu --test rl_learn`

**Done when** a sub-episode window yields `None` in Python and `count == 0` in Rust.

---

## A3 — the learners take no learning-rate schedule

**Goal.** `PpoLearner` and `ImitationLearner` accept an `LrSchedule`, as `Trainer`
already does.

**Why.** Measured on a real run: at `5e-4` the gradient norm spiked to 6.2 and
held-out agreement oscillated between 0.90 and 0.95; at `1e-4` the norm was 0.86 and
the curve was monotone. Finding that cost a restart. `LrSchedule`
(`src/train/sched.rs:5`) exists with four variants and is already threaded through
`TrainerConfig`; the learners simply do not expose it.

**Files.** `bindings/python/src/learner.rs` — the `trainer(...)` helper both
constructors call, and both `#[pyo3(signature = …)]` blocks. New
`bindings/python/src/config.rs` class `PyLrSchedule`.

**API (Python).**
```python
m3.LrSchedule.constant()
m3.LrSchedule.cosine(total_steps, warmup_steps=None, min_ratio=0.1)
m3.LrSchedule.linear(total_steps, warmup_steps=None, min_ratio=0.0)
m3.LrSchedule.inverse_sqrt(warmup_steps)
m3.LrSchedule.step(every, gamma)

m3.PpoLearner(policy, env, learning_rate=3e-4, schedule=None, ...)
m3.ImitationLearner(policy, env, learning_rate=3e-3, schedule=None, ...)
```
`schedule=None` means `LrSchedule::Constant`, i.e. today's behaviour.

**Steps.**
1. Add `PyLrSchedule` mirroring the Rust enum, with static constructors and a
   `.rate_at(base, step)` for testing.
2. Extend the `trainer(...)` helper to take `Option<LrSchedule>` and set
   `TrainerConfig::schedule`.
3. Add the parameter to both learner signatures — **both** the `#[pyo3(signature)]`
   list and the Rust argument list, in the same order.
4. Update the `.pyi`.

**Traps.** The two learner constructors have near-identical prologues. A textual
edit that matches both is how F1's first attempt broke: pyo3 errors with `missing
signature entry for argument …` when the Rust arg list and the signature list
disagree. Edit each explicitly.

**Test.** `bindings/python/tests/test_config.py`: `LrSchedule.cosine(100).rate_at(1.0, 50)`
lies strictly between the rate at 1 and at 100. `test_ppo.py`: a learner built with
`cosine(10)` reports a strictly decreasing `Stats.learning_rate` over 10 rounds.

**Verify.** `cd bindings/python && pytest -q`

**Done when** `Stats.learning_rate` tracks the schedule and `schedule=None` is
unchanged from today.

---

## A2 — checkpoints hold no optimizer state

**Goal.** A saved run can be resumed: same weights *and* same optimizer moments and
schedule position.

**Why.** `Checkpoint` is `{ step, state, metadata }` — weights only
(`src/train/checkpoint.rs:18`). AdamW's moments live in
`AdamW::state: HashMap<ParamId, Moments>` (`src/train/optim.rs:149`) and are lost.
Four continuation runs in one experiment each `--init`-ed weights and restarted Adam
from zero moments; each showed the same dip-then-recover in held-out agreement over
the first evaluations — the shape of an optimizer re-warming, not of a policy
improving. One restart was then spent tuning the learning rate for what may partly
have been this.

**Files.** `src/train/optim.rs`, `src/train/checkpoint.rs`,
`src/train/trainer.rs` (the step counter the schedule reads),
`bindings/python/src/learner.rs`.

**API (Rust).**
```rust
// src/train/optim.rs — on the Optimizer trait, defaulting to empty so Sgd need not
// implement it until it has state worth keeping.
fn state_dict(&self) -> StateDict { StateDict::default() }
fn load_state_dict(&mut self, state: &StateDict, strict: bool) -> Result<()> { … }

// src/train/checkpoint.rs
pub struct Checkpoint {
    pub step: u64,
    pub state: StateDict,
    pub optimizer: Option<StateDict>,   // new
    pub metadata: serde_json::Value,
}
impl Checkpoint {
    pub fn with_optimizer<R, E, O: Optimizer<R, E>>(self, optimizer: &O) -> Self;
    pub fn restore_optimizer<R, E, O: Optimizer<R, E>>(&self, optimizer: &mut O) -> Result<()>;
}
```
Key the moments `"<param path>.m"` / `"<param path>.v"` — **not** by `ParamId`,
which is not stable across process runs. `Module::named_parameters()` gives the
path; the optimizer only has the id, so `state_dict` needs the params passed in:
`fn state_dict(&self, params: &[(String, Param<R, E>)]) -> StateDict`.

**API (Python).**
```python
learner.save(path)          # policy weights + optimizer moments + step + schedule position
PpoLearner.load(path, env, ...)  # a learner that continues, not one that restarts
```

**Steps.**
1. Give `AdamW` `state_dict`/`load_state_dict` over `(path, Param)` pairs; store
   `steps` as a scalar entry so bias correction resumes correctly.
2. Add the `optimizer` field to `Checkpoint`, defaulting to `None` in serde so old
   files still load.
3. Persist `Trainer::step_count()` and restore it, so `LrSchedule` resumes at the
   right point (depends on A3 for the schedule to exist; the counter is worth
   saving regardless).
4. Add the Python `save`/`load` on both learners.

**Traps.**
- Serde default on the new field, or every existing checkpoint fails to parse.
- Bias correction uses `self.steps`; restoring moments without it gives a wrong
  first step after resume.
- `Checkpoint::capture` is generic over `M: Module`; the optimizer is a separate
  generic — do not try to fold them into one call.

**Test.** `tests/train.rs::resuming_is_indistinguishable_from_not_stopping`: train
20 steps, save, build a fresh policy+optimizer, restore, train 20 more; assert every
weight matches a single 40-step run to `1e-6`. That is the only test that proves
resumption; a round-trip equality test does not.

**Verify.** `cargo test --release --no-default-features --features cpu --test train`

**Done when** the 20+20 run equals the 40 run, and an old weights-only checkpoint
still loads.

---

## A5 — checkpoints are JSON

**Goal.** Weights save as binary; JSON still loads.

**Why.** A 1,009,302-parameter policy writes a **13.7 MB** `.json`. The same weights
as `f32` in a compressed `.npz` are **3.76 MB**, 3.6× smaller, and loading the JSON
is a parse of a million decimal floats. In one session these files were a real
fraction of a disk that filled up and stopped the run.

**Files.** `src/train/checkpoint.rs:55` (`save`), `:62` (`load`),
`src/nn/module.rs:342` (`StateDict::save`/`load`).

**Format.** Header + payload in one file:
```
magic  b"MAMBA3CK"                (8 bytes)
version u32 = 1                   (4)
header_len u32                    (4)
header  JSON                      (header_len bytes)
        {"step":…, "metadata":…,
         "tensors":[{"name":…,"shape":[…],"offset":…,"len":…}, …],
         "optimizer":[…]}         (same shape as tensors, or absent)
payload little-endian f32, tightly packed, in header order
```

**Steps.**
1. `Checkpoint::save` picks by extension: `.json` keeps today's writer, anything
   else writes binary. Default the docs and examples to `.m3ck`.
2. `Checkpoint::load` sniffs the magic; falls back to JSON.
3. Keep `StateDict`'s serde derives — the JSON path must stay byte-compatible.

**Traps.** Do this *after* A2, or the format changes twice. `f32` on disk regardless
of the compute element type, which is what the current writer already promises.

**Test.** `tests/train.rs`: round-trip a state dict through both formats, assert bit
equality of every value; assert the binary file is smaller; assert a JSON file
written before the change still loads (commit a small fixture under `tests/golden/`).

**Verify.** `cargo test --release --no-default-features --features cpu --test train`

**Done when** both formats round-trip bit-exactly and `.m3ck` is ≥3× smaller.

---

## A1 — no action masking

**Goal.** An environment can declare which actions are legal per step; the draw
never returns an illegal one and the replay scores the same distribution.

**Why.** In the driving project a turn is 26 decisions and **62% of them are idle by
construction** — unit slots past the number of hands hired, order slots with nothing
to order. Many of the rest are structurally invalid in a given state (plant with no
seed, place with nothing carried, a purchase the bank cannot cover) and every one
decodes to a silent no-op. So capacity goes on learning which actions do nothing
here, and exploration goes on rediscovering it. That is expensive in a domain where
a separate measurement showed the season collapses below ~0.95 action agreement.

**Files.** `src/tensor/ops/rl.rs:256` (`sample_categorical`), `src/rl/env.rs:62`
(`VecEnv`), `src/rl/buffer.rs:89` (a new column), `src/rl/collect.rs` (record it),
`src/rl/fused.rs` (the same in the fused kernel), `src/rl/ppo.rs`
(`PpoBatch`, `ppo_objective`), `src/rl/imitation.rs`
(`ImitationBatch`, `behaviour_cloning_loss`), the binding, the stubs.

**API.**
```rust
// src/rl/env.rs, on VecEnv, defaulting to None like expert_actions.
/// `[envs, action_dim]`, 1 where the action is legal on the observation most
/// recently returned and 0 where it is not. `None` means every action is legal.
fn action_mask(&self) -> Option<Tensor<R, E>> { None }

// src/tensor/ops/rl.rs
pub fn sample_categorical_masked<R, E>(
    logits: &Tensor<R, E>, mask: Option<&Tensor<R, E>>, temperature: f32, seed: u64,
) -> Result<(IdTensor<R>, Tensor<R, E>)>;
```
`PpoBatch::action_mask: Option<Tensor>` `[envs, steps, actions]`;
`ImitationBatch::action_mask` likewise.

**Steps.**
1. Masked draw: add `−inf` (use `f32::NEG_INFINITY`, not a large negative — the
   softmax must give exactly zero) inside the existing kernel's row loop, before the
   max and the normaliser. Keep the unmasked path branch-free at comptime.
2. Record the mask into a new `[envs, steps, actions]` buffer column.
3. **Apply the same mask in the replay**, before `Categorical::from_logits` in
   `ppo_objective` and in `behaviour_cloning_loss`. Invariant 1 fails otherwise.
4. Mirror the draw change in `src/rl/fused.rs` (invariant 2), and pass the mask
   through `GameLogic` — a device game can compute its own mask in `transition`.
5. Thread through the binding: `VecEnv` protocol gains an optional
   `action_mask() -> np.ndarray | None`, `[num_envs, action_dim]` float32.

**Traps.**
- An all-zero mask row is a division by zero in the normaliser. Treat a row with no
  legal action as "every action legal" and count it — add a `masked_rows_empty`
  diagnostic rather than silently choosing.
- Entropy is now over the legal set. `PpoLoss::entropy` will drop when masking turns
  on; that is correct, but do not compare across the change.
- The new column is the largest allocation added: 208 envs × 240 steps × 37 actions
  = 1.8M floats = 7.4 MB. Acceptable; a packed bitmask is 32× smaller and is the
  follow-up if a caller needs a wide action space.
- `tests/rl_collect_footprint.rs` asserts flat reserved bytes — it will need its
  expected figure updated for the new column, and only for that reason.

**Test.** New `tests/rl_masking.rs`:
1. a masked draw never returns a masked action, over 10,000 draws;
2. a masked replay reproduces the masked actor's log-probabilities to `1e-4`
   (invariant 1 under masking);
3. fused and unfused masked windows are byte-identical (invariant 2);
4. a policy trained on a task with half the actions masked reaches the same return
   as the same task with those actions absent from `action_dim` entirely.

**Verify.**
```bash
cargo test --release --no-default-features --features cpu --test rl_masking --test rl_fused --test rl_learn
```

**Done when** all four properties hold and the full suite is green.

---

## Kaizen round (R1, K1–K8)

Reviewed revision `ced3662`; implemented on branch `kaizen-k1-k8`. Plan:
`Kaggriculture/experiments/exp011_fused_ppo/MAMBA_TRAINER_KAIZEN_PLAN.md`.

**R1/K7 — the reference.** `ReferencePolicy::score` did not carry a cache at all:
a mixer only returns its end state when handed a starting one, the first call
passed `None`, so every window restarted the reference from zero. The old test
(`corrected != naive`) passed anyway. `reference_log_probs_from` now starts from
an explicit zero history and always returns the end cache.
Tests: `tests/rl_reference.rs::oracle` — an independent snapshot stepped through
`RolloutEngine`, resets taken from the environment's own done log, lanes ending on
different steps (on a window's last step, inside windows, never), three windows
with two multi-epoch updates between, SISO/MIMO, Euler/real without conv, and a
masked window checked against the environment's mask log; every cache component
checked non-zero. score vs oracle ≤ 1e-5 (measured ~1.2e-7). Mutations checked
locally, all caught: scoring from `batch.initial`; reset mask one step early; one
step late. Python: `test_resume.py::test_reset_starts_both_recurrent_histories_over`.

**K1 — atomic restore.** `Module::stage_state_dict` + `StagedWeights`;
`Optimizer::load_state_dict(params, state, steps, strict)` stages and *replaces*
(a strict restore used to keep moments the checkpoint lacked);
`Checkpoint::restore_training`. Learners restore into a new trainer and swap.
`load_checkpoint` returns what it restored, clears the stale PPO window, raises
`OSError`/`ValueError` consistently. Tests: `tests/train.rs::exact_restore`,
`bindings/python/tests/test_resume.py` (failed strict loads leave weights, moments,
counters and next LR identical for both learners, over six corruptions).

**K2 — configuration.** `metadata.trainer_config` (base LR, `lr_schedule`, AdamW
betas/eps/weight decay, `max_grad_norm`, PPO config or DAgger schedule and entropy
bonus, architecture without its init seed, reference weight fingerprint) and
`metadata.continuation`. `config="verify" | "checkpoint" | "live"`; legacy learner
checkpoints need `"live"`. `from_checkpoint` rebuilds a learner through its own
constructor. Tests: `test_resume.py` (step-decay resumes at 0.002; every
difference listed; each mode; legacy; reference mismatch; 10 + save/load + 10 vs 20
within 1e-5 — the CPU runtime itself varies ~1e-7 run to run).

**K3 — counters.** Every counter is a JSON integer in both encodings
(`Checkpoint::optimizer_steps`; binary format v2, v1 still read). Legacy `f32`
counters load; non-integral or negative ones are refused. Tests:
`tests/train.rs::exact_restore` (2^24+1, 2^53−1, u64::MAX through both formats;
update after a large-step restore identical to one that never stopped).

**K4 — optional masks.** `None` is all-legal per step in either order; mask column
allocated with ones, width changes refused, nothing allocated without masks.
`VecEnv::action_mask` returns `Result` and stops collection before the draw; a
failed window resets the collector. Masks must be 0/1 (device and host checks
share one message source). `ParallelEnvs` forwards masks. Imitation labels are
validated once at batch construction; the loss reads nothing back and survives
zero-weight illegal placeholders. `draw_action`'s fallback clamps to the last
action with nonzero probability. Tests: `tests/rl_masking.rs` (23),
`test_masking.py`.

**K5 — `Rollout` masks.** `action_mask=` on `step`/`evaluate`, checked on the host
before the state advances. Tests: `test_masking.py` (never illegal at T = 0, 1, 5;
log-probs equal the masked distribution's; all-ones identical to none; bad masks
raise without advancing; `Rollout` reproduces a learner's masked collection).

**K6 — fused masks.** `GameSpec::with_action_mask`, required `GameLogic::legal`,
`draw_action_with_mask`; built-in `Recall` masks `(cue+1) % symbols`. Tests:
`tests/rl_fused.rs::masked` (byte-identical masked windows incl. mask column),
`tests/rl_masked_footprint.rs` (zero reads; flat; 11 launches → 1 per step).

**WGSL: no infinity literals.** Found running the suites on wgpu: `mask_logits`, both
categorical entropy kernels (their `-inf` guards date from A1) and the masked draw
spelled `-inf`, which CubeCL emits as `f32(-inf)` — invalid WGSL. Nothing errors;
the kernels produce zeros, so on wgpu masking was ignored and PPO's entropy was
wrong. Masked logits are now `F::min_value()` and the guards test `p == 0`.
Evidence: `rl_masking::primitives` fail on wgpu before and pass after.

**V1/K8.** READMEs (root: masks, anchor, checkpoints; bindings: masks, schedules,
reference, save/resume table, boundary reads, wheel repair), examples
`bindings/python/examples/{ppo_anchored_masked,imitation_schedules}.py` (`--smoke`),
`examples/measure_reliability.rs`. Suites at the head of the branch, 2026-09-13,
Apple M1:

| | result |
|---|---|
| Rust, CPU | 252 passed, 0 failed across 19 result groups |
| Rust, wgpu<wgsl> (Metal, Apple M1) | RL, train, tensor, model, autograd, ssm suites all pass (incl. all masking, reference, fused and footprint tests); 21 failures in `distributions` (13), `distributions_bitexact` (4), `mixed_precision` (4) — see W1 |
| Python, CPU wheel | 150 passed, 1 skipped (the WGSL-only bf16 refusal test) |
| Python, wgpu<wgsl> wheel | 149 passed, 2 skipped (bf16 env-var cases WGSL cannot express; the WGSL bf16 refusal test runs and passes) |

Measurements (`cargo run --release --example measure_reliability`; measured, not
estimated):

Policy: `obs_dim=39`, 37 actions, `d_model=256`, 4 layers, 988,054 parameters;
208 lanes x 120 steps. Medians of 5 (checkpoints: of 3).

| | CPU runtime | wgpu<wgsl>, Apple M1 |
|---|---|---|
| `ReferencePolicy::score`, one window | 1.20 s | 268 ms |
| collect + batch, same window (for scale) | 2.24 s | 1.33 s |
| reference weights (snapshot) | 3.77 MiB of weights; pool reserve did not grow | same |
| reference cache carried between windows | 29.76 MiB | 29.76 MiB |
| reserved-bytes growth over 5 scored windows | 0 | 0 |
| mask column `[208, 120, 37]` | 3,694,080 bytes (3.52 MiB) | same |
| mask validation read, one window | 1.04 ms | 1.81 ms |
| checkpoint `.json` (weights + empty optimizer) | 12.54 MB; save 47.9 ms, load 26.2 ms | 12.54 MB; save 48.8 ms, load 28.8 ms |
| checkpoint `.m3ck` | 3.96 MB (3.17x smaller); save 8.7 ms, load 2.4 ms | 3.96 MB; save 7.8 ms, load 2.2 ms |

Checkpoint times are host serialisation and file I/O of the same bytes on both
rows; they exclude the device read that capturing weights costs.

The CPU wheel links Homebrew `libzstd` through the CPU runtime's code generator;
`maturin build --auditwheel=repair` vendors it (verified loading from
`mamba3_rl.dylibs/` in a fresh venv). The wgpu wheel links no such library.

---

## W1 — WGSL gaps outside the RL path (open)

Found by the wgpu run above; not introduced by this round and not fixed in it.

* `tests/distributions.rs`: 13 failures — Normal, Uniform, Exponential and
  multivariate normal kernels return zeros (`cdf(-2) = 0`, `variance = 0`). The
  univariate kernels assign `f32::INFINITY`/`NEG_INFINITY` constants
  (`src/distributions/univariate.rs`), the same WGSL defect; `reduce::max_dim` and
  `min_dim` seed with infinities too. Fix pattern: no infinity constants in kernel
  code — a finite sentinel where one suffices, or an infinity passed in as a runtime
  scalar where the value must be infinite.
* `tests/mixed_precision.rs`: 4 failures — bf16 kernels on WGSL, which has no bf16;
  those cases need to skip on WGSL as the Python suite now does.
* `tests/distributions_bitexact.rs`: 4 failures — Metal's libm differs from the
  host's (the suite's own probe says so), so device-vs-host bit-exactness cannot
  hold there; gate these on the probe.

---

## A6 — `GameLogic` and `collect_fused` are unreachable from Python

**Goal.** A Python caller can drive a compiled-in device game.

**Why, and why it is last.** Nothing in `bindings/python/src` mentions `GameLogic`,
`GameWorld` or `collect_fused`, so the 8-launches-to-1 path is Rust-only. Measured on
the driving project — 208 environments, `d_model=256` × 4 layers, 120 steps,
2 epochs, `wgpu<wgsl>`:

| section | seconds | share |
|---|---|---|
| environment (Python over a Rust engine) | 0.11 | **3.1%** |
| rollout — the policy stepping forward | 2.05 | 58.8% |
| update — 2 epochs forward and backward | 1.33 | 38.1% |
| one round | 3.49 | |

The three add to the whole, so there is no hidden host/device stall either. **Fusing
buys at most 3% at this model size**, which is why that project did not port its
ruleset to `GameLogic`. `fused.rs` earns its place where the policy is small (the
crate's own footprint test: 83 launches → 76, ~8%) or the environment is heavy host
work.

**Shape of the fix.** `GameWorld<G>` cannot be a general Python API — the transition
must be device code. What Python can be handed is a *named* game compiled into the
extension:
```python
env = mamba3_rl.game("recall", num_envs=64, symbols=4, horizon=8, seed=0)
```
plus a documented Rust-side procedure for adding one (implement `GameLogic`, add a
`#[pyclass]` wrapper, register in the `game()` factory). Do this only when a caller
has a game whose host cost justifies it.

---

## Performance

`PLAN.md` owns this. Two measurements from the driving project bear on it:

- **The RL loop's cost is the model.** 97% of a PPO round is the policy's forward
  and backward passes (table in A6). Work on `models::mamba3` speeds up
  reinforcement learning by nearly the same factor; work on the collection loop is
  chasing 3%.
- **`f16` bought nothing at this size.** 3.34 s a round against `f32`'s 3.36 s at
  `d_model=256`. Consistent with `PLAN.md`: halving operand bytes only pays at the
  bandwidth ceiling, and the real win needs tensor cores, which this backend path
  does not reach.
