# opencode manual for mamba-trainer

How this repository drives the [opencode](https://opencode.ai) CLI (v2.0.10, installed at `~/.bun/bin/opencode`)
as a coding agent. A supervisor (a human, or Claude Code) splits a plan into tasks, and opencode implements each one.
The supervisor then checks the result independently before starting the next task.

Everything here was learned implementing `ENTITY_KERNEL_PLAN.md`, including the failures.

---

## 1. Quick start

```bash
cd /Users/ods/Documents/mamba-trainer

# Check the model answers before handing it real work.
opencode run --standalone -m opencode-go/muse-spark-1.3-contributor "Reply with the single word OK."

# Run one task non-interactively, log everything, and keep the exit code.
opencode run --standalone --auto \
  -m opencode-go/muse-spark-1.3-contributor \
  --title "K3 pointer_scores kernel" \
  "$(cat prompts/k3.md)" > logs/k3.log 2>&1; echo "exit=$?"
```

| Flag | What it does | Why we use it |
|---|---|---|
| `--standalone` | Runs a private server for this invocation instead of the shared background service | The shared service can wedge (see §5.2). Always pass it for unattended runs. |
| `--auto` | Auto-approves tool permissions that are not explicitly denied | A non-interactive run cannot answer permission prompts. Without it, the agent stalls at its first edit. |
| `-m provider/model` | Chooses the model | See §2 |
| `--title "..."` | Names the session | Makes `opencode session list` readable when you need to resume |
| `-s <session-id>` | Continues an existing session | Resuming after an interruption (§4). Use with care (§5.2). |
| `-c` | Continues the most recent session | Convenient interactively; ambiguous in scripts, so prefer `-s` |
| `--format json` | Emits machine-readable events | For tooling; plain text is easier to review by eye |

The exit code of `opencode run` is **0 even when the model call failed** (for example, "Rate limit exceeded").
Always read the end of the log too (§5.1).

---

## 2. Models and providers

```bash
opencode auth list          # which providers have credentials
opencode models             # every model id, as provider/model
opencode models | grep -i muse
```

Providers configured on this machine:

| Provider | Credential | Notes |
|---|---|---|
| `opencode/…` | OpenCode API key | Includes the free tier (`…-free` models), which is shared and rate-limited |
| `opencode-go/…` | OpenCode Go plan API key | Paid plan. **Use this for real work** |
| `google/…` | `GEMINI_API_KEY` from the environment | Gemini models |

The model this repo uses by default is **`opencode-go/muse-spark-1.3-contributor`**. The same model on the free tier is
`opencode/muse-spark-1.3-contributor-free`. It worked for the first two tasks, then refused every request with
"Rate limit exceeded" for more than two hours (§5.1), and so did most other `…-free` models.

Model ids are exact. `muse-spark-1.3-contributor-free` (provider `opencode`) and `muse-spark-1.3-contributor`
(provider `opencode-go`) are different quotas.

---

## 3. Writing a task prompt

opencode starts with no memory of the conversation that produced the plan. A good prompt makes the task
self-contained, and it is what made the verified tasks (K0, K1) succeed on the first attempt.

1. **Scope in one line.** "Implement ONLY task K2. Do not start K3 or later."
2. **Where to read.** The plan file, plus the exact files and line anchors of the template to copy (for example,
   `src/tensor/ops/fused.rs` `ssm_coefficients_kernel`, ~line 2195).
3. **The spec.** Inputs and outputs with shapes, the thread layout, and the semantics that must not change (for example,
   presence `0.5` behaves exactly as in the composed path).
4. **Repo conventions it cannot guess.** For example:
   - every launch goes through `launch_1d`, because it counts launches;
   - `Var::record` is `pub(crate)`;
   - the `rule!` macro is private to `src/autograd/ops.rs`, so tape wrappers live there;
   - the CPU runtime also runs CubeCL kernels.
5. **Tests to write** and **tests that must keep passing**, in both modes of any switch.
6. **Measurements to report** (for example, `cargo run --release --no-default-features --features cpu --example
   profile_entity`), with the previous numbers so it can compare.
7. **Hard rules:**
   - do NOT commit, push, or touch git state;
   - do not delete files, and do not run `cargo clean`;
   - never run two cargo commands at once;
   - do not build `--features wgpu` (the supervisor runs GPU tests after the task);
   - the disk is nearly full.
8. **Verification with exit codes**: the exact commands, ending in `; echo "exit=$?"`, because `cargo test | grep`
   reports grep's status and hides failures.
9. **Required final summary**: files changed, commands with exit codes, and measured numbers before and after.

Keep prompts in files (for example, the session scratchpad) and pass them with `"$(cat file.md)"`, so a re-run
uses the identical text.

---

## 4. The supervise-and-verify loop

opencode's own summary is a claim, not evidence. After each task:

1. **Read the diff**: `git status --short`, `git diff --stat`, then the new kernel and wiring themselves. Look for
   over-claims in comments. K1 claimed its sum matched `reduce::sum_dim` "bit for bit", which it does not guarantee.
2. **Re-run the tests yourself, with exit codes**, on CPU:
   ```bash
   cargo test --release --no-default-features --features cpu --no-fail-fast \
     --test rl_entity --test rl_entity_parity --test rl_entity_footprint --test entity_kernels \
     --test rl --test rl_fused > /tmp/verify.log 2>&1; echo "exit=$?"
   grep -E "^test result|FAILED|^warning" /tmp/verify.log
   ```
3. **Run the GPU (wgpu) tests** that opencode was told not to run. Only do this after the CPU run has finished, because
   they share `target/`:
   ```bash
   cargo test --release --no-default-features --features wgpu --no-fail-fast \
     --test rl_entity --test rl_entity_parity --test rl_entity_footprint --test entity_kernels --test rl_fused
   ```
4. **Check the numbers** it reported (launch counts are deterministic; re-run the profiler).
5. Only then write the next task's prompt, carrying forward anything learned.

Resuming an interrupted task:

```bash
opencode session list                                    # newest first: id, title, time
opencode run --standalone --auto -m opencode-go/muse-spark-1.3-contributor \
  -s ses_XXXXXXXX "You were interrupted. Continue task K2 where you left off; first run git diff --stat."
```

If resuming hangs (§5.2), start a **fresh** session instead. Prepend a note to the original prompt listing what is
already written ("kernels X and Y exist in file Z; review them, do not rewrite them"), and say what is left to do.

---

## 5. Failure modes seen, and what to do

### 5.1 "Rate limit exceeded" (free tier)

```
> build · muse-spark-1.3-contributor-free
Error: Rate limit exceeded. Please try again later.
```

- It arrives mid-task, leaving half-written code, and the process still exits 0.
- **Detect it:** `tail -c 400 log | grep -q "Rate limit exceeded"`.
- **Distinguish a quota from a transient error:** send the one-word "OK" prompt. If even that is refused, the model's
  quota is exhausted, and retrying the task is pointless.
- **What happened here:** twelve retries at 10-minute intervals (2 hours) all failed on
  `opencode/muse-spark-1.3-contributor-free`. Of the other free models, only `opencode/nemotron-3-ultra-free`
  answered.
- **Fix:** switch to the paid provider (`opencode-go/…`). Model changes are the user's decision (cost), so ask first.

### 5.2 A run that produces no output and no file changes

- **Symptom:** `opencode run` is alive at ~0% CPU. The log stays empty for many minutes, and no source file's
  modification time changes.
- **Cause here:** a resumed session (`-s`) was killed while the shared background service
  (`opencode serve --service`, which had been up for 21 hours) still considered it busy. Later runs through that
  service, even fresh ones, then hung.
- **Fix:** use `--standalone`, which gives each invocation its own server. `opencode service restart` would also reset
  the shared one, but it is a user-wide service, so don't restart it without asking.
- **Check for activity:**
  ```bash
  ps -eo pid,etime,pcpu,command | grep "[o]pencode"
  stat -f "%Sm %N" src/... tests/...          # have the files it should be editing changed?
  ```

### 5.3 Process-global switches and parallel tests

opencode will write tests that flip a process-global switch (for example, `set_fused_entity`), and cargo runs tests in
parallel threads. Tell it explicitly: put a comparison of both modes in one `#[test]` function, or give it its own
test binary. Footprint tests that read `launch_count()` / `read_count()` must be alone in their binary for the same
reason.

### 5.4 Disk space

A release build with examples is ~16 GB in `target/`. When the disk fills, the failure looks like a link error
(`ld: write() failed, errno=28`). Check `df -h .` before long runs, and tell opencode not to create extra target
directories or run `cargo clean`.

---

## 6. Housekeeping

```bash
opencode session list                 # sessions in this project
opencode session export <id> > s.json # keep a transcript
opencode session delete <id>          # remove one (and its children)
opencode service status               # the shared background server
opencode stats                        # usage statistics
```

opencode never commits in this workflow. The supervisor reviews, verifies, and commits.
