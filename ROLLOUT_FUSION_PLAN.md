# Rollout fusion plan

Execution document, written to be worked start to finish without reading the
session that produced it. Each task is self-contained: goal, the measurement that
justifies it, exact files and line anchors, the code to write, the traps, the test,
and the command that proves it. Work the tasks **in the order given**. Each one
lands on its own and leaves the suite green.

The subject is the **rollout step** — `Mamba3Policy::step`, the thing a
reinforcement-learning collection loop runs once per environment per timestep. It
is 63 kernel launches and this plan removes 8 of them.

---

## 0. Before you start

```bash
cd /Users/ods/Documents/mamba-trainer

# Baseline. 185 tests, 17 suites, ~5 min. Must pass before and after every task.
cargo test --release --no-default-features --features cpu
```

### Check the exit code, not the output

`cargo test … | grep …` reports the **grep's** status, so a compile failure in one
test target looks like success. This has already caused a false "green" once in
this repo's history. Always:

```bash
cargo test --release --no-default-features --features cpu > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|^error|FAILED" /tmp/t.log
```

`exit=0` **and** `failed: 0` on all 17 suites, or the task is not done. Note the
suite includes 15 doc-tests; they are part of the 185.

### Never build and test at the same time

Do not run `cargo build --features wgpu` while `cargo test --features cpu` is
running. They share `target/`, the second build deletes the `.rlib` the first
run's doc-tests need, and you get 15 spurious doc-test failures that look like a
code defect. Run one at a time.

### Never benchmark while anything else runs

This workload is **host-CPU-bound**, not GPU-bound. A concurrent test suite
inflated a 130 ms measurement to 600 ms. Benchmark on an idle machine only.

---

## Repo map for these tasks

| path | what lives there |
|---|---|
| `src/models/mamba3.rs:326` | `Mamba3Mixer::project` — where B/C/dt biases are applied |
| `src/models/mamba3.rs:481` | `Mamba3Mixer::finish` — the output gate |
| `src/models/mamba3.rs:618` | `Mamba3Mixer::step_masked` — the rollout entry point |
| `src/tensor/ops/fused.rs:1400` | `fused::silu` / `:1405` `silu_backward` — **the template to copy** |
| `src/tensor/ops/fused.rs:1470` | `fused::softplus` / `:1475` `softplus_backward` |
| `src/tensor/ops/fused.rs:391` | `fused::rms_norm` / `:459` `rms_norm_backward` |
| `src/autograd/ops.rs:1071` | `Var::silu` / `:1081` `Var::silu_composed` — the wiring template |
| `src/autograd/ops.rs:1333` | `Var::softplus` / `:1343` `Var::softplus_composed` |
| `src/autograd/ops.rs:304` | `Var::rms_norm` |
| `src/autograd/ops.rs:26` | `reduce_grad_to` — sums a gradient back to a broadcast operand's shape |
| `src/autograd/ops.rs:103` | `Var::add` — shows how `reduce_grad_to` is used |
| `tests/autograd.rs:18` | `check_grad` — finite-difference gradient checker |
| `tests/autograd.rs:447` | `fused_silu_and_state_update_gradients` — the test template to copy |
| `src/backend.rs` | `start_launch_tally` / `launch_tally`, the attribution tool |
| `examples/profile_rollout.rs` | prints launches per source line for one rollout step |

---

## Invariants that must survive every task

1. **A fused op equals the composed form it replaces.** Every fusion in this crate
   ships with a `_composed` sibling and a test asserting they agree. No exceptions.
2. **Gradients stay correct.** Every new fused op gets a `check_grad` case. A
   forward-only fusion is not acceptable: the same code runs in training, where the
   rollout's `no_grad` guard is not in force.
3. **A replayed window reproduces the actor's log-probabilities.**
   `tests/rl_learn.rs` asserts every first-epoch PPO ratio is 1. Anything applied
   when acting must be applied identically when replaying — all three fusions here
   are inside `Mamba3Mixer`, which both paths share, so this holds automatically.
   It is listed because breaking it is silent.
4. **Defaults do not change behaviour.** Numerics must be unchanged to within the
   tolerances the tests state.

---

## The template — read this before writing any code

Every fusion in this crate has the same four parts. Copy them literally.

**(a) One kernel, forward and backward selected at comptime**
(`src/tensor/ops/fused.rs:1354`):

```rust
#[cube(launch_unchecked)]
fn silu_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    #[comptime] backward: bool,
    grad: &Array<Vector<F, N>>,
) {
    if ABSOLUTE_POS < output.len() {
        let x = input[ABSOLUTE_POS];
        let one = Vector::<F, N>::new(F::new(1.0_f32));
        let s = one / (one + (x * Vector::<F, N>::new(F::new(-1.0_f32))).exp());
        if comptime!(backward) {
            output[ABSOLUTE_POS] = grad[ABSOLUTE_POS] * (s + x * s * (one - s));
        } else {
            output[ABSOLUTE_POS] = x * s;
        }
    }
}
```

**(b) One launcher shared by both directions**, with the unused optional tensor
passed as a dummy alias of a real one (`grad.unwrap_or(input)`) — this is the
crate's idiom for an optional kernel argument and costs no allocation:

```rust
fn silu_launch<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    grad: Option<&Tensor<R, E>>,
) -> Tensor<R, E> {
    let out = Tensor::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let line = line_size_for::<R, E>(input.client(), n);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        silu_kernel::launch_unchecked::<E, R>(
            input.client(), count, dim, line,
            input.arg(), out.arg(),
            grad.is_some(), grad.unwrap_or(input).arg(),
        );
    }
    out
}
```

**(c) Two thin public functions** (`fused::silu`, `fused::silu_backward`).

**(d) A `Var` method plus a `_composed` reference** (`src/autograd/ops.rs:1071`):

```rust
/// `x * sigmoid(x)`.
pub fn silu(&self) -> Result<Self> {
    let value = fused::silu(&self.value);
    let x = self.value.clone();
    Ok(Self::record(value, &[self], || {
        rule!(|g| { Ok(vec![Some(fused::silu_backward(g, &x))]) })
    }))
}

/// `x * sigmoid(x)`, one primitive at a time. The reference [`Var::silu`] is
/// checked against.
pub fn silu_composed(&self) -> Result<Self> {
    self.mul(&self.sigmoid())
}
```

For an op with **two parents**, use `Self::record_with_mask` and return one
gradient slot per parent — `Var::add` at `src/autograd/ops.rs:103` is the example.

---

## What to expect, honestly

These three fusions remove **8 launches from a 63-launch policy step (13%)**.

Do not expect 13% off the clock. The last change in this area cut 16% of the
step's launches and bought roughly 5% of the time, because a fused kernel binds
more buffers per launch than the several small kernels it replaces, and binding is
a real part of the per-launch host cost. **A realistic expectation for the whole
plan is 4-8% of rollout time.**

Say so in the write-up. If the measurement comes out at the bottom of that range,
that is the result, not a failure to be explained away.

---

## Status

| | task | launches saved / step | difficulty | state |
|---|---|---|---|---|
| R0 | re-attribute before changing anything | — | 10 min | **done** |
| R1 | fuse the output gate (`silu` + `mul`) | 2 | easy | **done** |
| R2 | fuse the `dt` bias and softplus | 2 | medium | **done** |
| R3 | fuse the B/C bias into `rms_norm` | 4 | hard | **done** |
| R4 | measure, then write it down | — | 1 h | **done** |

R1 before R2 before R3: they are ordered by how much of the crate they touch, so
the pattern is proven on the cheapest one first. R3 changes a signature used
outside the mixer and is the only one that can break unrelated code.

---

## R0 — re-attribute before changing anything

**Goal.** Confirm the ranking below still holds on the current tree.

**Why.** Every change moves the bottleneck; the manual's rule is to let the new
profile pick the next lever rather than stack optimisations against a stale
assumption. The numbers in this plan were measured on wgpu at
`32 envs x 32 steps, d_model 64, 2 layers`.

**Steps.**

```bash
cargo build --release --features wgpu --example profile_rollout
./target/release/examples/profile_rollout
```

**Expected**, near enough:

```
policy step: 63 launches (63 per step)

launches    /step  site
       9      9.0  src/tensor/ops/elemwise.rs:501   <- broadcasting binary op
       8      8.0  src/tensor/ops/elemwise.rs:86    <- flat elementwise
       7      7.0  src/tensor/ops/fused.rs:204      <- per-row plane reduction (norms)
       4      4.0  src/tensor/ops/fused.rs:1383     <- silu
       4      4.0  src/tensor/ops/fused.rs:683      <- rotation
       4      4.0  src/tensor/ops/movement.rs:771   <- fused split
```

**Stop and report** if the step is not ~63 launches, or if a site not in that list
is above 6/step. Either means the tree has moved and this plan's ranking should be
re-derived before any of it is executed.

---

## R1 — fuse the output gate

**Goal.** `gated * silu(z)` in one launch instead of two.

**Why.** `Mamba3Mixer::finish` (`src/models/mamba3.rs:481`) ends every layer with

```rust
let gated = gated.mul(&z.silu()?)?;
```

which is a `silu` launch and a `mul` launch over `[batch, seq, d_inner]`. Two
layers, so 2 launches per rollout step. `silu` is already 4/step in the R0 table;
half of those are this one.

**Files.** `src/tensor/ops/fused.rs` (next to `silu`, around `:1400`),
`src/autograd/ops.rs` (next to `Var::silu`, around `:1081`),
`src/models/mamba3.rs:488`.

**API.**

```rust
// src/tensor/ops/fused.rs
/// `a * silu(b)`, the gate at the end of a Mamba-3 layer.
pub fn swiglu<R: Runtime, E: FloatElem>(
    a: &Tensor<R, E>, b: &Tensor<R, E>,
) -> Result<Tensor<R, E>>;

/// The adjoint of [`swiglu`]: `(da, db)`.
pub fn swiglu_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>, a: &Tensor<R, E>, b: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)>;

// src/autograd/ops.rs
/// `self * silu(other)`, fused.
pub fn swiglu(&self, other: &Self) -> Result<Self>;
/// `self * silu(other)`, one primitive at a time.
pub fn swiglu_composed(&self, other: &Self) -> Result<Self> {
    self.mul(&other.silu()?)
}
```

**The maths.** With `s = sigmoid(b)` and `silu(b) = b*s`:

- forward: `out = a * b * s`
- `da = g * b * s`
- `db = g * a * (s + b*s*(1-s))`

**Steps.**
1. Write `swiglu_kernel` on the `silu_kernel` template. It needs **two outputs**
   in the backward direction (`da` and `db`), so give the kernel two output arrays
   and, in the forward direction, write the result to the first and leave the
   second alone — pass `da` as the dummy for the unused slot exactly as
   `grad.unwrap_or(input)` does. Guard every write with `if comptime!(backward)`.
2. Require `a.shape() == b.shape()` and return `Error::shape` otherwise. **Do not**
   add broadcasting; the call site never needs it and broadcasting is what makes
   the backward need a reduction.
3. Wire `Var::swiglu` with `Self::record_with_mask`, two parents, following
   `Var::add` at `src/autograd/ops.rs:103`.
4. Change `src/models/mamba3.rs:488` from `gated.mul(&z.silu()?)?` to
   `gated.swiglu(z)?`.

**Traps.**
- `finish` is on the **windowed** path as well as the step path; `z` there is
  `[batch, seq, d_inner]` with `seq > 1`. The shapes still match, so nothing
  special is needed — but do not assume `seq == 1` anywhere in the kernel.
- The dummy-aliased second output must never be written in the forward direction,
  or the forward silently corrupts `a`'s buffer.

**Test.** Add to `tests/autograd.rs`, modelled on `fused_silu_and_state_update_gradients`
(`tests/autograd.rs:447`):

```rust
#[test]
fn fused_swiglu_matches_the_composed_form_and_differentiates() {
    let a = [0.5f32, -1.0, 2.0, 0.0, 1.5, -0.25];
    let b = [0.3f32, -0.7, 1.2, 2.0, -1.5, 0.05];
    let (va, vb) = (
        V::constant(Tensor::from_f32(&a, vec![2, 3], &dev()).unwrap()),
        V::constant(Tensor::from_f32(&b, vec![2, 3], &dev()).unwrap()),
    );
    let (fused, composed) = (
        va.swiglu(&vb).unwrap().to_f32(),
        va.swiglu_composed(&vb).unwrap().to_f32(),
    );
    for (f, c) in fused.iter().zip(&composed) {
        assert!((f - c).abs() < 1e-6, "swiglu fused={f} composed={c}");
    }
    // One `check_grad` per differentiated operand: the helper perturbs the single
    // traced input it is given, so the other operand is held constant here.
    check_grad("swiglu/a", &a, vec![2, 3], |v| v.swiglu(&vb).unwrap().sum().unwrap());
    check_grad("swiglu/b", &b, vec![2, 3], |v| va.swiglu(v).unwrap().sum().unwrap());
}
```

**Verify.**
```bash
cargo test --release --no-default-features --features cpu --test autograd --test model > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|FAILED" /tmp/t.log
./target/release/examples/profile_rollout | head -6   # after a wgpu rebuild
```

**Done when** the composed forms agree, both `check_grad` cases pass, the full
suite is green, and the policy step reports **61** launches.

---

## R2 — fuse the `dt` bias and softplus

**Goal.** `softplus(dt_raw + dt_bias)` in one launch instead of two.

**Why.** `src/models/mamba3.rs:446`:

```rust
let dt = dt_raw
    .add(&self.dt_bias.var(input).reshape(vec![1, 1, heads])?)?
    .softplus()?;
```

That is a **broadcasting** add — the expensive kind, because it runs the broadcast
kernel rather than the flat one — followed by a softplus. Two launches per layer,
2 per rollout step.

**Files.** `src/tensor/ops/fused.rs` (next to `softplus`, `:1470`),
`src/autograd/ops.rs` (next to `Var::softplus`, `:1333`), `src/models/mamba3.rs:446`.

**API.**

```rust
// src/tensor/ops/fused.rs
/// `softplus(x + bias)`, where `bias` holds one value per element of `x`'s
/// trailing axis.
pub fn bias_softplus<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>, bias: &Tensor<R, E>,
) -> Result<Tensor<R, E>>;

/// The adjoint of [`bias_softplus`], with respect to `x` only.
///
/// `d/dbias` is the same values summed over every axis but the last, which the
/// caller does with `reduce_grad_to` rather than a second kernel.
pub fn bias_softplus_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>, x: &Tensor<R, E>, bias: &Tensor<R, E>,
) -> Result<Tensor<R, E>>;

// src/autograd/ops.rs
pub fn bias_softplus(&self, bias: &Self) -> Result<Self>;
pub fn bias_softplus_composed(&self, bias: &Self) -> Result<Self> {
    self.add(bias)?.softplus()
}
```

**The maths.** `out = softplus(x + b)`, `d/dx = g * sigmoid(x + b)`, and
`d/db = sum over all axes but the last of d/dx`.

**Steps.**
1. Kernel: index `bias` by `ABSOLUTE_POS % bias_len` so one value per trailing
   element is broadcast over the rows. Pass `bias_len` as a scalar. Because the
   bias is indexed with a modulo rather than through the broadcast-metadata path,
   **this kernel uploads no metadata at all**, which is a second saving on top of
   the launch.
2. Restrict to the case the call site needs: `bias.rank() == 1` and
   `bias.len() == x.shape().dim_from_end(0)`. Return `Error::shape` otherwise, and
   say what was expected. Do not generalise.
3. `Var::bias_softplus`: two parents, `record_with_mask`. The `x` gradient is the
   kernel's output; the `bias` gradient is
   `reduce_grad_to(&dx, bias.shape())?` — `reduce_grad_to` is at
   `src/autograd/ops.rs:26` and `Var::add` at `:103` shows the idiom.
4. At `src/models/mamba3.rs:446`, the bias is currently reshaped to
   `[1, 1, heads]`. The new op wants it rank 1, so pass
   `self.dt_bias.var(input)` **without the reshape**:
   ```rust
   let dt = dt_raw.bias_softplus(&self.dt_bias.var(input))?;
   ```

**Traps.**
- **The bias gradient is the one that will be wrong.** `dt_bias` is `[heads]` and
  the activation is `[batch, seq, heads]`, so the gradient must be summed over
  `batch` and `seq`. Forgetting the sum produces a shape error if you are lucky and
  a wrong gradient if `batch == seq == 1` in your test. **Test with `batch > 1`
  and `seq > 1`.**
- `dt_bias` is initialised so that `softplus(dt_bias)` is log-uniform
  (`src/models/mamba3.rs:190`). Do not touch that initialisation; this task changes
  only how the value is consumed.
- Keep softplus numerically stable in the kernel — copy the existing
  `softplus_kernel` body (`src/tensor/ops/fused.rs:1470`), which uses
  `max(x,0) + ln(1 + exp(-|x|))`. A naive `ln(1+exp(x))` overflows at x=40 and
  there is an existing test at `tests/autograd.rs:454` that feeds it exactly that.

**Test.** In `tests/autograd.rs`:

```rust
#[test]
fn fused_bias_softplus_matches_the_composed_form_and_differentiates() {
    // batch 2, seq 3, heads 4 -- both leading axes > 1, so a missing reduction in
    // the bias gradient cannot hide.
    let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.37 - 4.0).collect();
    let bias = [0.25f32, -1.0, 40.0, -40.0];   // includes the overflow extremes
    let vb = V::constant(Tensor::from_f32(&bias, vec![4], &dev()).unwrap());
    let vx = V::constant(Tensor::from_f32(&x, vec![2, 3, 4], &dev()).unwrap());

    let (fused, composed) = (
        vx.bias_softplus(&vb).unwrap().to_f32(),
        vx.bias_softplus_composed(&vb).unwrap().to_f32(),
    );
    for (f, c) in fused.iter().zip(&composed) {
        assert!((f - c).abs() < 1e-4, "fused={f} composed={c}");
    }
    assert!(fused.iter().all(|v| v.is_finite()), "overflowed: {fused:?}");

    check_grad("bias_softplus/x", &x, vec![2, 3, 4], |v| {
        v.bias_softplus(&vb).unwrap().sum().unwrap()
    });
    check_grad("bias_softplus/bias", &bias[..2], vec![2], |v| {
        // A [2, 3, 2] activation against a [2] bias: the reduction is over 6 values
        // per bias element, so a dropped sum is off by 6x, not by rounding.
        let x = V::constant(Tensor::from_f32(&vec![0.1f32; 12], vec![2, 3, 2], &dev()).unwrap());
        x.bias_softplus(v).unwrap().sum().unwrap()
    });
}
```

**Verify.**
```bash
cargo test --release --no-default-features --features cpu --test autograd --test model --test ssm > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|FAILED" /tmp/t.log
```

**Done when** both `check_grad` cases pass, the full suite is green, and the policy
step reports **59** launches.

---

## R3 — fuse the B/C bias into `rms_norm`

**Goal.** `rms_norm(x + bias)` in one launch instead of two, for both B and C.

**Why.** `src/models/mamba3.rs:429`:

```rust
if let Some(bias) = &self.b_bias {
    b = b.add(&bias.var(input).reshape(vec![1, 1, heads, 1, state])?)?;
}
if let Some(bias) = &self.c_bias {
    c = c.add(&bias.var(input).reshape(vec![1, 1, heads, 1, state])?)?;
}
if let Some(norm) = &self.bc_norm {
    b = norm.apply(&b)?;
    c = norm.apply(&c)?;
}
```

Two broadcasting adds feeding two normalisations. Fusing the add into the norm
removes 2 launches per layer, **4 per rollout step — the largest single saving in
this plan**.

**Do this one last.** It changes `fused::rms_norm`, which has callers outside the
mixer.

**Files.** `src/tensor/ops/fused.rs:391` (`rms_norm`) and `:459`
(`rms_norm_backward`), `src/autograd/ops.rs:304` (`Var::rms_norm`),
`src/nn/norm.rs:95` (`RmsNorm::apply`), `src/models/mamba3.rs:429`.

**API.** Add the bias as an `Option`, so every existing caller passes `None` and is
unchanged:

```rust
// src/tensor/ops/fused.rs — extend the existing signatures.
pub fn rms_norm<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    bias: Option<&Tensor<R, E>>,     // new: added to `input` before the norm
    weight: Option<&Tensor<R, E>>,
    eps: f32,
) -> Result<(Tensor<R, E>, Tensor<R, E>)>;

pub fn rms_norm_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
    bias: Option<&Tensor<R, E>>,     // new
    weight: Option<&Tensor<R, E>>,
    scale: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Option<Tensor<R, E>>)>;

// src/autograd/ops.rs
pub fn rms_norm_biased(&self, bias: Option<&Self>, weight: Option<&Self>, eps: f32)
    -> Result<Self>;
```

Keep `Var::rms_norm(weight, eps)` as it is and have it delegate to
`rms_norm_biased(None, weight, eps)`, so nothing else in the crate has to change.

**Steps.**
1. Extend the kernel to add `bias[i % dim]` to each loaded element **before** the
   sum of squares and before the scaling. Both passes over the row must see the
   biased value, or the normaliser and the numerator disagree.
2. Extend `rms_norm_backward` the same way. `d/dinput` and `d/dbias` are the same
   tensor (the bias enters additively), so return `dx` and let the `Var` layer
   produce the bias gradient with `reduce_grad_to(&dx, bias.shape())?`.
3. Thread the option through `Var::rms_norm_biased` with three parents
   (`input`, `bias`, `weight`) — build the parent vector conditionally exactly as
   `Var::rms_norm` already does for `weight` at `src/autograd/ops.rs:308`.
4. Give `RmsNorm` a method that takes the pre-bias, e.g.
   `RmsNorm::apply_biased(&self, input, bias: Option<&Var<R, E>>)`, and leave
   `apply` delegating to it with `None`.
5. Rewrite `src/models/mamba3.rs:429-439` so the bias and the norm are one call.
   **Preserve the existing behaviour in all four combinations**: bias with norm,
   bias without norm, norm without bias, neither. When there is a bias but no
   `bc_norm`, there is nothing to fuse into — keep the plain `add`.

**Traps.**
- **Order matters and is easy to get subtly wrong.** Today the bias is added, then
  the norm computes its scale from the *biased* values. The fused kernel must do
  the same. Computing the scale from the unbiased values passes a shape check and
  fails only numerically.
- `b_bias`/`c_bias` and `bc_norm` are **independently optional**
  (`src/models/mamba3.rs:240-249`). All four combinations must keep working. The
  existing tests only cover some of them; add coverage for the rest.
- **The bias index is not `% dim`.** The parameter is `[heads, state]`
  (`src/models/mamba3.rs:242`) and the activation at that point is
  `[batch, seq, heads, rank, state]`, reshaped to `[1, 1, heads, 1, state]` to
  broadcast. The norm's trailing axis is `state`, but the bias also varies with
  `heads`, so for a flat position `p`:

  ```text
  s = p % state
  h = (p / (state * rank)) % heads
  bias index = h * state + s
  ```

  Pass `state`, `rank` and `heads` as scalars. Getting this wrong by using
  `p % (heads * state)` gives the right answer only when `rank == 1`, which is the
  default configuration in several tests — so it will look correct until it is not.
  Test with `heads > 1`, `state > 1`, **`rank > 1`**, and a different bias value per
  head.
- `RmsNorm::apply` is at `src/nn/norm.rs:95`. There is a second, unrelated `apply`
  at `src/nn/norm.rs:191` on a different type; do not edit that one.
- `rms_norm` has callers outside the mixer. After changing the signature:
  `cargo test --no-default-features --features cpu --no-run` catches the test
  targets that a `cargo check` on the library alone will not.

**Test.** Extend `tests/model.rs` with a mixer-level case covering all four
bias/norm combinations, and add a `check_grad` for the bias in `tests/autograd.rs`
with `heads = 2`, `state = 4`, and a distinct bias value per head.

**Verify.**
```bash
cargo test --release --no-default-features --features cpu --no-run > /tmp/c.log 2>&1; echo "compile exit=$?"
cargo test --release --no-default-features --features cpu > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|FAILED" /tmp/t.log
```

**Done when** all four combinations are tested and green, the full suite is green,
and the policy step reports **55** launches.

---

## R4 — measure, then write it down

**Goal.** A number that can be trusted, and a record of it.

**Why.** Launch count is deterministic; wall time on this machine is not. Run-to-run
noise is ±20%, which is several times the effect these three tasks produce
together. Comparing two separate runs proves nothing.

**The protocol.** Interleave both paths **inside one process** and take the
minimum, the way `examples/bench_split.rs` and `examples/bench_meta_upload.rs`
already do. That means each fusion needs a runtime toggle, matching the existing
`movement::set_fused_split` / `backend::set_meta_cache`:

```rust
pub fn set_fused_gate(on: bool);   // R1
pub fn set_fused_dt(on: bool);     // R2
pub fn set_fused_bc(on: bool);     // R3
```

Default on; off restores the composed path. This is also what lets a bisect find
which fusion broke something.

**Steps.**
1. On an **idle** machine (no test suite, no build):
   ```bash
   cargo build --release --features wgpu --example profile_ppo --example profile_rollout
   for i in $(seq 1 10); do
     MAMBA3_FUSED_GATE=0 MAMBA3_FUSED_DT=0 MAMBA3_FUSED_BC=0 \
       ./target/release/examples/profile_ppo 2>/dev/null | awk '/host submission/{print "base",$5}'
     ./target/release/examples/profile_ppo 2>/dev/null | awk '/host submission/{print "opt",$5}'
   done
   ```
2. Report **min and median** of each, and the percentage change of both. Use the
   **host submission** line, not the round total: submission is ~98% of the round
   on this backend and is the quantity being optimised; the round total also
   carries a noisy device drain.
3. Record the result in `PLAN.md` beside the D.2 and D.3 entries, in the same
   before/after voice. Include the launch counts (deterministic) *and* the timing
   (noisy, with its spread).

**Done when** `PLAN.md` states the measured effect of all three fusions together,
including the spread, and says plainly if it came in at the low end.

---

## What is deliberately not in this plan

- **Fusing the whole Mamba-3 step into one kernel.** This is the change that would
  actually transform rollout cost — the step is 55 launches after R3 and could in
  principle be a handful. It is also a large, risky piece of work with a
  hand-written backward, and it should not be started until the cheap fusions above
  are done and measured, because they are what tell you what is left.
- **`src/rl/fused.rs`** (the collect-level fused rollout). It fuses the environment
  step, the action draw and the buffer write — 83 launches to 76 on the crate's own
  footprint test. It is a different layer from this plan and is measured to be worth
  ~3% at production model sizes, where 97% of a round is the policy's forward and
  backward.
- **Anything in the backward pass.** The rollout runs under `no_grad`; backward
  work does not affect it. The fusions here improve training too, but that is a
  side effect and not what they are being judged on.
