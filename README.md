# mamba3

Mamba-3 state space models in Rust, running on [CubeCL](https://github.com/tracel-ai/cubecl)
— so the same code targets CPU, wgpu (Vulkan/Metal/DX12), CUDA and ROCm.

The crate is a **training stack**, not just a forward pass: it ships its own
tensor layer, reverse-mode autodiff, module system, optimizers and inference
caches, and it is built so that vision models, Transformer hybrids,
quantization-aware training, LoRA and streaming inference are configuration
choices rather than forks.

```rust
use mamba3::prelude::*;

type R = mamba3::backends::Cpu;

let device = Device::<R>::default();
let model = Mamba3LmConfig::builder()
    .vocab_size(32_000)
    .d_model(768)
    .n_layers(24)
    .pattern(LayerPattern::AttentionEvery { period: 8 })   // hybrid, if you want one
    .build()?
    .init::<R, f32>(&device)?;
```

---

## What Mamba-3 actually changes

Mamba-3 keeps Mamba-2's selective state space layer and improves it along three
axes. All three are implemented here, and each is an enum on `SsmConfig` so you
can ablate them one at a time.

### 1. Exponential-trapezoidal discretization

Mamba-2 discretizes the continuous system with a first-order (Euler / zero-order
hold) rule, which only samples the right endpoint of each interval. Mamba-3 uses a
generalised trapezoidal rule — a second-order approximation that takes a
data-dependent convex combination of both endpoints:

```text
h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t
a_t = exp(dt_t A)      b_t = (1 - l_t) dt_t a_t      g_t = l_t dt_t
```

`l_t = 1` recovers Mamba-2 exactly; `l_t = 1/2` is the classical trapezoid; the
default lets the model predict `l_t` per token and per head.

**The implementation trick.** Written naively this is a two-input recurrence and
does not fit the semiseparable-matrix form the fast scan relies on. Regrouping the
unrolled sum by *input* index rather than update index collapses it back:

```text
y_t = sum_{j<=t} decay(j->t) * ( g_j + [j<t] * f_j ) * (C_t . B_j) * x_j
      where f_j = (1 - l_{j+1}) dt_{j+1}
```

because `b_{j+1} · decay(j+1→t) = (1-l_{j+1}) dt_{j+1} · decay(j→t)`. So the
second-order rule is the *same* structured matrix as Mamba-2, with a per-source
weight `w_j = g_j + f_j` below the diagonal and `g_j` on it — one extra shifted
add, not a new kernel. The derivation is written out in
[`src/ssm/scan.rs`](src/ssm/scan.rs) and checked against a directly unrolled
recurrence in [`tests/ssm.rs`](tests/ssm.rs).

### 2. Complex (rotational) state updates

A real, positive decay can only forget. Mamba-3 pairs up state dimensions and
rotates each plane by a data-dependent angle `dt·theta`, which is exactly a complex
eigenvalue `exp(dt(A + i·theta))` — enough to represent periodic structure and
recover state-tracking tasks that real linear recurrences cannot express.

Implemented as a RoPE-style rotation of `B` and `C` by their cumulative angle
(`StateDynamics::Rotational`), which is equivalent to rotating the transition and
keeps the whole thing inside the fast scan. Cumulative angles are wrapped into
`[-pi, pi]` with a detached correction, so precision does not decay over long
sequences and the gradient stays exact.

### 3. MIMO

The SISO layer moves very little data per FLOP at decode time. The MIMO variant
gives each head `R` input and output channels, turning the rank-1 outer-product
state update into a rank-`R` matrix product: more arithmetic per byte of state,
same state size. `SsmMode::Mimo { rank }` enables it.

The reference path here decomposes MIMO into the `R²` SISO systems the paper
describes (states add, each output reads the shared state with its own `C`). That
is obviously correct and obviously not optimal; a rank-aware kernel would fold the
`R²` matmuls into one.

### Architecture details

`bc_norm` adds RMS normalisation after the `B`/`C` projections (the paper's
analogue of QK-norm) and `bc_bias` adds head-specific channel-wise biases. With
those on, the post-gate RMS norm is unnecessary and off by default, and the short
causal convolution becomes optional (`conv_kernel: None`).

---

## Layout

| layer | module | responsibility |
|---|---|---|
| 1 | [`tensor`](src/tensor) | contiguous device buffers, ~20 CubeCL kernels |
| 2 | [`autograd`](src/autograd) | tape-based reverse-mode differentiation |
| 3 | [`nn`](src/nn) | parameters, modules, initializers, LoRA, quantization |
| 4 | [`ssm`](src/ssm), [`models`](src/models) | the Mamba-3 scan and the model zoo |
| 5 | [`train`](src/train), [`infer`](src/infer), [`rl`](src/rl) | optimizers, schedules, trainer, caches, rollout engines, PPO, imitation learning |
| 5 | [`distributions`](src/distributions) | `torch.distributions` as kernels: sampling, scoring, gradients, divergences |

Three decisions shape everything above the kernels.

**Tensors are always contiguous.** `permute`, `slice` and friends materialise. In
exchange, every kernel is a flat `Array<E>` kernel, so porting the backend surface
to a new runtime is a small job and there is no stride logic to get wrong.

**Composition over fusion.** Only about two dozen primitives carry a hand-written
adjoint. Softmax, RMS norm, SiLU, GELU, RoPE, the causal convolution, fake
quantization and the entire SSD scan are *composed* from them, so their gradients
are correct by construction. Fusing any of them later is a performance change, not
a correctness one — which is the property you want when the interesting part of the
model is a recurrence whose backward pass is easy to derive wrongly.

**Parameters are shared handles.** A `Param` is a reference-counted, interior-mutable
cell. That is why the optimizer works on a flat `Vec<Param>` with no mutable
traversal of the model tree, why LoRA can freeze a base weight by flipping one
flag, and why merging an adapter takes effect immediately everywhere.

---

## Builders

Every layer and model is described by a `Config` that is plain data, validates
itself, and has a builder:

```rust
let ssm = SsmConfig::builder()
    .d_model(768)
    .n_heads(12)
    .head_dim(64)
    .d_state(128)
    .discretization(Discretization::LearnedTrapezoid)
    .dynamics(StateDynamics::Rotational)
    .mode(SsmMode::Mimo { rank: 4 })
    .chunk_size(64)
    .build()?;                       // validates; returns Err on bad shapes

let linear = LinearConfig::new(512, 2048)
    .with_bias(false)
    .with_lora(LoraConfig::builder().rank(16).alpha(32.0).build()?)
    .with_weight_quant(QuantConfig::int8_weights())
    .init::<R, f32>(&device, &mut rng);
```

`Config::init(&device, &mut rng)` is the only place randomness enters, so a run is
reproducible from a seed regardless of backend.

---

## The seven extension points

### Vision

[`models::vision`](src/models/vision.rs) reuses the same mixer with a patch
embedding and bidirectional scanning — images have no causal order, so each block
runs a forward and a backward mixer and adds them.

```rust
let model = VisionMamba3Config::builder()
    .image_size(224).patch_size(16).num_classes(1000)
    .d_model(384).n_layers(24)
    .direction(ScanDirection::Bidirectional)
    .pooling(Pooling::Mean)
    .build()?
    .init::<R, f32>(&device)?;
```

### Transformer hybrids

[`models::hybrid`](src/models/hybrid.rs) interleaves Mamba-3 mixers with
grouped-query attention through a declarative `LayerPattern`
(`AllMamba`, `AttentionEvery { period }`, `AttentionAt(indices)`, `Explicit(..)`).
Both mixer kinds are ordinary modules over the same `Linear`, so a hybrid inherits
LoRA, quantization and caching with no extra plumbing.

### Quantization-aware training

[`nn::quant`](src/nn/quant.rs) implements fake quantization with a straight-through
estimator and a clamp mask. Weights use dynamic per-tensor or per-channel ranges;
activations use an observed exponential moving average that freezes at eval time.
Because it is expressed with ordinary differentiable ops it needs no kernel and no
custom backward.

```rust
let model = Mamba3LmConfig::builder()
    .vocab_size(32_000).d_model(768).n_layers(24)
    .weight_quant(QuantConfig::int8_weights())
    .activation_quant(QuantConfig::int8_activations())
    .build()?.init::<R, f32>(&device)?;
```

### LoRA

[`nn::lora`](src/nn/lora.rs) adds `x @ A @ B * (alpha/r)` to any `Linear`. `B`
starts at zero so the adapter is a no-op at step 0; `merge_lora_adapters()` folds
every adapter in a model back into its base weights in one traversal.

```rust
model.freeze_matching(&[]);            // freeze everything
model.unfreeze_matching(&["lora"]);    // except the adapters
let task = LmTask::new(&model).only(&["lora"]);
```

### Inference

[`infer`](src/infer) carries a per-layer `StateCache`. The prompt is consumed in
one chunked parallel pass; each new token then costs one recurrent step, and in a
pure Mamba-3 stack nothing about that cost grows with context length. The test
suite asserts that stepwise decoding reproduces the parallel forward pass
bit-for-bit-close for every combination of discretization, dynamics and mode.

```rust
let mut generator = Generator::new(&model, GeneratorConfig::builder()
    .max_new_tokens(128)
    .sampler(SamplerConfig::temperature(0.8).with_top_k(40))
    .build());
let tokens = generator.generate(&prompt, &device)?;
```

### Reinforcement and imitation learning

[`rl`](src/rl) is the inference path turned into a rollout engine, plus everything
that makes a rollout trainable. Acting wants the smallest possible step — one
observation, `B` environments, latency that does not grow with the episode.
Learning wants the largest possible batch — a whole `[B, T]` trajectory through
one parallel pass. A state space model can do both from the same weights, and the
module is that pair with PPO and behaviour cloning built on top of it:

```rust
let policy = Mamba3PolicyConfig::new(obs_dim, actions, 256, 4).init::<R, f32>(&device)?;
let mut collector = Collector::new(&policy, 64, 128, obs_dim, &device)?;
let task = PpoTask::new(&policy, config);

loop {
    // Acting: 128 steps of 64 environments. One allocation, zero host reads.
    let report = collector.collect(&mut env)?;
    let batch = collector.ppo_batch(&report, &config)?;

    // Learning: the same weights over the window it just collected.
    for _ in 0..epochs {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }
}
```

**Episode boundaries are the interesting part.** A reset at position `t` has to
make everything before `t` invisible to `t` and everything after it, while letting
the observation *at* `t` seed the new episode. Two paths reach backwards in a
Mamba-3 layer and both are cut:

* the recurrence — the rollout step zeroes the two coefficients that carry the
  previous state, inside the kernel that already computes them; the scan floors
  the log-decay, which severs the intra-chunk band, the inter-chunk transfer and
  the carry-in in one stroke, because every one of them is weighted by a product
  containing `a_t`;
* the short causal convolution — taps that reach across a boundary are dropped,
  and the history carried between windows comes back with the cut positions
  zeroed.

Neither costs a launch of its own: resetting all 64 environments on every step
measures the same latency and the same dispatch count as resetting none. The
result is that `T` reset-aware rollout steps and one reset-aware scan over the
same buffer compute the same numbers — `tests/rl.rs` holds them to `1e-4` — so a
policy gradient taken from the second is a gradient of what the first did. The
sharpest form of the same statement is also tested: no gradient from a
post-boundary loss reaches a pre-boundary observation.

That agreement is not decoration; it is the precondition for PPO. The clipped
surrogate weights each action by how much *more* likely the current policy is to
take it than the policy that collected it, and for a memoryful policy that ratio
only means anything if the replay saw the same history the actor did.
`tests/rl_learn.rs` checks the consequence directly: replay a collected window and
the log-probabilities of the recorded actions come back to `1e-4`, so every ratio
in the first epoch is 1 and the reported KL is zero.

**Nothing in the loop touches the host.** Four operations normally drag a
reinforcement learning implementation back to Rust, and each is a kernel in
[`tensor::ops::rl`](src/tensor/ops/rl.rs) instead:

| operation | what the naive version does | here |
|---|---|---|
| sample an action | download logits, softmax in Rust, upload an id | one kernel: row max, normaliser, inverse CDF |
| score it | a second pass for `log π(a)` | the same kernel, same row |
| record the step | upload into a `[B, T, W]` buffer | an in-place write to column `t` |
| estimate advantages | a `for` loop over a downloaded `[B, T]` | one unit per environment, walking time backwards |
| DAgger's coin flip | a host RNG | a stateless hash of the row index |

Each of those reads would block until the whole queue drained, so the naive loop
synchronises four times per step and the environment and the device take turns
idling. `tests/rl_collect_footprint.rs` runs the full loop — collection, advantage
estimation, batch construction — and asserts that host reads stay at exactly zero
while reserved bytes and per-window dispatch count stay flat.

**Your game, inside the rollout kernel.** The one thing that *does* still stand
between two policy steps is the environment, because a
[`VecEnv`](src/rl/env.rs) is a host object the loop has to call. Hand the crate the
transition itself instead — a `#[cube]` implementation of
[`GameLogic`](src/rl/game.rs) — and it compiles the game into the same kernel as the
action draw and the trajectory write:

```rust
#[cube]
impl<F: Float + CubeElement> GameLogic<F> for Catch {
    fn reset(env: u32, ints: &mut Array<u32>, floats: &mut Array<F>,
             obs: &mut Array<F>, seed_lo: u32, seed_hi: u32,
             #[comptime] spec: GameSpec) { /* ... */ }

    fn transition(env: u32, action: u32, ints: &mut Array<u32>, floats: &mut Array<F>,
                  obs: &mut Array<F>, seed_lo: u32, seed_hi: u32,
                  #[comptime] spec: GameSpec) -> Outcome<F> { /* ... */ }

    // Only consulted under `GameSpec::with_action_mask()`.
    fn legal(env: u32, action: u32, ints: &Array<u32>, floats: &Array<F>,
             #[comptime] spec: GameSpec) -> bool { action < spec.action_dim as u32 }
}

let mut world: GameWorld<R, f32, Catch> = GameWorld::new(32, spec, seed, &device)?;
let report = collector.collect_fused(&mut world)?;   // one kernel a step, not eight
let batch = collector.ppo_batch(&report, &config)?;  // PPO, unchanged
```

A [`GameWorld`](src/rl/game.rs) is an ordinary `VecEnv` too, so the same game runs
through [`Collector::collect`](src/rl/collect.rs) unfused — which is what makes the
fusion checkable rather than merely plausible. `tests/rl_fused.rs` collects three
windows both ways over the same policy, game and seeds and asserts the two
trajectory buffers are **identical**, element for element, down to the advantages
and λ-returns PPO is handed. `tests/rl_fused_footprint.rs` prices it:

| | launches per rollout step |
|---|---|
| the policy | 75 |
| the tail, unfused: draw, score, transition, six column writes | 8 |
| the tail, fused | 1 |

Eight becoming one is 7 fewer dispatches per step per environment batch, exactly,
and the test asserts the number rather than measuring a time. It is worth the most
where reinforcement learning usually lives: the tail is the same eight launches
whatever the policy is, so it is 10% of a step for the small policy above and a
larger share of a smaller one. The policy's own 75 are untouched — that is a
fusion of [`models::mamba3`](src/models/mamba3.rs), not of this loop.
`cargo run --release --example train_rl_fused` trains Catch through it end to end:
0.17 return per episode to 1.00 in 120 rounds, against 0.20 for a paddle that never
moves.

**Masks, anchors and resuming.** Three pieces of the loop exist to keep a long run
honest, and each has one rule worth knowing before using it:

* *Legal-action masks.* [`VecEnv::action_mask`](src/rl/env.rs) returns
  `Result<Option<Tensor>>`, `[envs, actions]` of 0/1; `None` means every action is
  legal **on that step**, and a mask may come and go. An error stops collection
  before the draw, and a window that fails partway resets the collector. A device
  game opts in with `GameSpec::with_action_mask()` and answers
  [`GameLogic::legal`](src/rl/game.rs) (required for every game, `true`-valued for
  one that never restricts); fused and unfused masked windows are byte-identical
  (`tests/rl_fused.rs`). Masked logits are `f32::MIN`, not `-inf`: WGSL has no
  infinity literal, and a kernel that spells one compiles to nothing on wgpu and
  silently yields zeros. Masks are validated once per window, when it becomes a
  batch — a device read only when a mask is present (`tests/rl_masked_footprint.rs`).
* *A reference anchor.* [`ReferencePolicy::snapshot`](src/rl/ppo.rs) deep-copies
  weights, and `score` continues the reference's **own** recurrent history window to
  window, cut at the same episode boundaries as the actor's. `tests/rl_reference.rs`
  holds it to an independently stepped reference to `1e-5`, across SISO and MIMO
  mixers and a masked window.
* *Checkpoints.* `Checkpoint::with_optimizer` stores moments as tensors and the
  optimizer's step counter as an exact integer (`optimizer_steps`; binary format v2,
  v1 still read). [`Checkpoint::restore_training`](src/train/checkpoint.rs) is all or
  nothing, and [`Optimizer::load_state_dict`](src/train/optim.rs)`(params, state,
  steps, strict)` *replaces* optimizer state rather than merging into it. A
  weights-only `restore` is a warm start; the Python learners add the training
  configuration on top (see `bindings/python/README.md`). Exact RL continuation —
  environment, recurrent caches, sampling RNG — is not implemented yet.

**A simulator the crate cannot host.** `GameLogic` fits one signature: two state
arenas of the crate's element types, no read-only side inputs, and a transition
callable from inside a per-environment `if`. Real simulators often fit none of
those. So the loop can be handed back:
[`Collector::collect_with`](src/rl/fused.rs) runs the policy, keeps the recurrent
state, opens and closes the window and builds the PPO batch, and calls you once per
column with a [`FusedStep`](src/rl/fused.rs) — the logits, the critic's estimates,
the trajectory column to fill, the termination flag, the draw's seed and
temperature. You queue one kernel. Build it out of
[`tensor::ops::rl::step`](src/tensor/ops/rl.rs)'s `#[cube]` functions —
`draw_action` and the three recorders — rather than reimplementing them: those are
the crate's *only* sampler, which is what makes the window you collect the window a
PPO replay will score.

This was built against a real one: [`kaggriculture-gpu`](https://github.com/BectorVoom/kaggriculture_engine),
a bit-exact on-device port of a farming-game interpreter. It needs four state
arenas across three element types, four read-only tables, its episode count for
structure-of-arrays addressing, and two cooperative `sync_cube()` loads that must
run *above* the `ABSOLUTE_POS` guard — so it cannot be a `GameLogic`, and one
kernel of its own hosts it comfortably. Driving it from a `collect_with` loop, on
an Apple GPU through `wgpu<msl>` with one client shared between the two crates:

| | |
|---|---|
| correctness | 16 episodes x 12 fused steps land **bit-identical** to the CPU interpreter, replayed from the actions the trajectory buffer recorded |
| cost | 216 fewer dispatches per 24-step window — the tail there is 10 kernels, not 8, because that game also encodes its action into an arena and renders its own observation |
| learning | PPO takes the window reward from -22.3 to 0.0 over 60 rounds |

The last row is the one that cannot be faked: a window can be bit-exact and cheap
and still feed a gradient of something the actor did not do, and only training
catches that.

**Many environments, many threads.** [`MultiSyncCollector`](src/rl/parallel.rs) is
the analogue of TorchRL's `MultiSyncDataCollector`, split down a different seam.
TorchRL gives each worker *process* a replica of the policy, because Python cannot
run one in parallel. Here the workers hold **environments**, and the policy is
evaluated once over their concatenated observations — which is the right trade when
a rollout step costs ~140 dispatches regardless of batch width, so `W` policy
replicas would multiply the dispatch count by `W` and shrink every launch. Two
consequences fall out of that: there is no `update_policy_weights_()` because there
are no replicas to go stale, and observations are never serialised because the
threads share an address space.

```rust
let envs: Vec<_> = (0..8).map(|w| MySimulator::new(w)).collect();
let mut collector = MultiSyncCollector::new(&policy, envs, 128, &device)?;
let report = collector.collect()?;          // 8 threads step; one batched forward
let batch = collector.ppo_batch(&report, &config)?;
```

It earns its keep exactly when the environment is real host work. With a 20 ms
environment, 4 environments per worker, 10 steps per window, on 8 cores:

| workers | environments | one window | same work, one at a time | speedup |
|---|---|---|---|---|
| 1 | 4 | 240 ms | 244 ms | 1.01x |
| 2 | 8 | 250 ms | 518 ms | 2.07x |
| 4 | 16 | 264 ms | 982 ms | 3.72x |
| 8 | 32 | 271 ms | 2.00 s | 7.44x |

The second column is the one to read: the window takes about the same wall clock
whether it is gathering four environments or thirty-two. `tests/rl_learn.rs` asserts
that worker `w`'s rows of the joined batch are *exactly* what that worker's
environment produces on its own, and `tests/rl_collect_footprint.rs` asserts the
fan-out is still read-free — actions are split and observations joined with kernels,
not with a round trip through the host.

**Imitation learning** ([`rl::imitation`](src/rl/imitation.rs)) is the same replay
scored differently: cross entropy against an expert's actions, with an optional
entropy bonus so the cloned policy stays stochastic enough for the reinforcement
learning that usually follows. Behaviour cloning alone only ever sees states the
expert visits, so its first mistake takes it somewhere it was never trained;
[`DaggerSchedule`](src/rl/imitation.rs) decays the expert's share of the *acting*
from 1 to 0, which drifts the labelled states onto the learner's own distribution.
Swapping one for the other is a change of `TrainStep` and nothing else — same
policy, same scan, same episode mask.

**Does it learn?** `cargo run --release --example train_rl` on the task in
[`rl::env`](src/rl/env.rs): a symbol is shown once at the first step of an episode
and the only reward is for naming it at the last. A memoryless policy cannot beat
`1/symbols`, so everything above that line is the recurrent state and nothing
else. Four symbols, four-step episodes, 32 environments, a two-layer `d_model=64`
policy, on the CPU backend:

| run | greedy return | rounds to reach 0.9 |
|---|---|---|
| guessing | 0.25 | — |
| PPO from scratch | 1.00 | 15 |
| cloned (12 rounds), then PPO | 1.00 | 7 |

Cloning reaches the expert in six rounds and cannot pass it; PPO gets there from
reward alone and is what could exceed it. The point of the table is the first
column: both end at the ceiling of a task that is unsolvable without carrying
information across a gap the observation does not contain.

The state is the only thing that persists, it is allocated once, and
`Mamba3StateBuffer::reset_env_states` clears terminated environments in place.
`tests/rl_footprint.rs` runs the loop and asserts that reserved device bytes,
per-step dispatch count and host reads (zero) are all flat.

**Where the step time goes.** `cargo run --release --example bench_rollout`, at
`B = 64`, `D_inner = 256`, `N = 16` on an Apple GPU through wgpu → Metal:

| | per step | dispatches |
|---|---|---|
| recurrence alone, real | 434 µs | 5 |
| recurrence alone, rotational | 448 µs | 8 |
| full 4-layer policy step | 4.85 ms | 143 |

The recurrence — read the state, apply the reset, advance it, read it out — is a
handful of dispatches and comfortably under a millisecond. The policy step around
it is not, and the reason is legible in the second column rather than the first:
this backend charges about 10 µs per dispatch, so 143 of them is a 1.4 ms floor
before any arithmetic happens. A one-environment step costs 1.48 ms, which is
that floor and nothing else. The work to be done is in the projection,
normalisation and gating machinery of the layer, not in the state update.

The numbers above are what this repository can measure. The sub-millisecond
target in the specification is stated for CUDA and Vulkan, where launches cost
roughly a third as much and bandwidth is an order of magnitude higher; **that has
not been verified here** for want of the hardware.


### Distributions

[`distributions`](src/distributions) is `torch.distributions` written as CubeCL
kernels. It exists because a policy *is* a distribution, and because the four
operations a rollout loop performs on one — draw, score, differentiate, compare —
are exactly the four a naive port sends back to the host.

| | |
|---|---|
| scalar families | normal, uniform, exponential, Laplace, Cauchy, Gumbel, half-normal, half-Cauchy, log-normal, Pareto, Weibull, Kumaraswamy, gamma, χ², inverse-gamma, beta, Student's *t*, Fisher–Snedecor, von Mises, continuous Bernoulli, Bernoulli, geometric, Poisson, binomial, negative binomial, and both relaxed Bernoullis |
| over a class axis | categorical, one-hot categorical, multinomial, Dirichlet, relaxed one-hot (Gumbel-softmax) |
| multivariate | a Gaussian with a full covariance, Cholesky-factored on the device |
| combinators | `Independent`, `TransformedDistribution` (affine, exp, sigmoid, tanh, power), `MixtureSameFamily` |
| and | analytic Kullback–Leibler for sixteen same-family pairs, plus categorical and Dirichlet |

Each of `log_prob`, `cdf`, `icdf`, `entropy`, the moments and a draw is **one
kernel**, and so is each of their adjoints. Which family a kernel is for is a
`#[comptime]` discriminant, so the compiled kernel contains one density and no
dispatch; the twenty-six scalar families are one Rust function that reads like a
table of densities.

```rust
// A discrete policy: draw an action and score it in one pass over the logits.
let policy = output.distribution()?;                    // Categorical over the last axis
let (action, log_prob) = policy.sample_with_log_prob(1.0, seed)?;
let bonus = policy.entropy()?;                          // differentiable, one launch

// A squashed Gaussian, which is the continuous-control policy.
let base = Univariate::normal(&mean, &scale, &device)?;
let squashed = TransformedDistribution::new(base, vec![Box::new(TanhTransform)]);
let action = squashed.rsample(seed)?;                   // reparameterised
let log_prob = squashed.log_prob(&action)?;             // with the Jacobian
```

**Why that is faster.** A log-density composed out of elementwise tensor operations
— which is what PyTorch does — is a chain of launches with a full-size intermediate
between each pair. The arithmetic is identical, so the difference is overhead that
was there before. `cargo run --release --example bench_distributions`:

| | composed | fused | |
|---|---|---|---|
| normal log-density, 2²⁰ elements | 1.92 ms, 7 launches | **0.78 ms, 1 launch** | 2.5× |
| PPO's score + entropy, forward and back, 4096 × 18 | 1.25 ms, 44 launches | **0.81 ms, 23** | 1.5× |
| the same at 4096 × 256 | 8.83 ms, 48 launches | **5.06 ms, 23** | 1.7× |
| a categorical step: draw an action, then score it | 0.30 ms, 2 launches | **0.20 ms, 1** | 1.5× |

(Apple M1, eight cores, the CPU backend, best of twenty. The launch counts in the
middle two rows include the reductions the benchmark itself performs; the
distribution's own share falls from seven to two.)

Sampling gets its speed from a different place. The generator is
**Philox-4×32-10**, counter-based, so a draw is a pure function of the element index
and the seed — no state to carry, nothing to write back, no unit waiting on another.
One Philox evaluation produces four words, and the seventeen families that need
exactly one uniform per draw hand those four to four consecutive elements, so a unit
that owns a group of four pays for one evaluation instead of four. On a device with
64-bit integers the round function's high-word multiply is one instruction rather
than the eleven WGSL's lack of a 64-bit integer would otherwise force; which one a
kernel is compiled with is a device query, and the two are asserted to produce
identical bits.

| 2²⁰ draws, one launch each | | before those two changes |
|---|---|---|
| normal (inverse CDF) | **383 M/s** | 93 M/s |
| exponential (inverse CDF) | **654 M/s** | 115 M/s |
| gamma (Marsaglia–Tsang rejection) | **137 M/s** | 75 M/s |
| Poisson, rate 3 (inversion) | **132 M/s** | 77 M/s |
| Poisson, rate 40 (PTRS) | **108 M/s** | 63 M/s |
| binomial, 50 trials (BTRS) | **50 M/s** | 35 M/s |
| von Mises (Best–Fisher rejection) | **77 M/s** | 51 M/s |

The right-hand column is the same code with the four-at-a-time packing and the
64-bit multiply turned off, which is what those two changes are worth: between
1.4× and 5.7×, and most for the families whose only cost *is* the generator.

**Bit-exact, and precisely how far.** The mathematics is written once and compiled
twice — once by `#[cube]` for the device, once by rustc for the host — so "the
device agrees with the reference" is a statement about *one program* rather than
about two transcriptions of one idea. `tools/host_twins.py` performs the three
textual edits that separate the two files, and a test re-derives them and fails if
either has drifted. On top of that:

* `every_distribution_matches_its_host_twin_bit_for_bit` asks all twenty-six scalar
  families for all eight operations over 137 parameter settings and requires every
  result to match the host's exactly — including the samplers, whose *control flow*
  depends on the draws they make.
* `philox_matches_known_answer_vectors` checks the generator against Random123's
  published vectors, so a device-versus-host agreement cannot be two copies of the
  same mistake.
* `draws_do_not_depend_on_the_launch_shape` and
  `rows_are_independent_of_the_batch_they_are_in` pin the property the counter-based
  design exists for: element *i* is the same draw in a batch of seven and a batch of
  four thousand.
* `libm_primitives_agree_with_the_host` is where the claim stops. `exp`, `ln`, `sin`
  and `tan` are library calls, and a backend whose library differs from the host's in
  the last bit will differ. That test measures it and names the culprit, so a failure
  on some future backend reads as "this device's `exp` is one ulp out" rather than as
  an unexplained mismatch four layers up. On the CPU runtime — the reference backend
  — all of them agree, so bit-exactness holds end to end.

Correctness is a separate question from agreement, and gets separate tests.
`tests/golden/distributions.py` transcribes every density from the standard
references in Python, *verifies each transcription by integrating it over its
support and requiring one*, and emits the resulting CDFs, moments and entropies by
the same quadrature. Every sampler then has to reproduce its own analytic moments
over two hundred thousand draws and pass a Kolmogorov–Smirnov test against its own
CDF — which is what catches a rejection sampler whose acceptance test is subtly
wrong, since the density would be right and the draws still biased. Every analytic
gradient is checked against a central difference. The three fail in different ways,
which is the point.

**What is not there.** Three of PyTorch's classes are missing, each for its own
reason. `Wishart` and `LKJCholesky` are distributions over matrices, need a
decomposition per draw, and do not appear in a reinforcement-learning loop.
`LowRankMultivariateNormal` needs the inverse of an `r × r` capacitance matrix, and
a *differentiable* Cholesky is not something the autograd layer has — a version
whose parameters could not be trained would be worse than none. Separately,
`Gamma`, `Beta` and `Dirichlet` report `has_rsample() == false`, where PyTorch
reparameterises them by implicit differentiation of an incomplete gamma: a different
apparatus from a path derivative, and a large one. `mean` and its neighbours return
detached tensors. Parameters are validated by shape and not by value, because
checking that a scale is positive means reading the device back and stalling the
queue.

---

## Python

The reinforcement learning stack is also a Python extension module,
[`bindings/python`](bindings/python) — `pip install maturin && cd bindings/python
&& maturin develop --release`. It exists because the environment is the one part of
a reinforcement learning loop this crate does not own: a simulator that already
exists is usually a Python object, and rewriting it as a kernel to try an idea is
the wrong order to do things in.

```python
import mamba3_rl as m3

env = m3.RecallEnv(num_envs=32, symbols=4, horizon=4)
policy = m3.Policy(m3.PolicyConfig(env.obs_dim, env.action_dim, d_model=64, n_layers=2))
learner = m3.PpoLearner(policy, env, steps=16, learning_rate=1e-3)

for stats in learner.run(rounds=80, epochs=4):
    print(stats.round, stats.episode_return, stats.approx_kl)
```

Any object with `num_envs`, `obs_dim`, `action_dim`, `reset()` and `step(actions)`
takes `env`'s place; `expert_actions()` additionally makes it an expert that
`ImitationLearner` can clone. A round is one call, so the rollout, the action draws,
the trajectory writes, the advantage estimate and every gradient step stay on the
device and the two numbers a human reads are the only synchronisations — except for
an environment written in Python, which pays a copy each way per step and says so
through `read_count()`.

The backend is a build-time choice there, because an extension module is one
compiled artifact: `maturin develop --release --no-default-features --features cuda`.

---

## Running it

```bash
cargo test                                     # 165 tests, ~4 min (the matmul tuner is most of it)
cargo run --release --example train_lm         # train, evaluate, generate, checkpoint
cargo run --release --example generate         # cached decoding + per-token timing
cargo run --release --example finetune_lora    # freeze, adapt, ship, merge
cargo run --release --example train_rl         # PPO and imitation learning, side by side
cargo run --release --example train_rl_fused   # PPO on a #[cube] game fused into the rollout
cargo run --release --example vision           # bidirectional Vision Mamba-3
cargo run --release --example bench            # where the time goes
cargo run --release --example bench_train      # tokens/s at a realistic model size
cargo run --release --example bench_matmul     # every matmul kernel, at the shapes used
cargo run --release --example bench_elemwise   # broadcasting against same-shape
cargo run --release --example bench_distributions # fusion, sampling, a policy's step
cargo run --release --example vs_transformer   # head-to-head against attention
bench/compare.sh                               # head-to-head against PyTorch on ROCm
```

Three environment variables answer the three questions a slow step raises.
`MAMBA3_TUNE_LOG=1` prints the matmul plan chosen for each problem shape and what it
achieved (`=2` prints every candidate it was chosen over);
`CUBECL_DEBUG_LOG=stdout CUBECL_DEBUG_OPTION=profile-medium` prints device time per
kernel; and `MAMBA3_TRACE=1` prints one line per reduction, slice and strided copy
with the shape it ran on, which aggregates into a table of where the memory traffic
goes. The second says which kernel a step is in, the third says which of its calls.
A fourth answers the question a *fast* step should raise: `MAMBA3_TUNE_CHECK=1`
makes the tuner verify every candidate against the simple kernel on every shape it
probes, with the model's real operands, before any of them is allowed to win.

One more changes what is computed rather than reporting on it, which is why it is a
mode and not a tuner candidate: `MAMBA3_MATMUL_PRECISION=bf16` (or `f16`) rounds
every matmul's operands to that type once per call and accumulates in `f32`. Weights,
gradients and every other op stay `f32`. The default is `f32` and nothing changes
unless a run opts in — see "Half the bytes, and the cores that want them" below.

Every one of those runs on every backend. Tests and examples bind to
`backends::Auto`, which resolves from the feature flags, so the same suite runs on
the CPU, on Vulkan, and on ROCm without an edit:

```bash
cargo test --release                                          # CPU (MLIR/LLVM)
cargo test --release --no-default-features --features wgpu    # WGSL
cargo test --release --no-default-features --features vulkan  # wgpu, via SPIR-V
cargo test --release --no-default-features --features hip     # ROCm
cargo test --release --no-default-features --features cuda    # NVIDIA
```

What those print, on one CPU core-set with no GPU:

```text
train_lm      loss 3.19 -> 0.03 in 250 steps; held-out next-token accuracy 97.7%
              prompt [0, 2, 4, 6] -> [8, 10, 12, 14, 16, 18, 20, 22, 0, 2, ...]
generate      parallel vs incremental logits, max |difference|: 7.45e-8
              context   8 tokens ->  1.74 ms per decoded token
              context 512 tokens ->  1.71 ms per decoded token
vision        loss 0.63 -> 0.0001; held-out accuracy 100%
finetune_lora 14.17% of parameters trainable; 0 frozen tensors updated;
              adapter checkpoint 8 512 values vs 60 088 for the full model
train_rl      cue-recall, 4 symbols: guessing earns 0.25 per episode
train_rl_fused  Catch on a 5x5 grid: a still paddle earns 0.20 per episode
              PPO from scratch      0.22 -> 1.00, reaching 0.9 after 15 rounds
              cloned, then PPO      expert matched in 6 rounds; 0.9 after 7
```

The flat decode cost across a 64x context increase is the property the
architecture exists for, and it is measured rather than asserted.

### Backends

| feature | runtime | status |
|---|---|---|
| `cpu` | CubeCL CPU (MLIR / LLVM JIT) | full suite passes |
| `wgpu` | WebGPU, kernels as WGSL | full suite passes (RADV, gfx1151) |
| `vulkan` | the same runtime, kernels as SPIR-V | full suite passes |
| `hip` | AMD ROCm | full suite passes (ROCm 7.1, gfx1151) |
| `cuda` | NVIDIA | builds; no NVIDIA device here — run [`notebooks/mamba3_cuda_colab.ipynb`](notebooks/mamba3_cuda_colab.ipynb) on a Colab GPU for the on-device suite |

The four runnable configurations agree, and not only to a tolerance: `train_lm`
reaches the same 97.7% held-out accuracy and emits the same greedy continuation on
the CPU, on WGSL, on SPIR-V and on ROCm, and `vs_transformer` reports the same
losses to four decimal places on all of them. Weight initialisation is seeded on the
host, so the only way the numbers could drift is a kernel disagreeing.

Every kernel is JIT-compiled on first use, so the *first* forward pass of a process
pays for it. This crate generates a lot of kernels — one per op, times each vector
width the device supports — and on ROCm that came to **7.4 s** before the first
token. CubeCL can persist compiled binaries to disk, but not by default; the
`cubecl.toml` in the repository root turns it on:

```toml
[compilation]
cache = "target"
```

Every run after the first then starts from the cache. On ROCm:

| | cold | warm |
|---|---|---|
| first forward | 7.44 s | 93 ms |
| first forward + backward | 3.94 s | 11.6 ms |
| first decoded token | 116 ms | 1.0 ms |

wgpu compiles through naga fast enough that it barely notices, and the CPU runtime
keeps its MLIR cache in memory only, so it still pays ~0.65 s per process.

Three portability bugs turned up the first time the suite ran on a GPU, all of them
invisible on the CPU:

* An infinite float **literal** is a shader-creation error in WGSL, so `max_dim`'s
  `-inf` identity made every reduction return `NaN`. Reductions now seed the
  accumulator from the first element of the axis and start the loop at one, which
  needs no sentinel at all and is exact for infinite inputs too.
* CubeCL's C++ backends emit vector types as plain structs with no `operator-`, so a
  vectorised `-x` did not compile on CUDA or ROCm. `neg` is a multiply by `-1` now.
* HIP resolves a vectorised `rsqrt` to the `double` overload, which then fails to
  narrow inside the initializer list the vector is built from. `rsqrt` is
  `1 / sqrt(x)`.

### One measurement worth repeating

The matmul kernel started as the textbook shared-memory tiled version — stage a
tile, `sync_cube`, accumulate, `sync_cube`. On CubeCL's CPU runtime a single
64x64x64 product then cost **1.48 s**, against **0.12 ms** for a naive kernel with
no barriers at all: a factor of 12 000. A cube barrier is a hardware instruction on
a GPU and an expensive emulation on a CPU, and the "obviously better" kernel was
making the whole crate unusable — a forward pass took 25 s before the change and
18 ms after.

All four kernels ship, and the default is `MatmulKernel::Auto`, which asks the
device. A hardware plane means a GPU, and a GPU gets `RowTiled`; anything else gets
the barrier-free `Simple` kernel, which is never pathological. Any of them can be
forced with `tensor::ops::matmul::set_default_kernel`.

### The two that came after

The same "measure the runtime, do not assume it" habit found two more, and between
them they are worth two orders of magnitude on the CPU backend.

**Launch geometry is not a constant.** CubeCL's CPU runtime dispatches one task per
*unit* in the cube to a pool of worker threads and waits for all of them. An empty
launch costs ~13 us at one unit per cube and ~85 us at 64 — so the fixed
`ELEMWISE_CUBE_DIM` of 64 was charging every kernel, however small, for 64 thread
wake-ups. `backend::launch_1d` now derives the width from the device: a whole number
of planes on GPU-like runtimes, and on CPU-like ones one thread per ~32 K element
operations, capped at the core count. A tiny model's forward pass went from 313 ms
to 7.6 ms on that change alone.

**Kernels are vectorized.** Every flat kernel is written over `Vector<F, N>` and
launched with the widest width the device reports that divides the element count;
a width of 1 is the old scalar kernel, so nothing regresses on shapes that do not
divide. This is worth the most in `matmul`, where a unit owns `N` adjacent output
columns and the inner loop becomes a vector FMA: a 256x256x256 product costs 13.3 ms
scalar against 51 us at 16 lanes wide. Reductions vectorize both ways — across
`inner` lanes when a middle axis is reduced, and by folding lanes at the end when the
contiguous axis is. The latter reassociates the sum, which is why the parallel-vs-
incremental figure above moved from 2.98e-8 to 7.45e-8.

Together with a fused AdamW kernel (twelve launches per parameter down to one), the
`bench` numbers moved like this:

| | before | after |
|---|---|---|
| forward | 313 ms | 1.3 ms |
| forward + backward + step | 1.61 s | 7 ms |
| decode @ 128 tokens of context | 170 ms | 0.42 ms |
| `train_lm` end to end | 48 s | 19 s |

The last row is smaller than the others because most of what is left is the one-off
MLIR compile.

### And on a GPU, the same question with a different answer

A GPU has registers to spare and a long way to memory, so the winning shape is
different again. `MatmulKernel::RowTiled` gives each unit eight stacked output rows
and one vector column: the `rhs` vector it loads on each `k` step is reused by all
eight accumulators, and the `lhs` value is the same for every unit in the plane, so
it arrives as a broadcast out of cache. Arithmetic per byte goes from about one to
about sixteen. On a Radeon 860M at 1024x1024x1024, in GFLOP/s:

| kernel | wgpu | ROCm |
|---|---|---|
| `Simple` (one unit per output vector) | 300 | 165 |
| `Tiled` (shared memory + barriers) | 342 | 239 |
| `RowTiled` (eight rows in registers) | **586** | **617** |

Eight is where it stops: twelve and sixteen rows spill and fall back to ~330 and
~210 respectively.

### Then a second register axis, and a tuner instead of a rule

`RowTiled` reads eight `lhs` scalars and one `rhs` vector per `k` step and does eight
multiply-adds on them. `MatmulKernel::BlockTiled` gives each unit a *rectangle*
instead of a column — eight rows by one vector of four columns, thirty-two
accumulators — and stages both operands through shared memory first, so the loads are
from LDS rather than global memory. Two details were each worth more than a third of
the kernel:

* **The staged `lhs` tile is transposed, and padded by one element per row.** Without
  the padding, the `bk` units staging one row write addresses a block edge apart,
  every one of them lands in the same memory bank, and they serialise.
* **Which axis consecutive units walk has to follow which axis is contiguous.** That
  is a different answer for a transposed operand than an untransposed one, and
  getting it wrong makes every global load its own memory transaction.

At 1024³ on the same GPU, in GFLOP/s: 624 at a 64x64 block with a 4x4 register tile,
**953** at 128x64 with 8x4, and 649 at 128x128 — the last one spills.

But no shape wins everywhere, and neither does `RowTiled`. A tall block needs rows to
fill it and the scan's products only have sixty-four of them; a deeper `bk` amortises
more of the barrier pair but costs shared memory, which is what limits how many cubes
fit on a compute unit. So `MatmulKernel::Auto` no longer encodes a rule. It **times
the candidates once per problem shape and caches the winner** — four block shapes
(each in its vector-staged form when the shape allows it, scalar-staged otherwise),
the row-tiled kernel, a plane-per-output kernel for the skinny adjoints, and, for a
transposed problem, the option of materialising the transpose and using an
untransposed kernel anyway. A training run issues a few dozen
distinct shapes and repeats them every step, so the probe amortises to nothing, and a
device with a different register file simply produces a different table.
`MAMBA3_TUNE_LOG=1` prints what it chose.

What it chooses on a Radeon 860M, through ROCm, in GFLOP/s — best of twelve calls,
which is what `bench_matmul` reports and roughly what an uncontended machine gives:

| shape | row-tiled | tuned |
|---|---|---|
| 1024³ | 823 | **1120** |
| 2048x4640x512 — the input projection | 725 | **970** |
| 512x4640x2048 — its weight adjoint | 283 | **927** |
| 2048x512x4640 — its input adjoint | 529 | **887** |
| 2048x8192x512 — the output head | 614 | **971** |
| 512 batched 64³ — the scan | 482 | **535** |

For comparison, rocBLAS through PyTorch manages 385 GFLOP/s on the square case in
`f32` on this device — and 3.4 TFLOP/s in `f16`, which is where the real ceiling is
and where none of this goes.

The probes are interleaved rather than run one candidate at a time, which matters
more than it sounds: measuring candidate A five times and then candidate B five times
hands whichever of them coincided with a busy moment a permanent handicap, and the
result is cached for the process. It is still possible to mistune under heavy load —
every candidate computes the same product, so the cost is some percent rather than a
wrong answer.

One candidate in an early version of that list was fast because it was wrong: a `bk`
smaller than the block needs makes the staging loop run zero times, so the kernel
multiplies against whatever was left in shared memory. It looked like a 20% win until
`tests/model.rs` disagreed. `BlockShape::stages_evenly` is now a `const` assertion
over the whole candidate list, checked at compile time.

### Reading a transpose instead of writing one

The adjoint of a matrix product is two more matrix products against transposed
operands, `dA = G Bᵀ` and `dB = Aᵀ G`, and both transposes used to be materialised.
That is the one case the strided copy cannot vectorise — reordering the contiguous
axis is exactly what breaks adjacency — and a training step was doing two of them per
matmul on tensors of a few million elements each. Measured, it was 14% of a step.

Transposition is now a compile-time flag on the block-tiled kernel: both tiles are
staged through shared memory anyway, so it changes one index expression and the
loader's assignment of units, and nothing else. `matmul_nt` and `matmul_tn` are the
public entry points, and `tests/tensor.rs` checks them against materialised
transposes on shapes that do and do not divide the block.

**On a GPU the remaining cost is dispatch, not arithmetic.** One decoded token
through a four-layer model was 558 kernel launches at ~9 us each — 5.3 ms in which
almost nothing was computed. Two easy changes cut it to 460: `x`, `B` and `C` are
taken as one slice of the projection instead of three slices that are immediately
concatenated back together, and `mean_dim` carries its own division instead of
needing a second pass. Getting further meant measuring rather than guessing.

### Fusing what the profile actually pointed at

Counting launches per call site rather than guessing turned up five clusters, and
they were not the ones the architecture diagram suggests. Per decoded token, per
layer:

| | before | after | how |
|---|---|---|---|
| depthwise causal convolution | ~20 | 2 | one kernel reading history and input in place |
| `rotate_halves` (twice per step) | 20 | 2 | one kernel over both halves |
| `permute` on a size-1 axis (x4) | 9 | 0 | recognised as a relabelling, not a copy |
| RMS norm (x3) | 21 | 3 | one kernel per norm |
| state update `αh + βu' + gu` | 5 | 1 | one kernel, coefficients indexed by head |
| SiLU | 2 | 1 | one kernel |
| per-head `alpha`, `beta`, `g` | 8 | 1 | one kernel, packed `[3, batch * heads]` |
| angle advance `wrap(prev + dt*theta)` | 6 | 1 | one kernel |
| `cos phi`, `sin phi`, negation | 3 | 0 | computed inside the rotation |

The convolution was the surprise. Composed of primitives, one decoding step of a
four-tap depthwise convolution concatenates the three-position history onto the new
token, convolves the whole four-long window as a sum of zero-padded shifts — each
shift being a slice, a zero fill and a concatenation — and then throws away three
quarters of the result. Twenty launches to produce one token of output. The fused
kernel reads whichever buffer each window position lives in and never materialises
the window at all.

`permute` was the cheapest fix and worth calling out. A SISO Mamba-3 rotates a
`[b, h, n, 1]` tensor by swapping the last two axes, which moves no bytes — but
`permute` materialises, so it was paying for a full strided copy four times per token
per layer. Nine launches disappeared for a six-line contiguity check.

### The trick that made three of these fit

The last three of the decode kernels needed the same one, because a tape node has one
output: the three coefficients come out packed as a single `[3, batch * heads]` tensor
and the state update reads them packed, so nothing has to slice them apart again. The
rotation takes the angle rather than a `(cos, sin)` pair and computes both inside the
kernel — two transcendentals per element beat three launches, and the step rotates
twice from the same angle.

Together with a fused AdamW, that is **118 launches per decoded token down to 44**,
and decoding at a 128-token context from 1.03 ms to 0.42 ms on the CPU runtime,
0.91 ms to 0.30 ms on wgpu, and 1.05 ms to 0.34 ms on ROCm.

### Then the same question for the backward pass

Decoding is forward-only, so none of that touched training. Profiling a whole
optimizer step — forward, backward and update — put 477 launches on the board for a
single layer at 64 tokens, and pointed somewhere unexpected again:

| | before | after | how |
|---|---|---|---|
| the adjoint of a slice (x18) | ~49 | 13 | write the gradient into a zeroed frame |
| the global gradient norm | ~64 | ~17 | one launch per gradient into a shared buffer |
| scaling gradients by 1.0 | 14 | 0 | notice, and skip |

Differentiating a slice means placing the gradient back where it came from and zero
everywhere else. Composed of primitives that is: allocate a zero block for the head,
allocate another for the tail, concatenate three pieces — two fills and three copies
to move one band of numbers, eighteen times per layer. One kernel does it.

The gradient norm was worse. It squared each gradient, reduced it to a scalar through
a padded tree of `sum_dim`s — the padding itself being a fill and a concatenation —
and then concatenated fourteen scalars to reduce them again. Now every gradient sums
its own squares into a slice of one shared partial buffer, tail included, with no
padding; one reduction finishes the lot. `tests/train.rs` checks the result against
host arithmetic on lengths that do and do not divide the partial count.

**477 launches per training step down to 393.**

Fused ops are the crate's one deliberate exception to compose-don't-fuse, so every
one of them carries two tests: the fused forward against the composed form it
replaced, and its hand-written adjoint against central differences.
`Var::silu_composed`, `Var::rotate_halves_composed` and
`CausalConv1d::apply_composed` are kept public for exactly that.
`backend::launch_count()` is public so the count stays visible.

### Three things that were pure waste

Launch counting finds work that is issued too many times. It does not find work that
should not be issued at all, and profiling a realistic training step — a 31.7 M
parameter model on 2048 tokens, rather than the toy `bench` uses — turned up three of
those. None of them changes a single number the model computes.

**Gradients for constants were computed and then dropped.** A tape node keeps a slot
for every parent so the rule's outputs line up, and an untracked parent's gradient was
simply discarded after being computed in full. For a broadcasting multiply that is not
free: the scan multiplies `[batch, chunks, chunk, chunk, heads]` intermediates by
constant causal masks, and each discarded gradient was a full-size product followed by
a three-axis reduction down to the mask's shape. `Var::record_with_mask` hands the
rule a per-parent flag and the four broadcasting binary ops skip what nobody wants.

**The global gradient norm ran on 256 units.** `sum_squares` capped its partial count
at 256, which is a fine number of partials and a terrible number of lanes: a 2.4 M
element gradient — an ordinary input projection — had each unit walking some 2 400
vectors, and the norm cost 5% of a step. One lane per few hundred elements, capped so
the partial buffer stays negligible.

**The causal mask in the scan was redundant.** `decay` is only ever used multiplied by
`weight`, and `weight` is built from a strictly-lower mask plus a diagonal — so it is
already zero everywhere the causal mask is. The clamp on the exponent is what makes
dropping it safe: it bounds the upper triangle at `exp(0)`, so the values being
multiplied by zero are finite. One full-size broadcast multiply per layer, and its
adjoint, disappear for a comment.

### A step that never stops to look

The training loop used to read two device values in the middle of every step: the
loss, between the forward pass and the backward pass, and the gradient norm, to decide
a clipping factor before the update. Each one drains the queue — the backward pass
cannot be enqueued until the forward pass has finished *running*, not merely been
submitted.

Neither read is necessary. The loss is kept as a tensor and read after the optimizer
step is on the queue. The clip is computed on the device: one kernel turns the sum of
squares into a one-element scale factor — folding in the micro-batch averaging that
used to cost a launch per gradient — and the fused AdamW kernel reads that factor as a
buffer. `Optimizer::step_scaled` is the interface that carries it, and
`tests/train.rs` checks the device-side factor against the host arithmetic it
replaced, on norms either side of the threshold and with averaging that is not one.

One synchronisation per step, at the end, with the whole step already running. On a
loaded machine this was the difference between the median step and the best one.

### Permutations that mostly are not

`permute` was the last scalar kernel in the crate, on the grounds that reordering axes
is exactly what breaks adjacency. It breaks it less often than that suggests. Two
properties recover most of the cost:

* **Axes that stayed adjacent can be merged.** Output axes `d` and `d+1` describe one
  contiguous run of source memory whenever `src_stride[d] == src_stride[d+1] *
  dim[d+1]`, so they fold into one axis before the kernel runs. The rank-5
  permutation the scan uses to put heads in front of positions collapses to rank 3,
  and the per-element index arithmetic is a division and a modulo per axis.
* **The innermost axis is often untouched.** If it is, the copy moves whole vectors,
  and both the loads and the index arithmetic are divided by the vector width.

What is left — a permutation that genuinely transposes the contiguous axis — still
runs the scalar kernel, and that is the honest cost of moving those bytes. Between
this and reading transposes in the matmul kernel, strided copies went from 14% of a
training step to under 4%.

---

## Asking a different question: where does the *time* go?

Everything above counts launches, because for a small model the launch count is the
wall clock. At a realistic training size it stops being: a step issues about 2 400
launches and takes the better part of two seconds, so the average launch is worth most
of a millisecond and dispatch is a rounding error. The question becomes which kernels those seconds are in,
and — because a kernel name covers many shapes — which *shapes*.

Two tools answer the two halves. CubeCL will time every dispatch and print a summary:

```bash
CUBECL_DEBUG_LOG=stdout CUBECL_DEBUG_OPTION=profile-medium \
    cargo run --release --no-default-features --features hip --example bench_train
```

and `MAMBA3_TRACE=1` prints one line per launch of every reduction, slice and strided
copy with the shape it ran on, which aggregates into a table of where the memory
traffic goes. The first says `SumDimKernel` was 16% of a step; the second says which
of its 178 calls that was. Five things fell out, and between them they were more than
half of a training step.

### A slice's adjoint is cheap. Five slices' adjoints are not.

A Mamba-3 layer cuts its fused projection five ways — `z`, `xBC`, `dt`, `lambda`,
`theta` — and then cuts `xBC` three ways again. Differentiating a slice writes the
gradient into a full-size buffer that is zero outside one band, which is one kernel
and unavoidable. Differentiating *five* slices of the same tensor writes five
full-size buffers that are zero outside one band each, and then adds them together.
On `[4, 512, 4640]` that is 190 MiB written to place 38 MiB of gradient, plus four
full-size adds to combine them: some 650 MiB per layer, five gigabytes per backward
pass, to move each number exactly once.

The pieces tile the axis, so the gradient of the whole is their *concatenation*, and
a concatenation writes one buffer in bands. `Var::split` says so. It needs a trick,
because a tape node has one output gradient and a split has many: the pieces share a
**sink** node created *before* them, each piece stashes its band and hands the sink an
empty token, and the sink — reached last, because ids increase with creation order and
the backward walk descends them — assembles the whole thing in one `cat`. Pieces that
never receive a gradient leave a zero band.

Five full-size writes and four full-size adds became one buffer written once. While
the same file was open: a slice covering its entire axis is the identity, and
recording it as a slice made a rank-1 MIMO scan pay a full-size copy per rank pair in
the backward pass for a tensor it was moving onto itself.

### A reduction with few outputs runs on few units

`sum_dim` gives each output element one unit and walks the reduced axis. That is the
right shape when there are plenty of outputs. The gain gradient of an RMS norm is
`[32768, 64]` summed down the rows — **sixty-four output elements, so sixteen
vectorised lanes on the whole device**, each walking half a million floats. Reductions
were 16% of a training step and most of it was that.

Splitting the axis fixes it without a new kernel: reduce to `groups` partials first
and then reduce the partials, which is two reshapes and the same kernel twice, and the
first pass runs `groups` times as wide. The factor is chosen to bring the lane count
up to a device's worth while leaving each lane real work to do, and it has to divide
the axis exactly, which every axis here does. **481 ms to 53 ms.**

### One row per unit is the wrong shape for a row-wise kernel

RMS norm had the same disease in a form the split cannot reach: one unit owns one row
and walks it. Neighbouring units then read addresses a whole row apart, so a wave's
load instruction touches as many cache lines as it has lanes and uses a fraction of
each. Measured across a step's worth of norms: **about 6 GB/s**, against roughly 40 GB/s
for a plain elementwise pass over the same tensors.

Giving each row a *plane* instead — lanes striding the row, `plane_sum` for the total —
makes every load contiguous across the lanes that issue it. The reduction stops being
a serial loop and the divergence is confined to the lane-zero write of the scale.
The two kernels together went from **163 ms to 28 ms**. The per-unit kernels
stay for runtimes with no planes to give, which is CubeCL's CPU backend, where a unit
is a thread and one row per thread was right all along.

### The scan's mixing matrix is a function of three vectors

The intra-chunk step multiplies `C Bᵀ` by a decay `exp(acum[t] - acum[s])` and a
weight that is `w[s]` below the diagonal, `g[s]` on it and zero above. Written out of
primitives that is a broadcast subtract, a clamp, an exponential, two masked broadcast
multiplies and an add — six passes over a `[batch, chunks, chunk, chunk, heads]`
tensor, each existing only to be consumed by the next, and a seventh to multiply the
result into `C Bᵀ`. Then two full-size permutations, because the band is built with
heads last and the matmuls on either side of it want heads first.

But the decay and the weights never depend on the state or on the head dimension:
**the whole `chunk × chunk` band is a function of three vectors of length `chunk`.**
`Var::ssd_band` builds it in one launch that reads three `[rows, chunk]` vectors and
writes the band — and builds it head-major, so both permutations disappear as well.
Its adjoint is one launch that walks a row and a column per output and produces the
three vector gradients directly; the clamp's derivative, which is the part that is
easy to get wrong, is checked against central differences across the floor in
`tests/autograd.rs`.

Seven passes and four permutations per layer, forward and backward, became one launch
each: **7.5 ms forward and 2.9 ms backward for the whole step**.

### A weight gradient does not need one product per batch element

`[batch, seq, in] @ [in, out]` broadcasts the weight across the batch, so the weight
adjoint `Aᵀ G` is a batched product that yields one `[in, out]` gradient per batch
element and then sums them. For a Mamba-3 input projection that is four `[512, 4640]`
gradients written and a 38 MiB reduction to get back to one. Folding the batch axes
into the contraction instead makes it a single product with a four times longer inner
dimension — the same arithmetic, a quarter of the memory, no reduction — and it is a
reshape, because both operands are contiguous with the batch axes outermost.

The same file learned that a broadcast gradient's *adjacent* reduced axes are one axis
after a free reshape, so undoing a `[1, 1, heads, 1, state]` broadcast is one launch
rather than two with an intermediate between them.

### What that came to

Device time per step on the benchmark model, from one alternating pair of profiled
runs — before and after, on the same machine within a minute of each other, because
absolute numbers here move with whatever else is running:

| | before | after |
|---|---|---|
| reductions (`sum_dim`, both kernels) | 481 ms | 53 ms |
| the adjoint of the projection's five slices | 236 ms | 0 ms |
| RMS norm, forward and backward | 163 ms | 28 ms |
| broadcast multiplies, adds and strided copies — mostly the scan | 609 ms | 230 ms |
| the fused band that replaced most of that row | – | 10 ms |
| **everything that is not a matrix product** | **1828 ms** | **739 ms** |
| matrix products | 1124 ms | 1100 ms |

Two things are worth saying plainly about that table. The first is that the matmul row
did not move — the batch fold changes *which* matmul kernel runs and deletes a 38 MiB
reduction after it, not how much arithmetic there is. The second is that this is
therefore where it stops being worth pushing: matrix products are now 60% of a step,
they run at 630–890 GFLOP/s in `f32` on the shapes that matter, and the scan's own
products — 512 batched 64³ — are *memory* bound at about 10.7 FLOP per byte, which
puts their ceiling near 400–450 GFLOP/s against the 350–460 they achieve. No
block-shape tuning moves those, and the honest next lever is not a better `f32`
kernel; it is `bf16` and the matrix cores this crate does not use.

### That paragraph was half wrong

No block-shape tuning moved the matmuls — but their *staging* did, and staging was
never about block shapes. Three changes, each measured as a tuner candidate against
the kernel it replaces, interleaved in the same run so both sides see the same
machine:

* **Prefetch one `bk` step ahead.** The block kernels' global loads sat on the
  critical path between the two barriers: stage, wait on DRAM, compute, repeat.
  Fetching step `s + 1` into registers right after the first barrier lets those
  loads retire behind step `s`'s arithmetic. Measured alone it does nothing for the
  plain kernel — eight waves per cube already hide the latency — and helps the
  transposed one on most shapes, by up to a third, though not uniformly.

* **Stage `lhs` as vectors.** The plain kernel's one scalar global access was the
  `lhs` tile: `bm * bk` loads per step, each its own instruction using a fraction of
  a cache line. Rows of `lhs` are contiguous along `k`, so when the vector width
  divides `k` — every shape a training step issues but one — a unit can fetch four
  `k` at once and scatter them into the transposed shared tile. A quarter of the
  load instructions, and the shared-memory pad grows from one to two, because the
  scatter changes which stores collide. This is the change that moved everything:
  1.34–1.45x on the forward projections and, decisively, on the *adjoints* — the
  tuner now prefers materialising a transpose and running the vector-staged kernel
  over the transposed-index kernel it used to pick, 1.35x on the weight gradients
  and 1.71x on the input projection's.

* **A plane per output element for the skinny adjoints.** `[8, 4096] @ [4096, 8]ᵀ`
  is sixty-four dot products, not a matmul; a block kernel owns at most 64 outputs
  per 256-unit cube and spends its life at barriers, measured at 7 GFLOP/s. The
  right kernel is the one the fused reductions already use — lanes stride `k` in
  vectors, one `plane_sum` at the end — and it is possible exactly when both
  operands are contiguous along `k`, which is what `dA = G Bᵀ` gives. 8x on that
  shape.

The wrong-and-fast lesson from the `bk` bug also became a harness instead of a
memory: `MAMBA3_TUNE_CHECK=1` makes the tuner verify every candidate against the
simple kernel, on every shape, with the model's real operands, before any of them
can win. Running `bench_train` under it checks the full candidate list on the full
shape repertoire of an actual model; it is how every number above was validated.

On the training benchmark, same alternating protocol as always — this time under a
load average above ten, because the measurement methodology exists precisely so a
busy machine is not an excuse. Three alternating rounds, best and median step each:
the new build won five of the six comparisons, its best step was 1.33 s against the
old build's 1.76 s, and its median beat the old one every round, by 1.26x to 2.10x.
The comparison it lost — one round's best step, by 16% — is what a load spike does
to a best-of under contention, which is exactly why the medians are printed too.
`train_lm` still reaches 97.7% held-out accuracy from the same seed.

### Four chains that were still four chains

The fusion discipline had been applied to whatever the profile pointed at, and the
profile had stopped pointing. So the next round counted *launches* instead —
backend-independent, and the one number that says what a step costs before it says
what it costs here. A training step of the benchmark model issued 1093 of them.

Four of the remaining chains were worth a kernel:

* **The scan's decay pattern.** `exp(clamp(a - b, floor, 0))`, sometimes times a
  mask, appears five times in `ssd_chunked` on tensors sized by the chunk count.
  Composed it is three or four passes forward, and backward the clamp's adjoint
  alone is four launches and three full-size intermediates. `Var::exp_decay` is one
  each way, with broadcasting on all three operands so the sites that subtract a
  broadcast pair and the site that multiplies a causal mask are the same kernel.

* **The trapezoid weights.** `g = λ·dt` and `w = g + shift_left((1-λ)·dt)` was two
  multiplies, a scalar subtract, a zero fill, a slice, a concatenation and an add —
  seven launches to produce two tensors the size of `dt`, one of which exists only
  to be read one position over. One kernel now writes both and reads the shift in
  place. A tape node has one output, so the pair rides the same sink arrangement
  `Var::split` uses: the outputs stash their gradients with a node created before
  them, which the backward walk therefore visits after both.

* **Cross entropy.** This one was not about launches. `log_softmax` composed with
  `take_along_last` is six vocab-sized passes forward, and the adjoint of
  `take_along_last` *materialises a dense `[positions, vocab]` one-hot* and
  multiplies it away — the largest transient allocation in the step, to encode one
  integer per row. The fused kernel is one launch each way: a plane per row for the
  two reductions, label smoothing folded in as the closed form
  `lse - (1-s)·x_target - s·mean(x)`, and a backward that rebuilds the softmax from
  a saved `[positions]` log-sum-exp instead of saving anything vocab-sized.

* **Softplus**, seven launches to one, which mattered least and was cheapest.

1197 → 1093 → **911** launches per step, forward 405 → 272 and backward 686 → 533.
The wall-clock on the CPU box these were written on is noise at this model size —
the same build varies ±15% between runs — which is the reason to count launches: on
a GPU each one is a fixed 9–13 µs of dispatch, and 182 of them is real time that no
profile of *this* machine would have shown.

### Half the bytes, and the cores that want them

The paragraph two sections up ended by naming the next lever: `bf16` and the matrix
cores. Both are now here, and the first is measured.

The mode is opt-in and it is a *mode*, not a tuner candidate. Everything else in this
file is a backend tuning knob — every kernel computes the same product to within
floating-point associativity, so the tuner may pick freely. Rounding operands to 8 or
11 mantissa bits breaks that contract, and a candidate that can silently win by being
less accurate is not a tuner candidate; it is a decision the caller makes.
`set_matmul_precision(Bf16)` — or `MAMBA3_MATMUL_PRECISION=bf16` — rounds each matmul
operand once and accumulates in `f32`. Master weights, gradients, the optimizer and
every non-matmul op are untouched, which is the autocast recipe and needs no loss
scaling.

It cost one generic parameter. All seven matmul kernels now carry a storage element
`ES` beside the accumulator `F`: operands are read as `ES`, shared tiles are staged in
`ES`, and the widening happens at the multiply. The ordinary path instantiates
`ES = F`, where the cast is the identity — the `f32` build is bit-identical and its
launch count unchanged. The cast happens once per operand in `matmul_3d_t`, one place,
so the tuner's own probes and the transposed adjoint forms all see one consistent
mode; the tune cache is keyed by the element-type pair, and `MAMBA3_TUNE_CHECK`
compares a mixed candidate against the simple kernel *reading the same rounded
operands*, which is what keeps its tolerance meaningful rather than merely loose.

The first surprise was that none of this needed a GPU to test. CubeCL's CPU runtime
compiles and runs real `bf16` and `f16` — so the modes are checked locally against a
host-side reference that rounds the same way, on every kernel the tuner can reach,
forced one at a time because an instantiation that never launches is never compiled.
`train_lm` under `bf16` reproduces the `f32` run to four decimal places: 3.1907 →
0.0335, 97.7% held-out, the same greedy continuation.

There are two narrow types because the *hardware* has two. Every tensor-core
generation does `f16`; `bf16` fragments arrived with Ampere and RDNA3. `bf16` is the
training recommendation — its exponent range is `f32`'s, so nothing overflows — and
`f16` is what makes the fragment path testable on a free Colab T4, which is Turing.

Which brings the matrix cores in as a tuner candidate *inside* those modes. The kernel
keeps the block-tiled skeleton and changes the two things a cooperative instruction
forces. A fragment belongs to a plane rather than a unit, and the lane-to-element
mapping is deliberately opaque, so a plane owns one 16×16 output tile and nothing
indexes a register tile. And the staging layout flips: the register kernels stage
`lhs` transposed because a unit wants `tm` values for one `kk` adjacent, while
`cmma::load` wants the tile as the fragment reads it, so both tiles are staged
row-major and both fragments are `RowMajor` — transposition is absorbed into the
staging index, as it already was for the transposed block kernel, so one kernel covers
all four operand combinations. The accumulator lives across the whole `k` walk and is
written out once through a shared scratch, the bounce the CubeCL manual describes as
the unavoidable cost on the output side, which is also the only place the `m` and `n`
tails get bounds-checked.

It is offered only when the device reports the exact `(ES, ES, f32, 16, 16, 16)`
fragment it needs — a membership test against the capability set the runtimes register
per architecture, not a guess from a device name — so CPU and wgpu never see it.

**And it has not yet run on hardware.** Nothing here has tensor cores; the kernel
compiles, its capability gate and its fallback are tested, and the test that would
exercise the fragments skips itself for want of a device. The Colab notebook
(`notebooks/mamba3_cuda_colab.ipynb`) is where it first executes, under
`MAMBA3_TUNE_CHECK`, which will reject a wrong product before the tuner can prefer
it. Until those numbers exist the honest summary is: the `bf16` mode is measured and
converges, and the matrix-core path is written, gated, and unproven.

---

## Against PyTorch, on the same GPU

`bench/torch_mamba3.py` is a direct port of `src/models/mamba3.rs` and
`src/ssm/scan.rs`: the same fused input projection, the same depthwise causal
convolution, the same `B`/`C` bias and RMS norm, the same learned-trapezoidal weights,
the same rotating frame, and the same chunked scan written out of the same primitives.
Neither side hand-fuses the scan, so what is being compared is two implementations of
one algorithm rather than two algorithms.

`bench/compare.sh` alternates them. That matters more than it sounds: this is an
integrated GPU that shares its memory bandwidth with the rest of the machine, so
running one to completion and then the other measures the machine's mood as much as
the code. Interleaving gives both the same distribution of interference, and both
report the best step of the run rather than the mean.

31.7 M parameters, batch 4 x 512 tokens, `f32`, on a Radeon 860M (gfx1151):

| | tokens/s |
|---|---|
| PyTorch 2.13 + ROCm 7.1, eager | 882 – 1090 |
| PyTorch 2.13 + ROCm 7.1, `torch.compile` | 1146 – 1196 |
| **mamba3, ROCm** | **1871 – 2271** |
| **mamba3, Vulkan (the same GPU, through wgpu)** | **2457** |

and where that started:

| | tokens/s | ms/step |
|---|---|---|
| mamba3, ROCm, at the first commit | 551 | 3720 |
| mamba3, ROCm, after the launch-count work | 1225 – 1517 | ~1760 |
| mamba3, ROCm, after the device-time work | 1871 – 2271 | ~1190 |

The ranges are real and they are why the ranges are printed: this machine's load
average moved between 1 and 50 while these were being taken, and both implementations
move with it. What does not move is the ordering. Alternating runs on a quiet machine
came out at **1.92x against eager PyTorch and 1.90x against `torch.compile`**, and
mamba3 won every individual round of both — 1.72x to 2.00x eager, 1.68x to 1.95x
compiled. Against the crate's own previous state, on the same alternating protocol,
every round on a quiet machine was between **1.31x and 1.48x** — and every round on a
loaded one was better than that, up to 2.20x, which is the expected shape of a change
that mostly deletes memory traffic: the less bandwidth a step needs, the less it
cares who else is asking for some.

So: comfortably ahead of PyTorch on ROCm whether or not the reference is compiled, and
**4x ahead of where this crate started.** Running the identical Rust through Vulkan
instead of ROCm — the same GPU, the same kernels, a different shader compiler — is
faster again, which remains a fact about hipRTC rather than a compliment to this crate.

It holds across shapes rather than depending on one: **1.77x** at `d_model` 768,
**2.10x** at 1024 tokens by batch 2, **1.89x** at 256 tokens by batch 8 — all against
eager PyTorch, all on the same alternating protocol.

The honest other half: PyTorch's peak memory on this benchmark is 3.2 GiB against a
larger figure here, because the scan materialises intermediates a fused kernel would
keep in registers — see what is not optimised, below. And `torch.compile` spends 66
seconds warming up, against three to six for this crate with its kernel cache
populated — which matters for short runs and not at all for long ones.

```bash
python -m venv .torch && .torch/bin/pip install torch \
    --index-url https://download.pytorch.org/whl/rocm7.1   # match your ROCm

cargo build --release --no-default-features --features hip --example bench_train
PYTHON=.torch/bin/python bench/compare.sh 4               # eager
TORCH_ARGS=--compile bench/compare.sh 3                   # against inductor
SEQ=1024 BATCH=2 bench/compare.sh 3                       # any shape bench_train takes
```

---

## Against a Transformer

`examples/vs_transformer.rs` builds both models out of the same `Mamba3LmConfig`.
The only difference is `LayerPattern`: `AllMamba` gives a recurrent stack,
`AllAttention` plus a SwiGLU feed-forward gives an ordinary pre-norm Transformer
with RoPE and causal masking. Embeddings, RMS norm, the optimizer, the schedule, the
batches and the initialisation seed are shared, so what is being compared is the
mixer.

Two Transformers are trained, because parameter matching cuts both ways at this
scale. `transformer` uses the textbook `8/3 * d_model` SwiGLU width and ends up with
twice the parameters; `transformer-lite` has its width shrunk until the counts
match. Numbers below are one run on a Radeon 860M through wgpu, 200 steps on
64-token sequences.

> **These timings predate the device-time round above and have not been re-measured.**
> Both columns moved — the fixes were to reductions, RMS norm and the projection
> split, which both architectures use, plus the scan's band, which only one of them
> does — so the crossovers below are now pessimistic for Mamba-3 and the Transformer
> column is faster than it says too. Re-running it on this machine gave a spread of
> more than 2x on the short rows between three consecutive runs, which is not a table
> worth printing; the quality figures and the cache sizes are unaffected.

```text
model                   params      loss@1    loss@end    held-out  train time
mamba-3                 390448      4.1728      0.0340       98.6%       8.97s
transformer             808064      4.1914      0.0417       98.4%       5.34s
transformer-lite        390272      4.2066      0.0468       98.2%       4.39s
```

Then the same two models, timed as the sequence grows:

```text
one training step (fwd + bwd + update, batch 2)   one decoded token (batch 8)
 tokens    mamba-3  transformer  speedup           ctx    mamba-3  transformer  speedup
     64    28.71ms      18.62ms    0.65x           128     2.61ms       2.68ms    1.02x
    128    30.53ms      20.11ms    0.66x           512     2.58ms       5.45ms    2.11x
    256    42.32ms      30.83ms    0.73x          1024     2.64ms       9.65ms    3.65x
    512    53.07ms     245.72ms    4.63x          2048     2.43ms      35.24ms   14.50x
   1024   139.49ms       1.19 s    8.51x          4096     2.36ms      89.30ms   37.86x
```

and the cache each one carries while decoding:

```text
 context   mamba-3      transformer KV
     128    1176 KB            4640 KB
    4096    1176 KB          131616 KB
```

So: better loss and better held-out accuracy than a Transformer with **twice** the
parameters, 21x faster to prefill 2048 tokens, faster per decoded token at *every*
context measured — rising to 38x at 4096 — and a decoding state that does not grow at
all, 112x smaller than the KV cache by the time the context reaches 4096.

The honest other half used to be larger than it is. Below a few hundred tokens
neither model does enough arithmetic to matter and the cost is kernel dispatch, of
which the recurrence issues more; the Transformer won every short-sequence column.
Cutting launches per decoded token from 118 to 44 — see the fusion section — removed
the decoding crossover entirely and left one only on a training step, at around 350
tokens. What is below that is still a dispatch-count problem, not an architectural
one.

Run-to-run spread on the Transformer's long-context rows is wide, tens of percent,
because a 4096-token attention pass on an integrated GPU is memory-bound and shares
its bandwidth with the desktop; the 4096 row has been seen anywhere from 54 ms to
89 ms. The Mamba-3 rows are flat to within a few percent across every run, which is
itself the point.

---

## What is *not* optimised

Being explicit about this, because the gap is real and the code is written so it
can be closed incrementally:

* **The scan is composed, not fused.** Its `[chunk, chunk]` band is now one kernel,
  but the five matrix products around it still hand each other whole tensors where a
  fused kernel would keep a tile in registers and never write it. It is
  `O(T·(N+P)·chunk)` and matmul-bound, which is the right asymptotics. The fusion
  target is `ssd_chunked`; its contract (inputs, outputs, boundary state) is exactly
  what a fused kernel would need.
* **MIMO runs `R²` SISO scans** instead of one rank-aware kernel.
* **Broadcasting binary ops upload a small metadata buffer per call.** Measured
  against the same-shape path, a broadcasting op is not slower per output element than
  a flat one — the pooled allocator absorbs a hundred-byte upload — so this one is
  listed for honesty rather than as a lead.
* **Most things are still one op per launch.** Roughly a dozen chains are fused; the
  rest are not. At ~13 us of fixed dispatch on the CPU runtime and ~9 us on wgpu a
  small model is still launch-bound, and what is left is no longer clustered: a
  training step's biggest remaining category is plain broadcasting multiplies and
  adds inside the adjoints, spread across every rule in `autograd::ops`. Picking those
  off one at a time has stopped paying; the next real win is a general elementwise
  fusion pass — a chain builder that emits one kernel from a comptime opcode list —
  not another hand-written kernel. `backend::launch_count()` reports the running
  total.
* **No tensor cores.** Every matmul kernel here is a plain FMA kernel. CubeCL exposes
  `cmma`, an RDNA3 iGPU reports `min_tensor_cores_dim: 16`, and PyTorch reaches
  3.4 TFLOP/s in `f16` on this device against 385 GFLOP/s in `f32` — so the ceiling
  this is measured against is about four times higher than the one it is hitting.
  Using it needs the mixed-precision work below first.
* **ROCm is the slower of the two paths to this GPU.** The same kernels compiled
  through naga to SPIR-V and run on RADV stay ahead of the same kernels through
  hipRTC — 2457 tokens/s against 2271 on the training benchmark. The gap narrowed as
  the crate stopped being bound by things a shader compiler cannot help with, which
  is itself a hint about where it comes from, but nothing in this crate explains it
  and it has not been chased.
* **Mixed precision is wired but untested at scale.** `FloatElem` is implemented
  for `f16`/`bf16`; the accumulation strategy needed for stable low-precision
  training is not yet in place.

---

## Layout of the source

```
src/
├── backend.rs           Runtime + element-type abstraction, Device
├── error.rs
├── tensor/
│   ├── base.rs          contiguous device tensor
│   ├── shape.rs         shapes, strides, broadcasting
│   └── ops/             elemwise, matmul, reduce, movement, index, scan, random,
│                         rl (action sampling, advantages, trajectory writes),
│                         fused (kernels that collapse an op chain into one launch)
├── autograd/
│   ├── graph.rs         tape, node ids, Grads
│   ├── grad_mode.rs     no_grad
│   ├── var.rs           Var + backward
│   └── ops.rs           the differentiable primitives
├── nn/                  param, module, init, linear, norm, embedding, conv,
│                        rope, attention, mlp, dropout, lora, quant
├── ssm/
│   ├── config.rs        Discretization / StateDynamics / SsmMode
│   └── scan.rs          the chunked scan + single-token step
├── models/              mamba3, hybrid, lm, vision
├── train/               optim, sched, loss, tasks, trainer, checkpoint
├── infer/               state cache, samplers, generator
├── distributions/       torch.distributions as kernels:
│   ├── rng.rs           Philox-4x32-10, counter-based, with its host twin
│   ├── special.rs       lgamma, digamma, trigamma, erf, erfc, erfinv, Bessel
│   ├── univariate.rs    26 scalar families: density, CDF, quantile, entropy,
│   │                     moments, sampler and every adjoint, comptime-dispatched
│   ├── categorical.rs   row kernels over a class axis
│   ├── simplex.rs       Dirichlet, one-hot and relaxed categoricals, multinomial
│   ├── multivariate.rs  full-covariance Gaussian, Cholesky, triangular solves
│   ├── combinator.rs    Independent, TransformedDistribution, mixtures
│   ├── elementwise.rs   the special functions as differentiable tensor ops
│   └── kl.rs            analytic Kullback-Leibler divergences
└── rl/                  persistent rollout state, actor-critic policy, rollout
                         engine, trajectory buffer, collector, parallel worker
                         pool, PPO, imitation learning, a vectorised environment
                         trait, and user game logic as device code (game.rs)
                         fused into the rollout step (fused.rs)

bench/
├── torch_mamba3.py      the same model in PyTorch, as the reference to beat
└── compare.sh           runs both, alternating, and reports the best step of each

bindings/python/         the rl stack as a Python extension module (pyo3 + maturin)
├── src/                 policy, rollout, environments, PPO, imitation learning
├── python/mamba3_rl/    the package: re-exports, the VecEnv protocol, type stubs
├── tests/               pytest, against whichever backend the module was built with
└── examples/            the recall task, and an environment written in numpy
```

---

## References

* Mamba-3: *Improved Sequence Modeling using State Space Principles* (ICLR 2026) —
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
* Mamba-2 / SSD: *Transformers are SSMs* — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
* CubeCL — <https://github.com/tracel-ai/cubecl>

## License

MIT OR Apache-2.0.
