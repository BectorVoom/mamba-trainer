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
| 5 | [`train`](src/train), [`infer`](src/infer) | optimizers, schedules, trainer, caches |

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

## The five extension points

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

---

## Running it

```bash
cargo test                                     # 59 tests on the CPU backend, ~15 s
cargo run --release --example train_lm         # train, evaluate, generate, checkpoint
cargo run --release --example generate         # cached decoding + per-token timing
cargo run --release --example finetune_lora    # freeze, adapt, ship, merge
cargo run --release --example vision           # bidirectional Vision Mamba-3
cargo run --release --example bench            # where the time goes
```

What those print, on one CPU core-set with no GPU:

```text
train_lm      loss 3.19 -> 0.03 in 250 steps; held-out next-token accuracy 97.7%
              prompt [0, 2, 4, 6] -> [8, 10, 12, 14, 16, 18, 20, 22, 0, 2, ...]
generate      parallel vs incremental logits, max |difference|: 2.98e-8
              context   8 tokens -> 29.21 ms per decoded token
              context 512 tokens -> 28.86 ms per decoded token
vision        loss 0.63 -> 0.0001; held-out accuracy 100%
finetune_lora 14.17% of parameters trainable; 0 frozen tensors updated;
              adapter checkpoint 8 512 values vs 60 088 for the full model
```

The flat decode cost across a 64x context increase is the property the
architecture exists for, and it is measured rather than asserted.

Backends are feature flags; exactly one runtime needs to be enabled:

```bash
cargo build --no-default-features --features wgpu
cargo build --no-default-features --features cuda
```

The CPU runtime JIT-compiles every kernel through MLIR, so the *first* forward
pass of a process pays a one-off compile (~0.6 s for the models in `bench`);
everything after that runs at steady state.

### One measurement worth repeating

The matmul kernel started as the textbook shared-memory tiled version — stage a
tile, `sync_cube`, accumulate, `sync_cube`. On CubeCL's CPU runtime a single
64x64x64 product then cost **1.48 s**, against **0.12 ms** for a naive kernel with
no barriers at all: a factor of 12 000. A cube barrier is a hardware instruction on
a GPU and an expensive emulation on a CPU, and the "obviously better" kernel was
making the whole crate unusable — a forward pass took 25 s before the change and
18 ms after.

Both kernels ship. The default is the barrier-free one, because it is never
pathological; backends where barriers are cheap can switch with
`tensor::ops::matmul::set_default_kernel(MatmulKernel::Tiled)`. The same lesson
set `ELEMWISE_CUBE_DIM` to 64 rather than a GPU-typical 256.

---

## What is *not* optimised

Being explicit about this, because the gap is real and the code is written so it
can be closed incrementally:

* **The scan is composed, not fused.** It is `O(T·(N+P)·chunk)` and matmul-bound,
  which is the right asymptotics, but it materialises intermediates a fused kernel
  would keep in registers. The fusion target is `ssd_chunked`; its contract
  (inputs, outputs, boundary state) is exactly what a fused kernel would need.
* **MIMO runs `R²` SISO scans** instead of one rank-aware kernel.
* **Kernels are unvectorized.** Every kernel is scalar; CubeCL's `Vector<F, N>`
  path is not used yet.
* **Broadcasting binary ops upload a small metadata buffer per call.** Same-shape
  operands take a fast path with no upload.
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
│   └── ops/             elemwise, matmul, reduce, movement, index, scan, random
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
└── infer/               state cache, samplers, generator
```

---

## References

* Mamba-3: *Improved Sequence Modeling using State Space Principles* (ICLR 2026) —
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
* Mamba-2 / SSD: *Transformers are SSMs* — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
* CubeCL — <https://github.com/tracel-ai/cubecl>

## License

MIT OR Apache-2.0.
