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
cargo test                                     # 69 tests, ~15 s
cargo run --release --example train_lm         # train, evaluate, generate, checkpoint
cargo run --release --example generate         # cached decoding + per-token timing
cargo run --release --example finetune_lora    # freeze, adapt, ship, merge
cargo run --release --example vision           # bidirectional Vision Mamba-3
cargo run --release --example bench            # where the time goes
cargo run --release --example bench_train      # tokens/s at a realistic model size
cargo run --release --example bench_matmul     # every matmul kernel, at the shapes used
cargo run --release --example bench_elemwise   # broadcasting against same-shape
cargo run --release --example vs_transformer   # head-to-head against attention
bench/compare.sh                               # head-to-head against PyTorch on ROCm
```

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
| `cuda` | NVIDIA | builds; no NVIDIA device here to run it on |

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
the candidates once per problem shape and caches the winner** — five block shapes,
the row-tiled kernel, and, for a transposed problem, the option of materialising the
transpose and using an untransposed kernel anyway. A training run issues a few dozen
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
| PyTorch 2.13 + ROCm 7.1, eager | 1000 – 1112 |
| PyTorch 2.13 + ROCm 7.1, `torch.compile` | 1143 – 1254 |
| **mamba3, ROCm** | **1224 – 1407** |
| **mamba3, Vulkan (the same GPU, through wgpu)** | **1692** |

and where that started, before any of the work above:

| | tokens/s | ms/step |
|---|---|---|
| mamba3, ROCm, before | 551 | 3720 |
| mamba3, ROCm, after | ~1300 | ~1570 |

The ranges are real and they are why the ranges are printed: this machine's load
average moved between 1 and 50 while these were being taken, and both implementations
move with it. What does not move is the ordering. Against eager PyTorch, alternating
runs came out at **1.22x, 1.28x, 1.29x, 1.45x** on four separate sessions, and
mamba3 won every individual round of every session. Against `torch.compile` it is
narrower — **1.04x to 1.14x** — and the rounds are split.

So: comfortably ahead of eager PyTorch on ROCm, ahead of `torch.compile` on ROCm by a
margin that a busy machine can swallow, and **2.4x ahead of where this crate started.**
Running the identical Rust through Vulkan instead of ROCm — the same GPU, the same
kernels, a different shader compiler — is faster again, and by more than the gap being
argued about. That row is not a compliment to this crate.

The advantage grows with the model: at `d_model` 768 it is 1.43x against eager
PyTorch, and it holds across sequence lengths (1.30x at 1024 tokens, 1.32x at 256).

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
64-token sequences:

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

* **The scan is composed, not fused.** It is `O(T·(N+P)·chunk)` and matmul-bound,
  which is the right asymptotics, but it materialises intermediates a fused kernel
  would keep in registers. The fusion target is `ssd_chunked`; its contract
  (inputs, outputs, boundary state) is exactly what a fused kernel would need.
* **MIMO runs `R²` SISO scans** instead of one rank-aware kernel.
* **Splitting the fused projection costs a full-size frame per piece.** Differentiating
  a slice means placing the gradient back where it came from and zero everywhere else,
  and the input projection is sliced five ways. Each adjoint writes a full
  `[tokens, in_proj_width]` frame and then four broadcasting adds combine them — so
  one split of a 9.5 M element tensor costs about fourteen passes over it. The bands
  are disjoint and cover the axis exactly, so the whole thing is a concatenation; what
  is missing is a way to say so, because a tape node has one output and `slice` is
  recorded five separate times. Measured at roughly a tenth of a training step, and
  the largest single item left after the matmuls.
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
  not another hand-written kernel. In the scan specifically, nine full-size
  `[batch, chunks, chunk, chunk, heads]` intermediates stand between `acum` and
  `mixing`, and one kernel could produce the last from the first.
  `backend::launch_count()` reports the running total.
* **No tensor cores.** Every matmul kernel here is a plain FMA kernel. CubeCL exposes
  `cmma`, an RDNA3 iGPU reports `min_tensor_cores_dim: 16`, and PyTorch reaches
  3.4 TFLOP/s in `f16` on this device against 385 GFLOP/s in `f32` — so the ceiling
  this is measured against is about four times higher than the one it is hitting.
  Using it needs the mixed-precision work below first.
* **ROCm is the slower of the two paths to this GPU.** The same kernels compiled
  through naga to SPIR-V and run on RADV are about 1.3x faster than the same kernels
  through hipRTC — 1692 tokens/s against 1310 on the training benchmark, and 1324
  GFLOP/s against 978 at 1024³. Nothing in this crate explains that, and it has not
  been chased.
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
└── infer/               state cache, samplers, generator

bench/
├── torch_mamba3.py      the same model in PyTorch, as the reference to beat
└── compare.sh           runs both, alternating, and reports the best step of each
```

---

## References

* Mamba-3: *Improved Sequence Modeling using State Space Principles* (ICLR 2026) —
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
* Mamba-2 / SSD: *Transformers are SSMs* — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
* CubeCL — <https://github.com/tracel-ai/cubecl>

## License

MIT OR Apache-2.0.
