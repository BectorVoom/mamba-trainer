# The Python boundary of the existing bindings (GRAPH_MAMBA_PLAN.md §2.6)

Measured 2026-10-01 on the Apple M1 with `bindings/python/examples/bench_boundary.py`, on the release wheel installed
in `bindings/python/.venv-rl-wgpu` (wgpu, built 2026-09-29, Python 3.14, NumPy 2.5), with the entity model and data of
`bench_entity_training.py` (batch 32). These are the existing `EntityDataset` / `EntityModel` bindings, measured to
decide how the graph bindings should cross the boundary; nothing here is a graph-model number.

| what | measured |
|---|---|
| a trivial bound call (`m3.launch_count()`) | 29–46 ns |
| NumPy copy of 56 MB of input arrays | 2.0 ms (27.7 GB/s) |
| `EntityDataset(spec, arrays)` on the same 56 MB | 171–188 ms (0.30–0.33 GB/s), **28 device reads** |
| the same with float64 inputs / Fortran-ordered inputs | 170 ms / 185 ms: the dtype conversion is not the cost |
| `queue_train_step`, per step | 255 ms wall, **37 ms CPU on the calling thread** |
| `read_losses`, per queued step | 18 ms wall, 0.0 ms CPU |
| a second Python thread while training | 4 wakeups/s against 830 idle: **0.5%** |
| `predict`, batch 32 | 167 ms, **9 device reads** |

## What it says

1. **Crossing the boundary costs nothing**: tens of nanoseconds per call. The number of Python calls is not a lever.
2. **The calling thread mostly waits, and it waits holding the GIL.** Of 255 ms per queued step, 37 ms is this
   thread's CPU; the rest is waiting on the device. No other Python thread runs meanwhile (0.5% of its idle rate).
3. **Ingest is 90x slower than a copy, and it reads from the device.** `EntityDataset::from_arrays` uploads each
   array and reads tensors back to derive its tables (`src/models/entity/batch.rs:929`): 28 reads for one dataset.
   Input dtype and memory order make no difference.
4. **`predict` reads 9 times per call** (one per decode step and head), at ~1.4 ms each plus a drain.

## A bug found on the way: Fortran-ordered inputs are read scrambled

`PyReadonlyArray::as_slice` (numpy crate 0.29.0) succeeds for C-contiguous **and** Fortran-contiguous arrays and
returns the bytes in memory order. `bindings/python/src/array.rs:46` and `bindings/python/src/entity_model.rs:139`
take that slice whenever it succeeds and treat it as row-major. With the same logical contents:

| input arrays | first-step loss, same seed |
|---|---|
| C-ordered | 21.409803 |
| a strided view (falls back to `as_array()`, logical order) | 21.409803 |
| `np.asfortranarray(...)` | 21.524588 |

No error is raised. Not fixed here. The fix is to use the slice only when `is_c_contiguous()`.

## What it does not say

- Whether a debug build of the bindings is slower, and by how much: the in-tree `python/mamba3_rl/*.so` files
  (167 MB each, debug) fail to load (`mis-aligned LINKEDIT string pool`), so they could not be measured.
- Anything about CPU-only builds, other GPUs, or free-threaded Python.
- One machine, one run per row except where a range is given.

---

# The graph bindings, built to those findings (`mamba3_graph`, GM9)

Measured 2026-10-01 with `python examples/bench_boundary.py graph` on the **cpu** release wheel (AMD Ryzen AI 7 350,
16 threads, Linux, Python 3.14, NumPy 2.5), two runs. This is a different machine and a different backend from the
entity rows above, so the times are not comparable with them; the counts and the second thread's share are.

| what | measured |
|---|---|
| NumPy copy of 202 MB (180,000 × 256 `float32` features, 1 M `int64` edges, labels) | 11.8–12.2 ms (16.6–17.1 GB/s) |
| `GraphDataset(spec, arrays)` on the same 202 MB | 267–274 ms (**0.74–0.76 GB/s**), **0 device reads**, 5 uploads |
| the same with `float64` features / `int32` edges | 260–270 ms / 257–263 ms |
| the same with Fortran-ordered features | 303 ms (875–927 ms before the copy was handed to NumPy, see 2) |
| `train_epoch`, 10 steps of 960 rows (400 graphs of 24 nodes, GINE), per step | 136–140 ms wall, 126–128 ms CPU on the calling thread, **0 reads, 1 upload for the epoch** |
| `read_losses` for those 10 steps | 1 read, < 0.1 ms per step |
| a second Python thread while training | 892–896 wakeups/s against 950 idle: **94%** |
| `predict`, 400 graphs in 10 batches | 500 ms, **1 device read** |
| `evaluate` | **1 device read** |

## What it says

1. **Ingest reads nothing and reaches 0.75 GB/s**, 2.3x the entity bindings' rate on a machine whose plain copy is
   slower. The 1 GB/s of P4 is not reached: the time is the canonicalisation (symmetrising and sorting 1 M edges,
   and writing the features in degree order, which is a gather rather than a copy), not the upload.
2. **Input dtype is free; memory order costs one copy.** `float64` features and `int32` edges cost the same as the
   native ones, because the one pass that reorders also converts. A Fortran-ordered, sliced or unaligned array is
   first copied into C order (the alternative, `as_slice` on it, is the scrambled read described above, or for an
   unaligned one no slice at all). The first version made that copy element by element through an `ndarray`
   iterator and took 875–927 ms for the 184 MB feature array; `numpy.require(a, requirements="CA")` makes the same
   copy in 40–50 ms, so a Fortran-ordered dataset now ingests in 303 ms against 261 ms for a C-ordered one.
3. **Another Python thread keeps running**: 94% of its idle rate, against 0.5% for the entity bindings, although on
   the cpu backend the calling thread computes for the whole step (126 of 136 ms). On a GPU the calling thread
   mostly waits, which is the case P2 was written for; that run is still owed (Mac wgpu).
4. **One read per unit of work**: none while an epoch is queued, one for all of its losses, one per `predict` and
   one per `evaluate`, whatever the number of batches.

## What it does not say

- Anything about a GPU: every graph row is the cpu backend.
- How long a debug build takes. `build_info()["profile"]` reports it and constructing a model warns; not timed.
- Free-threaded Python: the classes are `unsendable` and were only run on the default build.
