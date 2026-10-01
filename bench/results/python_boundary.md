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
