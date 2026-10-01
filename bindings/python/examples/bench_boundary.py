"""Where time goes at the Python boundary (GRAPH_MAMBA_PLAN.md §2.6).

Measures, on the installed wheel and with the entity model of
``bench_entity_training.py``: the cost of a trivial bound call, dataset ingest
against a NumPy copy of the same arrays (and the device reads it makes), how
much of a queued training step the calling thread computes versus waits, how
much a second Python thread gets to run meanwhile, and the reads ``predict``
makes.

    python examples/bench_boundary.py
"""

import threading
import time

import numpy as np

import mamba3_rl as m3
from bench_entity_training import arrays, spec

BATCH = 32
STEPS = 10


def best(f, n=3):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    print("backend", m3.backend())

    calls = 1_000_000
    start = time.perf_counter()
    for _ in range(calls):
        m3.launch_count()
    print(f"trivial bound call: {(time.perf_counter() - start) / calls * 1e9:.0f} ns")

    s = spec()
    host = arrays(2048)
    megabytes = sum(v.nbytes for v in host.values()) / 1e6
    copy = best(lambda: {k: v.copy() for k, v in host.items()}, 5)
    m3.reset_read_count()
    m3.EntityDataset(s, host)
    reads = m3.read_count()
    ingest = best(lambda: m3.EntityDataset(s, host))
    print(
        f"ingest of {megabytes:.0f} MB: NumPy copy {copy * 1e3:.1f} ms "
        f"({megabytes / copy / 1e3:.1f} GB/s), EntityDataset {ingest * 1e3:.1f} ms "
        f"({megabytes / ingest / 1e3:.2f} GB/s), {reads} device reads"
    )

    data = m3.EntityDataset(s, arrays(BATCH * 4))
    model = m3.EntityModel(s)
    rng = np.random.default_rng(0)
    batches = [rng.permutation(BATCH * 4)[:BATCH] for _ in range(STEPS)]

    def round_():
        for ids in batches:
            model.queue_train_step(data, ids)
        return model.read_losses()

    round_()
    round_()
    wall0, cpu0 = time.perf_counter(), time.thread_time()
    for ids in batches:
        model.queue_train_step(data, ids)
    wall1, cpu1 = time.perf_counter(), time.thread_time()
    model.read_losses()
    wall2, cpu2 = time.perf_counter(), time.thread_time()
    print(
        f"train batch {BATCH}: queue {(wall1 - wall0) / STEPS * 1e3:.1f} ms/step wall, "
        f"{(cpu1 - cpu0) / STEPS * 1e3:.1f} ms/step CPU on the calling thread; "
        f"read {(wall2 - wall1) / STEPS * 1e3:.1f} ms/step wall, "
        f"{(cpu2 - cpu1) / STEPS * 1e3:.1f} ms/step CPU"
    )

    stop = False
    ticks = 0

    def spin():
        nonlocal ticks
        while not stop:
            ticks += 1
            time.sleep(0.001)

    thread = threading.Thread(target=spin)
    thread.start()
    time.sleep(0.5)
    idle = ticks / 0.5
    ticks = 0
    start = time.perf_counter()
    round_()
    busy = ticks / (time.perf_counter() - start)
    stop = True
    thread.join()
    print(
        f"a second Python thread: {idle:.0f} wakeups/s idle, {busy:.0f} during training "
        f"({busy / idle * 100:.1f}%)"
    )

    obs = {k: v[:BATCH] for k, v in arrays(BATCH).items() if not k.startswith("label.")}
    model.predict(obs)
    m3.reset_read_count()
    model.predict(obs)
    reads = m3.read_count()
    elapsed = best(lambda: model.predict(obs), 5)
    print(f"predict batch {BATCH}: {elapsed * 1e3:.1f} ms, {reads} device reads")


if __name__ == "__main__":
    main()
