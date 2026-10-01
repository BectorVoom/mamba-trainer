"""Where time goes at the Python boundary (GRAPH_MAMBA_PLAN.md §2.6).

Measures, on the installed wheel and with the entity model of
``bench_entity_training.py``: the cost of a trivial bound call, dataset ingest
against a NumPy copy of the same arrays (and the device reads it makes), how
much of a queued training step the calling thread computes versus waits, how
much a second Python thread gets to run meanwhile, and the reads ``predict``
makes. Then the same questions for the graph model of ``mamba3_graph``.

    python examples/bench_boundary.py            # both
    python examples/bench_boundary.py graph      # the graph section only
    python examples/bench_boundary.py entity     # the entity section only
"""

import threading
import time

import numpy as np

import sys

import mamba3_graph as mg
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


def second_thread_share(work):
    """Wakeups per second of a thread sleeping 1 ms in a loop: idle, and while
    ``work`` runs on the calling thread."""
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
    work()
    busy = ticks / (time.perf_counter() - start)
    stop = True
    thread.join()
    return idle, busy


def random_edges(n, count, rng):
    pairs = rng.integers(0, n, size=(2, count))
    return pairs[:, pairs[0] != pairs[1]]


def graph():
    print("graph model, backend", mg.backend(), mg.build_info()["profile"])
    rng = np.random.default_rng(0)

    # Ingest: one graph whose arrays are 200 MB (float32 features and int64 edges).
    n, features = 180_000, 256
    host = dict(
        edge_index=random_edges(n, 1_000_000, rng),
        x=rng.standard_normal((n, features), dtype=np.float32),
        y=rng.integers(0, 8, n),
    )
    megabytes = sum(v.nbytes for v in host.values()) / 1e6
    wide = mg.GraphMambaSpec(node_features=features, task=mg.NodeClassification(8))
    copy = best(lambda: {k: v.copy() for k, v in host.items()}, 5)
    mg.reset_read_count()
    mg.reset_upload_count()
    mg.GraphDataset(wide, host)
    reads, uploads = mg.read_count(), mg.upload_count()
    ingest = best(lambda: mg.GraphDataset(wide, host))
    print(
        f"ingest of {megabytes:.0f} MB: NumPy copy {copy * 1e3:.1f} ms "
        f"({megabytes / copy / 1e3:.1f} GB/s), GraphDataset {ingest * 1e3:.1f} ms "
        f"({megabytes / ingest / 1e3:.2f} GB/s), {reads} device reads, {uploads} uploads"
    )
    for name, other in [
        ("float64 features", {**host, "x": host["x"].astype(np.float64)}),
        ("Fortran-ordered features", {**host, "x": np.asfortranarray(host["x"])}),
        ("int32 edges", {**host, "edge_index": host["edge_index"].astype(np.int32)}),
    ]:
        elapsed = best(lambda: mg.GraphDataset(wide, other))
        print(f"  the same with {name}: {elapsed * 1e3:.1f} ms")
    del host

    # Training: 400 graphs of 24 nodes, ten batches an epoch.
    sizes = [24] * 400
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    edges = [random_edges(size, 3 * size, rng) + start for size, start in zip(sizes, offsets)]
    arrays_ = dict(
        edge_index=np.concatenate(edges, axis=1),
        x=rng.standard_normal((offsets[-1], 16), dtype=np.float32),
        y=np.arange(len(sizes)) % 4,
        graph_ptr=offsets,
    )
    small = mg.GraphMambaSpec(node_features=16, task=mg.GraphClassification(4), mpnn="gine")
    data = mg.GraphDataset(small, arrays_)
    model = mg.GraphMamba(small)
    rows = 960

    def epoch(index):
        steps = model.train_epoch(data, index, batch_rows=rows)
        model.read_losses()
        return steps

    epoch(0)
    epoch(1)
    mg.reset_read_count()
    mg.reset_upload_count()
    wall0, cpu0 = time.perf_counter(), time.thread_time()
    steps = model.train_epoch(data, 2, batch_rows=rows)
    wall1, cpu1 = time.perf_counter(), time.thread_time()
    queue_reads, queue_uploads = mg.read_count(), mg.upload_count()
    model.read_losses()
    wall2, cpu2 = time.perf_counter(), time.thread_time()
    print(
        f"train_epoch, {steps} steps of {rows} rows: queue {(wall1 - wall0) / steps * 1e3:.1f} "
        f"ms/step wall, {(cpu1 - cpu0) / steps * 1e3:.1f} ms/step CPU on the calling thread, "
        f"{queue_reads} reads, {queue_uploads} uploads; read_losses "
        f"{(wall2 - wall1) / steps * 1e3:.1f} ms/step wall, "
        f"{(cpu2 - cpu1) / steps * 1e3:.1f} ms/step CPU, {mg.read_count() - queue_reads} read"
    )

    idle, busy = second_thread_share(lambda: [epoch(index) for index in range(3, 6)])
    print(
        f"a second Python thread: {idle:.0f} wakeups/s idle, {busy:.0f} during training "
        f"({busy / idle * 100:.1f}%)"
    )

    model.predict(data, batch_rows=rows)
    mg.reset_read_count()
    model.predict(data, batch_rows=rows)
    reads = mg.read_count()
    elapsed = best(lambda: model.predict(data, batch_rows=rows), 5)
    print(f"predict, {len(sizes)} graphs in {steps} batches: {elapsed * 1e3:.1f} ms, {reads} device reads")
    mg.reset_read_count()
    model.evaluate(data, split="train", batch_rows=rows)
    print(f"evaluate: {mg.read_count()} device reads")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "entity"):
        main()
    if which in ("both", "graph"):
        graph()
