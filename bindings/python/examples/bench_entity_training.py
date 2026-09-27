"""Time EntityModel training through the Python binding.

Kaggriculture-shaped spec (ENTITY_MODEL_PLAN.md §1.4): 100 grid tiles, 20
units x 3 steps, d_model 128, 3 + 3 layers. Reports milliseconds per
training step for the loop a user writes — ``queue_train_step`` for
``--steps`` batches, then one ``read_losses`` — so the per-step cost includes
whatever ``read_losses`` does for each queued step.

    python examples/bench_entity_training.py --batch 128 --steps 10 --rounds 3
"""

import argparse
import time

import numpy as np

import mamba3_rl as m3

N, U, K, Q = 100, 20, 3, 60


def spec():
    return m3.EntityModelSpec(
        globals=114,
        context=[m3.ContextSet("tiles", count=N, features=48, layout=m3.Grid(10, 10))],
        queries=m3.QuerySet(
            "units", count=U, features=36, anchor="tiles", steps=K, autoregressive_on="target"
        ),
        heads=[
            m3.Head.pointer("target", set="tiles", extra_actions=1, step_weights=[1.0, 0.5, 0.5]),
            m3.Head.categorical("op", classes=13, condition_on="target"),
            m3.Head.multilabel("opset", labels=13, condition_on="target", loss_weight=0.3),
            m3.Head.categorical("crop", classes=5, condition_on="target", loss_weight=0.3),
            m3.Head.regression("eta", outputs=1, steps="first", loss_weight=0.1),
        ],
        d_model=128,
        context_layers=3,
        decoder_layers=3,
        seed=0,
    )


def arrays(samples, seed=99):
    rng = np.random.default_rng(seed)
    anchor = np.where(rng.random((samples, U)) < 0.8, rng.integers(0, N, (samples, U)), -1)
    r = rng.random((samples, U, K))
    tgt = np.where(r < 0.7, rng.integers(0, N, (samples, U, K)), np.where(r < 0.8, N, -1))
    labelled = r < 0.7
    op = np.where(labelled, rng.integers(0, 13, (samples, U, K)), -1)
    crop = np.where(labelled, rng.integers(0, 5, (samples, U, K)), -1)
    opset = np.zeros((samples, U, K, 13), dtype=np.float32)
    opset[labelled, rng.integers(0, 13, labelled.sum())] = 1.0
    eta = np.where(anchor >= 0, rng.integers(0, 20, (samples, U)), np.nan)
    return {
        "tiles": rng.uniform(-1, 1, (samples, N, 48)).astype(np.float32),
        "globals": rng.uniform(-1, 1, (samples, 114)).astype(np.float32),
        "units": rng.uniform(-1, 1, (samples, U, 36)).astype(np.float32),
        "units.anchor": anchor.astype(np.int64),
        "label.target": tgt.astype(np.int64),
        "label.op": op.astype(np.int64),
        "label.opset": opset,
        "label.crop": crop.astype(np.int64),
        "label.eta": eta.astype(np.float32).reshape(samples, U, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10, help="steps queued per read_losses")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    s = spec()
    data = m3.EntityDataset(s, arrays(args.batch * 4))
    model = m3.EntityModel(s)
    rng = np.random.default_rng(0)
    batches = [rng.permutation(args.batch * 4)[: args.batch] for _ in range(args.steps)]

    def round_():
        for ids in batches:
            model.queue_train_step(data, ids)
        return model.read_losses()

    round_()  # warm up: kernel compilation, matmul autotuning, allocator growth
    times = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        out = round_()
        times.append((time.perf_counter() - start) * 1000 / args.steps)
    print(
        f"batch {args.batch}: {np.median(times):.1f} ms/step "
        f"(rounds {', '.join(f'{t:.1f}' for t in times)}); last loss {out[-1]['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
