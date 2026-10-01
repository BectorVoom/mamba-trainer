"""Node classification on a heterophilic benchmark graph with Graph Mamba.

Takes the ``.npz`` files distributed with *A critical look at the evaluation of
GNNs under heterophily* (Platonov et al., ICLR 2023) -- ``roman_empire.npz``,
``amazon_ratings.npz``, ``minesweeper.npz``, ``tolokers.npz``, ``questions.npz``
-- which hold ``node_features`` ``[N, F]``, ``node_labels`` ``[N]``, ``edges``
``[E, 2]`` and ``train_masks`` / ``val_masks`` / ``test_masks`` ``[splits, N]``.
Nothing is downloaded: pass the path of a file you have.

    python examples/graph_node_classification.py roman_empire.npz
    python examples/graph_node_classification.py minesweeper.npz --metric roc_auc --parts 1

The graph is uploaded once. An epoch is ``parts`` optimizer steps over
stratified node partitions, queued in one call; the epoch's losses come back in
one read, and each metric in one more.
"""

import argparse
import time

import numpy as np

import mamba3_graph as mg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="a heterophilic-benchmark .npz file")
    parser.add_argument("--split", type=int, default=0, help="which of the file's splits")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--parts", type=int, default=None, help="node partitions per epoch")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "f1_macro", "roc_auc"])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--max-hops", type=int, default=4, help="the paper's m; 0 = node tokens only")
    parser.add_argument("--walks", type=int, default=8, help="the paper's M")
    parser.add_argument("--repeats", type=int, default=4, help="the paper's s")
    parser.add_argument("--node-layers", type=int, default=2)
    parser.add_argument("--mpnn", default="gine", choices=["gine", "none"])
    parser.add_argument("--pe", type=int, default=16, help="random-walk encoding width; 0 = none")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--loss-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    file = np.load(args.path)
    x = file["node_features"]
    y = file["node_labels"]
    edge_index = file["edges"].T  # a view, [2, E]; it need not be contiguous
    n = x.shape[0]
    classes = int(y.max()) + 1
    print(
        f"{args.path}: {n} nodes, {edge_index.shape[1]} edges, {x.shape[1]} features, "
        f"{classes} classes; backend {mg.backend()}, {mg.build_info()['profile']} build"
    )

    spec = mg.GraphMambaSpec(
        node_features=x.shape[1],
        task=mg.NodeClassification(classes),
        pe_dim=args.pe,
        d_model=args.d_model,
        max_hops=args.max_hops,
        walks=args.walks,
        repeats=args.repeats,
        node_layers=args.node_layers,
        mpnn=None if args.mpnn == "none" else args.mpnn,
        dropout=args.dropout,
        seed=args.seed,
    )
    arrays = dict(
        edge_index=edge_index,
        x=x,
        y=y,
        train_mask=file["train_masks"][args.split],
        val_mask=file["val_masks"][args.split],
        test_mask=file["test_masks"][args.split],
    )
    if args.pe:
        started = time.perf_counter()
        arrays["pe"] = mg.rwse(edge_index, n, args.pe)
        print(f"random-walk encoding: {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    data = mg.GraphDataset(spec, arrays, dtype=args.dtype)
    print(
        f"dataset on the device: {data.nbytes / 2**20:.1f} MiB, {data.num_edges} directed edges, "
        f"{time.perf_counter() - started:.2f} s"
    )
    model = mg.GraphMamba(
        spec,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dtype=args.dtype,
        loss_scale=args.loss_scale,
    )
    estimate = model.memory_estimate(data, parts=args.parts)
    print(
        f"{model.num_parameters} parameters; {estimate['rows']} rows per step, "
        f"about {estimate['live'] / 2**20:.0f} MiB live in a step"
    )

    best_val, test_at_best = -1.0, float("nan")
    for epoch in range(args.epochs):
        started = time.perf_counter()
        steps = model.train_epoch(data, epoch, parts=args.parts)
        losses = model.read_losses()  # one read for the whole epoch; waits for it
        seconds = time.perf_counter() - started
        val = model.evaluate(data, split="val", metric=args.metric, parts=args.parts)[args.metric]
        test = model.evaluate(data, split="test", metric=args.metric, parts=args.parts)[args.metric]
        if val > best_val:
            best_val, test_at_best = val, test
        print(
            f"epoch {epoch:3d}  loss {np.mean([entry['loss'] for entry in losses]):.4f}  "
            f"val {args.metric} {val:.4f}  test {test:.4f}  "
            f"{steps} steps in {seconds:.2f} s"
        )
    print(f"best val {args.metric} {best_val:.4f}; test at that epoch {test_at_best:.4f}")


if __name__ == "__main__":
    main()
