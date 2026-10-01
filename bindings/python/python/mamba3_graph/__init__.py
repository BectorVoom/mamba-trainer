"""Graph Mamba Networks on Mamba-3, on the GPU.

*Graph Mamba: Towards Learning on Graphs with State Space Models* (Behrouz and
Hashemi, KDD 2024) as a model of this library. A graph model in two stages, both
made of the bidirectional Mamba-3 block:

1. **Per node.** Each node gets a short sequence of subgraph tokens -- samples of
   its 1-hop, 2-hop, ... ``max_hops``-hop neighbourhood drawn with random walks.
   A Mamba scans the sequence from the farthest neighbourhood inwards; the last
   position, the node itself, is the node's encoding.
2. **Per graph.** The node encodings of a graph, ordered by degree, form one
   sequence, scanned in both directions and optionally summed with a
   message-passing layer over the real edges.

A dataset is uploaded once. After that, sampling the walks, building the tokens,
assembling a batch and computing the loss are device kernels: a training step
uploads nothing and reads nothing.

A run, end to end::

    import mamba3_graph as mg

    spec = mg.GraphMambaSpec(node_features=300, task=mg.NodeClassification(18),
                             pe_dim=16, mpnn="gine")
    data = mg.GraphDataset(spec, dict(
        edge_index=edge_index,                      # int[2, E]
        x=x, y=y,                                   # float[N, 300], int[N]
        train_mask=train, val_mask=val, test_mask=test,
        pe=mg.rwse(edge_index, x.shape[0], 16)))
    model = mg.GraphMamba(spec, learning_rate=1e-3)

    for epoch in range(100):
        model.train_epoch(data, epoch)              # queued on the device
        losses = model.read_losses()                # one read for the epoch
        print(epoch, losses[-1]["loss"], model.evaluate(data, split="val"))

    predictions = model.predict(data)               # float32[N, 18], your node order

This module and :mod:`mamba3_rl` are two import names of one extension library:
they share one device, one set of counters and one learning-rate schedule class.
"""

from mamba3_rl._mamba3_rl import (
    Categorical,
    GraphClassification,
    GraphDataset,
    GraphMamba,
    GraphMambaSpec,
    GraphMultiLabel,
    GraphRegression,
    GraphTask,
    LrSchedule,
    NodeClassification,
    __version__,
    _warn_if_debug_build,
    backend,
    build_info,
    laplacian_pe,
    launch_count,
    read_count,
    reset_launch_count,
    reset_read_count,
    reset_upload_count,
    rwse,
    supports_dtype,
    synchronize,
    upload_count,
)

__all__ = [
    "Categorical",
    "GraphClassification",
    "GraphDataset",
    "GraphMamba",
    "GraphMambaSpec",
    "GraphMultiLabel",
    "GraphRegression",
    "GraphTask",
    "LrSchedule",
    "NodeClassification",
    "__version__",
    "backend",
    "build_info",
    "laplacian_pe",
    "launch_count",
    "read_count",
    "reset_launch_count",
    "reset_read_count",
    "reset_upload_count",
    "rwse",
    "supports_dtype",
    "synchronize",
    "upload_count",
]
