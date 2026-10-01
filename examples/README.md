# Examples

## Profiling with an honest timer

`profile_entity_model` brackets every timed stage with a read-based drain:

```text
drain = synchronize() + read one scalar tensor
```

A bare `synchronize()` can return once the queue is submitted while kernels are
still running (with few timed steps the queue absorbs the launches, so 2 steps
report a much smaller ms/step than 8 for the same work). A device-to-host read
is the only operation guaranteed to wait for every queued kernel, so the stage
is drained with a 1-element read before starting the timer and again before
taking `elapsed()`. The cost of the two drain reads themselves is measured once
with a bare probe (`bare drain read: … ms`) and subtracted from the reported
ms/step. Compare step counts (e.g. `MAMBA3_ENTITY_STEPS=2` vs `=8`): after the
fix they agree within ~15%.

## Graph Mamba

`train_graph` trains [Graph Mamba](../README.md#graph-mamba) end to end on two
synthetic tasks whose answer is known, and prints the loss, the validation
metric, the time and the launches per step for every epoch:

```bash
cargo run --release --example train_graph                               # neighbour majority: node classification
MAMBA3_GRAPH_TASK=triangles cargo run --release --example train_graph   # triangle counts: graph regression
MAMBA3_GRAPH_HOPS=0 cargo run --release --example train_graph           # node tokens only: stays at chance
```

`MAMBA3_GRAPH_{HOPS,WALKS,REPEATS,D_MODEL,NODE_LAYERS,TOKEN_LAYERS,MPNN,LOCAL,LR,
EPOCHS,PARTS,BATCH_ROWS,NODES,GRAPHS}` override the defaults.

`profile_graph` attributes a training step of the two reference workloads
(`b`, the default: many small graphs of about 150 nodes batched whole to 4,800
rows, with GINE; `a`: one large graph of 22,662 nodes with 300 features, in four
node partitions):

```bash
cargo run --release --example profile_graph                            # workload b
MAMBA3_GRAPH_WORKLOAD=a cargo run --release --example profile_graph
MAMBA3_GRAPH_SCALE=0.25 cargo run --release --example profile_graph    # a quarter of the size
```

It reports three clocks that must not be mixed — *host submit* (the time to
queue a step), *drained step* (queue, then one read) and, with a `cubecl.toml`
profiling logger, device time per kernel — plus launches by region
(`graph.data`, `graph.embed`, `graph.local`, `graph.token`, `graph.node`,
`graph.mpnn`, `graph.head`, the backward pass, the optimizer), uploads and
reads per step, tuner keys and misses, and the memory estimate against the
measured peak. Host submit close to the drained step means the step is
host-bound, and a faster kernel will not help it.

`bench_graph_kernels` prices the indirection in the two kernels that gather
through an index, each against a contiguous baseline of the same work:

```bash
cargo run --release --example bench_graph_kernels
ROWS=4800 cargo run --release --example bench_graph_kernels
```
