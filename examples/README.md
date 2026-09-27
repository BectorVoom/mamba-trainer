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
