# MS2 representation v2

The revised Python experiment retains distinctions that the v1 forward target,
pooled graph encoder and dataset selection discarded. Its version is
`continuous-forward-canonical-graph-v2`.

## Changes

- Forward targets combine the existing 2,000 coarse mass bins with 2,048 signed
  continuous-mass Fourier features, using fixed seeded frequencies at 0.01 and
  0.001 Da kernel scales. Nearby peaks in the same integer bin now have distinct
  targets. The output head is signed rather than Softplus. This is a finite
  spectral sketch, not a lossless spectrum or a guaranteed resolution bound.
- Graph inputs retain a full padded canonical atom/bond representation in
  addition to the invariant message-passing branch. It includes parent atom
  types, all bond orders, canonical local chiral parity and E/Z annotations.
  Full canonical connectivity avoids the specific collision caused by identical
  local messages and pooled counts. Learned embeddings can still lose distinctions;
  the input representation and embedding should not be conflated.
- Parsed graphs retain canonical isomeric SMILES separately from the benchmark
  connectivity grouping key. Incomplete motifs retain their parent hydrogen
  types through internal serialization colors; these are not accepted isotope
  inputs. Charged, isotopic and radical molecules remain outside the domain.
- Data loading retains measurements of the same molecule under different
  conditions. Deduplication uses connectivity, stereo annotation, peak content,
  adduct, precursor and supplied condition metadata. `max_train` now caps unique
  connectivity identities; the record count can be greater. Scaffold/identity
  split exclusions still keep measurements of one molecule together.
- Conditional training includes a differentiable same-formula candidate loss
  (weight 0.2): spectrum–graph alignment at temperature 0.1 plus candidate graph
  likelihood cross-entropy. Candidates contain the true graph first and up to
  four observed/rewired alternatives. Both encoders and the graph decoder receive
  gradients. The final frozen-feature reranker remains a separate subsequent
  stage. Records without an alternative receive no candidate loss.
- Checkpoints record the representation version. Restore rejects historical
  v1 checkpoints with an explicit retraining instruction. The full T4 notebook
  embeds revised sources and regression tests; old experiment artifacts remain
  unchanged.

## Validation and limits

The final 23-test suite passed locally (21 passed, two CUDA skips) and on an
actual Colab Tesla T4 via `colab exec` (22 passed, one Rust-executable skip).
The local run executed and passed native Rust/Python grammar parity; the T4
run executed CPU/CUDA likelihood and forward-head parity, CUDA learning and
the new CUDA same-formula learning check. Logs are saved under
`experiments/molecular_completion/20261004_representation_fix/`.

Regression tests cover sub-Dalton target differences, intensity-scale invariance,
atom-order invariance, a triangular prism versus K3,3 collision of the old graph
encoder, enantiomer and E/Z input differences, retained collision-energy
measurements, leakage exclusions, old-checkpoint rejection, and paired-loss
gradients plus synthetic learning on CPU/CUDA. The existing CPU/CUDA numerical
parity and Python/Rust grammar parity checks also remain applicable.

The revised encoder and forward target are implemented only in the Python
experiment. Native Rust parity covers the unchanged connectivity grammar, not
these new neural features. Stereo annotations are preserved for encoding; the
decoder still generates connectivity without stereo assignments. Peak filtering
and the top-160 limit remain. A finite embedding/sketch cannot guarantee that
every possible input is distinguishable, or that MS2 determines stereochemistry.

Regression and synthetic training results do not establish improved benchmark
identification or physical stability. New weights and a full benchmark retraining
are required before reporting a revised identification score. Historical v1
scores are recorded in `MS2_FULL_T4_VERIFICATION.md`.
