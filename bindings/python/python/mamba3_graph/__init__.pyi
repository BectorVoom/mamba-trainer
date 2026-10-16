"""Type stubs for :mod:`mamba3_graph`."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from mamba3_rl._mamba3_rl import (
    LrSchedule as LrSchedule,
    __version__ as __version__,
    backend as backend,
    launch_count as launch_count,
    read_count as read_count,
    reset_launch_count as reset_launch_count,
    reset_read_count as reset_read_count,
    synchronize as synchronize,
)

Dtype = Literal["f32", "bf16"]
Pool = Literal["mean", "sum"]
Split = Literal["train", "val", "test"]
MetricName = Literal["accuracy", "f1_macro", "mae", "mse", "ap", "roc_auc"]

def upload_count() -> int:
    """Buffers created from host data since the counter was last reset."""

def reset_upload_count() -> None:
    """Reset the counter :func:`upload_count` reports."""

def supports_dtype(dtype: str) -> bool:
    """Whether the compiled backend can store and compute ``dtype``."""

def build_info() -> Dict[str, str]:
    """How the extension was built: ``{"profile": "release" | "debug",
    "backend", "version"}``."""

def _warn_if_debug_build(debug: Optional[bool] = None) -> None:
    """Raise a ``RuntimeWarning`` if the extension is a debug build (or if
    ``debug`` is true)."""

class Categorical:
    """Categorical features: one integer id per field, field ``f`` taking
    values in ``0..vocab[f]``. The model reads them as one multi-hot."""

    def __init__(self, vocab: Sequence[int]) -> None: ...
    @property
    def vocab(self) -> List[int]: ...

class GraphTask:
    """What a model predicts; build one with :func:`NodeClassification`,
    :func:`GraphClassification`, :func:`GraphRegression` or
    :func:`GraphMultiLabel`."""

    @property
    def outputs(self) -> int:
        """Width of the model's output."""
    @property
    def per_graph(self) -> bool:
        """Whether the model predicts per graph rather than per node."""

def NodeClassification(classes: int) -> GraphTask:
    """One of ``classes`` per node."""

def GraphClassification(classes: int, *, pool: Pool = "mean") -> GraphTask:
    """One of ``classes`` per graph, read out of the pooled nodes."""

def GraphRegression(
    targets: int, *, pool: Pool = "mean", loss: Literal["l1", "mse"] = "l1"
) -> GraphTask:
    """``targets`` floats per graph."""

def GraphMultiLabel(labels: int, *, pool: Pool = "mean") -> GraphTask:
    """``labels`` independent binary labels per graph."""

class GraphMambaSpec:
    """A Graph Mamba model, completely. A ``ValueError`` names the argument at
    fault.

    ``node_features`` and ``edge_features`` are an ``int`` (floats per row) or
    a :class:`Categorical`. ``max_hops``, ``walks`` and ``repeats`` are the
    paper's ``m``, ``M`` and ``s``: a node's sequence holds ``repeats`` tokens
    for each walk length ``1..max_hops`` plus the node itself, and each token
    is the set of nodes ``walks`` random walks visit. ``max_hops=0`` gives node
    tokens only (GPS with the Transformer replaced by bidirectional Mamba).
    ``token_layers`` defaults to 1 with walk tokens and 0 without; heads are
    per direction; ``pe_sign_flip`` is a ``(start, end)`` range of encoding
    columns whose sign flips per graph while training."""

    def __init__(
        self,
        *,
        node_features: Union[int, Categorical],
        task: GraphTask,
        edge_features: Union[int, Categorical, None] = None,
        pe_dim: int = 0,
        pe_sign_flip: Optional[Tuple[int, int]] = None,
        d_model: int = 64,
        max_hops: int = 4,
        walks: int = 8,
        repeats: int = 4,
        token_sampling: Literal["step", "epoch", "static"] = "step",
        local: Literal["mean", "sgc"] = "sgc",
        token_layers: Optional[int] = None,
        token_tail: Literal["forward", "bidirectional"] = "forward",
        node_layers: int = 2,
        mpnn: Optional[Literal["gine", "gated_gcn", "none"]] = None,
        d_state: int = 8,
        token_heads: int = 1,
        node_heads: int = 1,
        node_sequences: int = 1,
        bidirectional: bool = True,
        order: Literal["degree", "degree_desc", "ppr", "kcore", "given"] = "degree",
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> GraphMambaSpec: ...
    @property
    def d_model(self) -> int: ...
    @property
    def tokens_per_node(self) -> int:
        """``max_hops * repeats + 1``."""
    @property
    def num_parameters(self) -> int:
        """Scalars a model built from this spec holds."""
    @property
    def task(self) -> GraphTask: ...

class GraphDataset:
    """A graph dataset on the device: validated against its spec, put in
    canonical (degree) order and uploaded once.

    ``arrays`` is a dict: ``edge_index`` ``int[2, E]`` (sources, then
    destinations), ``x`` ``float[N, F]`` (or ``int[N, fields]`` for categorical
    features), and optionally ``edge_attr`` (one row per edge), ``y``
    (``int[N]`` for node classes with ``-1`` unlabelled, ``int[G]`` for graph
    classes, ``float[G, targets]`` with ``NaN`` missing otherwise),
    ``graph_ptr`` ``int[G + 1]`` (node offsets; one graph when omitted),
    ``train_mask`` / ``val_mask`` / ``test_mask`` (one flag per node for node
    targets, per graph for graph targets) and ``pe`` ``float[N, pe_dim]``.

    Floats may be ``float16``, ``float32`` or ``float64`` and integers any
    width; arrays need not be C-contiguous and are not modified. Unknown keys,
    wrong shapes and out-of-range ids raise ``ValueError`` naming the key.
    Nothing is read back from the device."""

    def __init__(
        self,
        spec: GraphMambaSpec,
        arrays: Dict[str, object],
        *,
        symmetrize: bool = True,
        dtype: Dtype = "f32",
    ) -> None: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_graphs(self) -> int: ...
    @property
    def num_edges(self) -> int:
        """Directed edges after symmetrising and removing duplicates."""
    @property
    def nbytes(self) -> int:
        """Bytes the dataset's tables take on the device."""
    @property
    def dtype(self) -> Dtype: ...

def rwse(
    edge_index: np.ndarray, num_nodes: int, k: int, *, max_ball: int = 200000
) -> np.ndarray:
    """Random-walk structural encoding, ``float32[num_nodes, k]``: column
    ``j - 1`` is the probability that a ``j``-step random walk returns to its
    start. Exact; refuses a ``k``-hop neighbourhood above ``max_ball`` nodes."""

def laplacian_pe(
    edge_index: np.ndarray,
    num_nodes: int,
    k: int,
    *,
    graph_ptr: Optional[np.ndarray] = None,
    max_nodes: int = 2048,
) -> np.ndarray:
    """Laplacian positional encoding, ``float32[num_nodes, k]``: the ``k``
    eigenvectors of each graph's normalised Laplacian with the smallest
    non-zero eigenvalues (zero columns where a graph has fewer). A dense
    solver: graphs above ``max_nodes`` are refused; use :func:`rwse`."""

class GraphMamba:
    """Graph Mamba with its AdamW optimizer.

    ``loss_scale`` multiplies the loss before it is differentiated, which keeps
    a 16-bit model's gradients above underflow; reported losses and gradient
    norms are divided back. The dataset and the model must share a ``dtype``:
    ``"f32"``, or ``"bf16"`` where the backend has it (:func:`supports_dtype`),
    which stores weights, moments and activations in 16 bits. ``"f16"`` raises
    ``NotImplementedError``: the mixer's backward pass is not finite in f16
    in this build. A debug build raises a ``RuntimeWarning``."""

    def __init__(
        self,
        spec: GraphMambaSpec,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        lr_schedule: Optional[LrSchedule] = None,
        dtype: Dtype = "f32",
        loss_scale: Optional[float] = None,
    ) -> None: ...
    def train_epoch(
        self,
        data: GraphDataset,
        epoch: int,
        *,
        batch_rows: Optional[int] = None,
        parts: Optional[int] = None,
    ) -> int:
        """Queue every training step of one epoch; returns how many were
        queued. Nothing is read back and only the epoch's own table is
        uploaded. ``batch_rows`` is the row budget of whole-graph batches,
        ``parts`` the number of node partitions of one large graph (1 is full
        batch); with neither, a graph-level task or a dataset of several graphs
        trains on whole graphs and one large graph on node partitions, both
        sized automatically. A model that cannot run on node partitions
        (``mpnn="gated_gcn"``, edge features) uses the whole graph. The interpreter lock is released for the whole
        call, and Ctrl-C stops it after a completed step."""
    def read_losses(self) -> List[Dict[str, float]]:
        """The report of every step queued since the last call, oldest first:
        ``step``, ``loss``, ``grad_norm``, ``learning_rate``. One device read."""
    def evaluate(
        self,
        data: GraphDataset,
        *,
        split: Split = "val",
        metric: MetricName = "accuracy",
        batch_rows: Optional[int] = None,
        parts: Optional[int] = None,
    ) -> Dict[str, float]:
        """``{metric: value, "count": targets counted}``. The model runs over
        every node or graph; the split only selects what is counted. One
        device read."""
    def predict(
        self,
        data: GraphDataset,
        *,
        split: Optional[Split] = None,
        batch_rows: Optional[int] = None,
        parts: Optional[int] = None,
    ) -> np.ndarray:
        """``float32[items, outputs]`` for every node (node tasks) or graph
        (graph tasks), in the order the dataset was given in. ``split``
        returns only the rows of the items in that mask of the dataset, in the
        same order; the model runs over every node either way. One device
        read."""
    def memory_estimate(
        self,
        data: GraphDataset,
        *,
        batch_rows: Optional[int] = None,
        parts: Optional[int] = None,
    ) -> Dict[str, int]:
        """Bytes a step is expected to hold on the device: ``store``,
        ``parameters``, ``live``, ``largest_allocation``, and the ``rows`` and
        ``threshold`` the estimate is for."""
    def save(self, path: str, step: Optional[int] = None) -> None: ...
    @staticmethod
    def load(
        path: str,
        *,
        dtype: Dtype = "f32",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        lr_schedule: Optional[LrSchedule] = None,
        loss_scale: Optional[float] = None,
    ) -> GraphMamba:
        """The weights and the spec, with a fresh optimizer."""
    @property
    def num_parameters(self) -> int: ...
    @property
    def step(self) -> int:
        """Optimizer steps taken so far."""
    @property
    def spec(self) -> GraphMambaSpec: ...
    @property
    def dtype(self) -> Dtype: ...

IntArray = Union[np.ndarray, Sequence[int]]

class MatchQueries:
    """Every query vector of a run, uploaded once: ``bits`` (integers) and
    ``weights`` (floats), both ``[queries, k]`` with ``k <= 256``. A weight of
    0 is padding, whatever its bit; every other entry names a bit below
    ``n_bits`` with a finite weight, and a bit named twice adds its weights."""

    def __init__(self, bits: np.ndarray, weights: np.ndarray, n_bits: int = 4096) -> None: ...
    @property
    def num_queries(self) -> int: ...
    @property
    def k(self) -> int:
        """Entries per query row."""
    @property
    def n_bits(self) -> int: ...
    def __len__(self) -> int: ...

class MatchGroups:
    """The groups of a run: ``query`` ``[G]`` (query ids), ``member_ptr``
    ``[G + 1]`` (CSR offsets from 0) and ``members`` ``[M]`` (dataset graph
    ids), answer first in each group, and optionally ``offsets`` ``[M]``
    (finite floats aligned with ``members``): the fixed term a residual
    matcher adds to each member's logit. Repeated members of a group are
    dropped (the first occurrence is kept with its offset); every group needs
    at least one member."""

    def __init__(
        self,
        query: IntArray,
        member_ptr: IntArray,
        members: IntArray,
        offsets: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> None: ...
    @property
    def num_groups(self) -> int: ...
    @property
    def has_offsets(self) -> bool: ...
    def offsets(self, g: int) -> Optional[np.ndarray]:
        """The offsets of the distinct members of group ``g``, answer first;
        ``None`` without offsets."""
    def trivial(self) -> np.ndarray:
        """``uint32`` ids of the groups with fewer than two distinct members:
        refused by :meth:`GraphMatchTrainer.train_epoch`, accepted by
        :meth:`GraphMatchTrainer.score_groups`."""
    def members(self, g: int) -> np.ndarray:
        """The distinct members of group ``g``, answer first."""
    def query(self, g: int) -> int: ...
    def __len__(self) -> int: ...

class GraphMatchTrainer:
    """A graph matcher with its AdamW optimizer, in ``float32``.

    Graph Mamba (``spec``, which needs a graph-level task such as
    ``GraphRegression(d_out, pool="mean")``; its loss is not used) embeds each
    member graph; a query encoder (a bias-free projection of the bag of
    weighted bits, RmsNorm, Linear-GELU-Linear) embeds each query. A member
    scores ``scale * cos(q, z)``, ``scale = exp(log_scale)`` with
    ``log_scale`` learned from ``ln(1 / 0.07)`` and clamped to ``[0, ln 100]``;
    the loss is the softmax cross entropy of the answer among its group's
    members, averaged over the groups of a batch. A ``residual`` matcher adds
    ``offset_weight * offset`` of each member's fixed offset (the groups must
    carry offsets), ``offset_weight`` learned from 1: the network learns the
    correction to the offset's ranking."""

    def __init__(
        self,
        spec: GraphMambaSpec,
        *,
        n_bits: int = 4096,
        query_hidden: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        lr_schedule: Optional[LrSchedule] = None,
        seed: int = 0,
        residual: bool = False,
    ) -> None: ...
    def train_epoch(
        self,
        data: GraphDataset,
        queries: MatchQueries,
        groups: MatchGroups,
        batches: Sequence[Sequence[int]],
        epoch: int = 0,
        *,
        batch_rows: Optional[int] = None,
    ) -> int:
        """Queue one step per batch (a list of group ids); returns how many
        were queued. A batch's graphs are the distinct members of its groups,
        each encoded once. ``batch_rows`` is the row capacity of every graph
        batch (default: the largest batch's); the same value every epoch keeps
        the shapes few. Groups without a negative raise ``ValueError``.
        Nothing is read back; Ctrl-C stops after a completed step."""
    def read_losses(self) -> List[float]:
        """The loss of every step queued since the last call. One read."""
    def score_groups(
        self,
        data: GraphDataset,
        queries: MatchQueries,
        groups: MatchGroups,
        batches: Sequence[Sequence[int]],
        *,
        batch_rows: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Evaluation mode, no gradient: per group, in the order of the
        batches, ``float32`` scores of its distinct members, answer first
        (the full logits: with the weighted offsets for a residual matcher)."""
    def embed_graphs(
        self,
        data: GraphDataset,
        graph_ids: Optional[IntArray] = None,
        batch_graphs: int = 4096,
    ) -> np.ndarray:
        """``float32[n, d_out]`` L2-normalised graph embeddings (evaluation
        mode), read batch by batch."""
    def embed_queries(self, queries: MatchQueries, batch_queries: int = 1024) -> np.ndarray:
        """``float32[queries, d_out]`` L2-normalised query embeddings."""
    @property
    def scale(self) -> float:
        """``exp(clamp(log_scale))``."""
    @property
    def offset_weight(self) -> Optional[float]:
        """The learned weight of the member offsets; ``None`` unless residual."""
    @property
    def residual(self) -> bool: ...
    @property
    def step(self) -> int: ...
    @property
    def num_parameters(self) -> int: ...
    @property
    def d_out(self) -> int: ...
    @property
    def n_bits(self) -> int: ...
    @property
    def spec(self) -> GraphMambaSpec: ...
    def save(self, path: str, step: Optional[int] = None) -> None:
        """The spec, the encoder's config and every weight."""
    @staticmethod
    def load(
        path: str,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        lr_schedule: Optional[LrSchedule] = None,
    ) -> GraphMatchTrainer:
        """The matcher, with a fresh optimizer at the saved step."""
