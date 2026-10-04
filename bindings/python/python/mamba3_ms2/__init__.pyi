"""Type stubs for :mod:`mamba3_ms2`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from mamba3_rl._mamba3_rl import (
    __version__ as __version__,
    backend as backend,
    launch_count as launch_count,
    read_count as read_count,
    reset_launch_count as reset_launch_count,
    reset_read_count as reset_read_count,
    synchronize as synchronize,
)

ELEMENTS: List[Dict[str, Any]]

def upload_count() -> int:
    """Buffers created from host data since the counter was last reset."""

def reset_upload_count() -> None:
    """Reset the counter :func:`upload_count` reports."""

def request_status_names(bits: int) -> List[str]:
    """Names of the set request-status bits, in bit order."""

def candidate_status_names(bits: int) -> List[str]:
    """Names of the set candidate-status bits, in bit order."""

class SpectrumBatch:
    """A batch of spectra (contract §3.1): ``B`` spectra, row-major
    ``[B, n_raw]`` per-peak fields. Keyword arguments are named exactly as
    the contract's fields. Arrays may be C- or Fortran-ordered or
    non-contiguous (copied once); integer fields take any integer width,
    float fields any float width, and anything else raises ``TypeError``
    naming the field."""

    def __init__(
        self,
        *,
        n_raw: object,
        spectrum_id: np.ndarray,
        raw_peak_count: np.ndarray,
        peak_count: np.ndarray,
        peak_id: np.ndarray,
        mz_udalton: np.ndarray,
        intensity: np.ndarray,
        mz_uncertainty_udalton: np.ndarray,
        precursor_mz_udalton: np.ndarray,
        precursor_uncertainty_udalton: np.ndarray,
        adduct: np.ndarray,
        polarity: np.ndarray,
        collision_energy_ev: np.ndarray,
        collision_energy_known: np.ndarray,
        energy_count: np.ndarray,
        fragment_tolerance_ppm_tenths: np.ndarray,
        precursor_tolerance_ppm_tenths: np.ndarray,
        instrument_class: np.ndarray,
        schema_version: object = 1,
        intensity_scale: object = 0,
    ) -> None: ...
    @property
    def batch(self) -> int:
        """Spectra per batch."""
    @property
    def n_raw(self) -> int:
        """Raw peak capacity of the shape bucket."""
    def validate(self) -> List[int]:
        """One request-status code per spectrum; malformed batches raise
        ``ValueError`` with the Rust message."""
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> SpectrumBatch: ...

class ModelConfig:
    """Model hyperparameters (contract §3.3). Every keyword argument
    defaults to the documented V0 value; ``encoder``, ``decoder`` and
    ``formula_table`` are dicts (or JSON strings) of the same shape
    :meth:`to_json` produces."""

    def __init__(
        self,
        *,
        schema_version: Optional[object] = None,
        version: Optional[str] = None,
        chemistry: Optional[str] = None,
        n_peaks: Optional[object] = None,
        d_model: Optional[object] = None,
        encoder: Optional[Dict[str, Any]] = None,
        decoder: Optional[Dict[str, Any]] = None,
        encoder_blocks: Optional[object] = None,
        decoder_blocks: Optional[object] = None,
        attention_heads: Optional[object] = None,
        fourier_features: Optional[object] = None,
        max_atoms: Optional[object] = None,
        max_ring_closures: Optional[object] = None,
        formula_table: Optional[Dict[str, Any]] = None,
        assignment: Optional[Any] = None,
        energy_scale_ev: Optional[float] = None,
        energy_clip_ev: Optional[float] = None,
        dtype: Optional[str] = None,
    ) -> None: ...
    @staticmethod
    def v0() -> ModelConfig:
        """The documented V0 configuration."""
    def validate(self) -> None: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> ModelConfig: ...
    @property
    def schema_version(self) -> int: ...
    @property
    def version(self) -> str: ...
    @property
    def chemistry(self) -> str: ...
    @property
    def n_peaks(self) -> int: ...
    @property
    def d_model(self) -> int: ...
    @property
    def encoder(self) -> Dict[str, Any]: ...
    @property
    def decoder(self) -> Dict[str, Any]: ...
    @property
    def encoder_blocks(self) -> int: ...
    @property
    def decoder_blocks(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def fourier_features(self) -> int: ...
    @property
    def max_atoms(self) -> int: ...
    @property
    def max_ring_closures(self) -> int: ...
    @property
    def formula_table(self) -> Dict[str, Any]: ...
    @property
    def assignment(self) -> Optional[Dict[str, Any]]: ...
    @property
    def energy_scale_ev(self) -> float: ...
    @property
    def energy_clip_ev(self) -> float: ...
    @property
    def dtype(self) -> str: ...

class GenerationConfig:
    """Generation hyperparameters (contract §3.4). Every keyword argument
    defaults to the documented value; ``mode``, ``control`` and
    ``formula_source`` are strings (``"sampling"``, ``"none"``,
    ``"table"``)."""

    def __init__(
        self,
        *,
        schema_version: Optional[object] = None,
        trajectories: Optional[object] = None,
        formulas: Optional[object] = None,
        seed: Optional[object] = None,
        temperature: Optional[float] = None,
        max_steps: Optional[object] = None,
        max_device_bytes: Optional[object] = None,
        formula_rows_visited_max: Optional[object] = None,
        formula_rows_scored_max: Optional[object] = None,
        mode: Optional[str] = None,
        oracle_formula: Optional[bool] = None,
        control: Optional[str] = None,
        formula_source: Optional[str] = None,
        formula_window: Optional[object] = None,
        enum_lanes_max: Optional[object] = None,
        enum_lane_visits_max: Optional[object] = None,
        enum_dispatch_visits_max: Optional[object] = None,
        allocation: Optional[str] = None,
        identity: Optional[str] = None,
        identity_work_max: Optional[object] = None,
        returned: Optional[object] = None,
        evidence: Optional[bool] = None,
        ion_request_work_max: Optional[object] = None,
    ) -> None: ...
    def validate(
        self, max_atoms: Optional[int] = None, max_ring_closures: Optional[int] = None
    ) -> None:
        """Enforce every documented range under these structure limits
        (default the V0 ``max_atoms = 16``, ``max_ring_closures = 4``)."""
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> GenerationConfig: ...
    @property
    def schema_version(self) -> int: ...
    @property
    def trajectories(self) -> int: ...
    @property
    def formulas(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def temperature(self) -> float: ...
    @property
    def max_steps(self) -> int: ...
    @property
    def max_device_bytes(self) -> int: ...
    @property
    def formula_rows_visited_max(self) -> int: ...
    @property
    def formula_rows_scored_max(self) -> int: ...
    @property
    def mode(self) -> str: ...
    @property
    def oracle_formula(self) -> bool: ...
    @property
    def control(self) -> str: ...
    @property
    def formula_source(self) -> str: ...
    @property
    def formula_window(self) -> int: ...
    @property
    def enum_lanes_max(self) -> int: ...
    @property
    def enum_lane_visits_max(self) -> int: ...
    @property
    def enum_dispatch_visits_max(self) -> int: ...
    @property
    def allocation(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def identity_work_max(self) -> int: ...
    @property
    def returned(self) -> int: ...
    @property
    def evidence(self) -> bool: ...
    @property
    def ion_request_work_max(self) -> int: ...

class ChemistryDomain:
    """The chemistry domain as checkpoint data (contract §3.2). Built with
    no arguments it is the V0 domain; any field may be overridden with a
    keyword argument of the same shape :meth:`to_json` produces."""

    def __init__(
        self,
        *,
        schema_version: Optional[object] = None,
        version: Optional[str] = None,
        mass_scale: Optional[object] = None,
        elements: Optional[List[Dict[str, Any]]] = None,
        electron_mass: Optional[object] = None,
        electron_residual_nda: Optional[object] = None,
        atom_types: Optional[List[Dict[str, Any]]] = None,
        bond_orders: Optional[List[int]] = None,
        adducts: Optional[List[Dict[str, Any]]] = None,
        max_hydrogen_shift: Optional[object] = None,
        grammar: Optional[str] = None,
        traversal: Optional[str] = None,
        recipe: Optional[str] = None,
    ) -> None: ...
    @staticmethod
    def v0() -> ChemistryDomain:
        """The V0 domain."""
    def validate(self) -> None: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> ChemistryDomain: ...
    @property
    def schema_version(self) -> int: ...
    @property
    def version(self) -> str: ...
    @property
    def mass_scale(self) -> int: ...
    @property
    def elements(self) -> List[Dict[str, Any]]: ...
    @property
    def electron_mass(self) -> int: ...
    @property
    def electron_residual_nda(self) -> int: ...
    @property
    def atom_types(self) -> List[Dict[str, Any]]: ...
    @property
    def bond_orders(self) -> List[int]: ...
    @property
    def adducts(self) -> List[Dict[str, Any]]: ...
    @property
    def max_hydrogen_shift(self) -> int: ...
    @property
    def grammar(self) -> str: ...
    @property
    def traversal(self) -> str: ...
    @property
    def recipe(self) -> str: ...

class FormulaTable:
    """The resident V0 formula table (contracts §9)."""

    @staticmethod
    def from_json(path_or_text: str) -> FormulaTable:
        """Read the ``tools/ms2/formula_table.py`` JSON format. The
        argument is a path when it names an existing file, otherwise the
        JSON text itself."""
    def to_json(self) -> str: ...
    @property
    def rows(self) -> int:
        """Rows in the table."""
    def __len__(self) -> int: ...

class CandidateBatch:
    """Generated candidates of a batch (contract §3.5): exactly ``batch *
    trajectories`` records in ``(spectrum, trajectory)`` order. Every field
    is a NumPy array with the contract's dtype and shape."""

    @staticmethod
    def from_json(json: str) -> CandidateBatch: ...
    def to_json(self) -> str: ...
    def validate(self) -> None:
        """Check lengths and every invariant of contract §§3.5, 4.4 and 8."""
    def distinct_traces(self) -> List[int]:
        """Indices of the finished, valid records without ``duplicate_trace``
        or ``request_failed``."""
    def pack(self, returned: int) -> PackedCandidateBatch:
        """Rank and compact to ``returned`` slots per spectrum with the raw
        score and trace-only identity (the host side of the packed readout)."""
    @property
    def schema_version(self) -> int: ...
    @property
    def batch(self) -> int: ...
    @property
    def trajectories(self) -> int: ...
    @property
    def max_steps(self) -> int: ...
    @property
    def max_atoms(self) -> int: ...
    @property
    def max_ring_closures(self) -> int: ...
    @property
    def spectrum_id(self) -> np.ndarray:
        """``uint64[B*K]``."""
    @property
    def trajectory(self) -> np.ndarray:
        """``uint32[B*K]``, ``0..K`` per spectrum."""
    @property
    def actions(self) -> np.ndarray:
        """``uint32[B*K, T, 4]``."""
    @property
    def length(self) -> np.ndarray:
        """``uint32[B*K]``."""
    @property
    def formula_row(self) -> np.ndarray:
        """``uint32[B*K]``."""
    @property
    def formula_log_prob(self) -> np.ndarray:
        """``float32[B*K]``."""
    @property
    def trace_log_prob(self) -> np.ndarray:
        """``float32[B*K]``."""
    @property
    def open_valence(self) -> np.ndarray:
        """``uint8[B*K, A]``."""
    @property
    def attachment_partition(self) -> np.ndarray:
        """``uint8[B*K]``."""
    @property
    def status(self) -> np.ndarray:
        """``uint32[B*K]`` candidate bits."""
    @property
    def evidence_status(self) -> np.ndarray:
        """``uint8[B*K]``."""
    @property
    def identity_resolution(self) -> np.ndarray:
        """``uint8[B*K]``."""
    @property
    def request_status(self) -> np.ndarray:
        """``uint32[B]`` request bits."""
    @property
    def rows_visited(self) -> np.ndarray:
        """``uint32[B]``."""
    @property
    def rows_joined(self) -> np.ndarray:
        """``uint32[B]``."""
    @property
    def rows_scored(self) -> np.ndarray:
        """``uint32[B]``."""
    @property
    def formula_support_complete(self) -> np.ndarray:
        """``uint8[B]``."""
    @property
    def formula_mass_retained(self) -> np.ndarray:
        """``float32[B]``."""
    @property
    def peaks_kept(self) -> np.ndarray:
        """``uint32[B]``."""
    @property
    def intensity_retained(self) -> np.ndarray:
        """``float32[B]``."""
    @property
    def formula_counts(self) -> np.ndarray:
        """``uint16[B*K, 10]``."""
    @property
    def formula_source(self) -> np.ndarray:
        """``uint8[B]``."""
    @property
    def formula_rank(self) -> np.ndarray:
        """``uint32[B*K]``."""
    @property
    def evidence_count(self) -> np.ndarray:
        """``uint8[B*K]``."""
    @property
    def evidence_peak_id(self) -> np.ndarray:
        """``uint32[B*K, 4]`` original peak ids."""
    @property
    def evidence_hypothesis(self) -> np.ndarray:
        """``uint8[B*K, 4]``."""
    @property
    def evidence_shift(self) -> np.ndarray:
        """``int8[B*K, 4]``."""
    @property
    def evidence_residual(self) -> np.ndarray:
        """``int32[B*K, 4]``."""
    @property
    def evidence_log_prob(self) -> np.ndarray:
        """``float32[B*K, 4]``."""

class ExperimentSet:
    """A loaded experiment dataset: molecules plus their spectra in file
    order, with parents and pseudo-labels built under the frozen V0 recipe
    limits."""

    @staticmethod
    def from_export(path: str, table: Optional[FormulaTable] = None) -> ExperimentSet:
        """Load an export file. ``table`` is accepted for the call shape;
        the labels come from the export itself and the table binds at
        trainer/model construction."""
    def take_labeled(self, n: int) -> ExperimentSet:
        """The first ``n`` labeled spectra in file order, one per molecule."""
    @property
    def spectrum_count(self) -> int: ...
    @property
    def labeled_count(self) -> int:
        """In-domain spectra with at least one target."""
    @property
    def molecule_count(self) -> int: ...

class TrainConfig:
    """Training hyperparameters. Every keyword argument defaults to the
    documented V0 value."""

    def __init__(
        self,
        *,
        batch: Optional[object] = None,
        slots: Optional[object] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        formula_weight: Optional[float] = None,
        seed: Optional[object] = None,
        control: Optional[str] = None,
        grad_clip: Optional[float] = None,
        gold_formula_conditioning: Optional[str] = None,
        formula_source: Optional[str] = None,
        formula_window: Optional[object] = None,
        lambda_assign: Optional[float] = None,
        ion_request_work_max: Optional[object] = None,
    ) -> None: ...
    def validate(self) -> None: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> TrainConfig: ...
    @property
    def batch(self) -> int: ...
    @property
    def slots(self) -> int: ...
    @property
    def lr(self) -> float: ...
    @property
    def weight_decay(self) -> float: ...
    @property
    def formula_weight(self) -> float: ...
    @property
    def seed(self) -> int: ...
    @property
    def control(self) -> str: ...
    @property
    def grad_clip(self) -> Optional[float]: ...
    @property
    def gold_formula_conditioning(self) -> str: ...
    @property
    def lambda_assign(self) -> float: ...
    @property
    def ion_request_work_max(self) -> int: ...

class Ms2Model:
    """The composed MS2 model: spectrum encoder, formula head and
    graph-action decoder, initialised together from one
    :class:`ModelConfig`. The formula table is uploaded once and reused."""

    def __init__(self, config: ModelConfig, table: FormulaTable, seed: int = 0) -> None: ...
    def generate(self, batch: SpectrumBatch, config: GenerationConfig) -> CandidateBatch:
        """Run the full generation pipeline with one batched read."""
    def generate_packed(
        self, batch: SpectrumBatch, config: GenerationConfig
    ) -> PackedCandidateBatch:
        """Run the full packed pipeline: ``B * R`` records in rank order
        with exactly one device read."""
    def generate_resident(
        self, batch: SpectrumBatch, config: GenerationConfig
    ) -> ResidentCandidates:
        """Run the full packed pipeline with no device read and return the
        device-resident result."""

class PackedCandidateBatch:
    """Ranked, compacted candidates of a batch: ``B * R`` records in
    ``(spectrum, rank)`` order. Every field is a NumPy array with the
    contract's dtype and shape."""

    @staticmethod
    def from_json(json: str) -> PackedCandidateBatch: ...
    def to_json(self) -> str: ...
    def validate(self) -> None:
        """Check every invariant of architecture §4.4."""
    @property
    def schema_version(self) -> int: ...
    @property
    def batch(self) -> int: ...
    @property
    def returned(self) -> int: ...
    @property
    def trajectories(self) -> int: ...
    @property
    def max_steps(self) -> int: ...
    @property
    def max_atoms(self) -> int: ...
    @property
    def max_ring_closures(self) -> int: ...
    @property
    def spectrum_id(self) -> Any: ...
    @property
    def trajectory(self) -> Any: ...
    @property
    def actions(self) -> Any: ...
    @property
    def length(self) -> Any: ...
    @property
    def formula_row(self) -> Any: ...
    @property
    def formula_rank(self) -> Any: ...
    @property
    def formula_counts(self) -> Any: ...
    @property
    def formula_log_prob(self) -> Any: ...
    @property
    def trace_log_prob(self) -> Any: ...
    @property
    def score(self) -> Any: ...
    @property
    def open_valence(self) -> Any: ...
    @property
    def status(self) -> Any: ...
    @property
    def evidence_status(self) -> Any: ...
    @property
    def evidence_count(self) -> Any: ...
    @property
    def evidence_peak_id(self) -> Any: ...
    @property
    def evidence_hypothesis(self) -> Any: ...
    @property
    def evidence_shift(self) -> Any: ...
    @property
    def evidence_residual(self) -> Any: ...
    @property
    def evidence_log_prob(self) -> Any: ...
    @property
    def identity_resolution(self) -> Any: ...
    @property
    def attachment_partition(self) -> Any: ...
    @property
    def returned_count(self) -> Any: ...
    @property
    def request_status(self) -> Any: ...
    @property
    def rows_visited(self) -> Any: ...
    @property
    def rows_joined(self) -> Any: ...
    @property
    def rows_scored(self) -> Any: ...
    @property
    def formula_support_complete(self) -> Any: ...
    @property
    def formula_mass_retained(self) -> Any: ...
    @property
    def peaks_kept(self) -> Any: ...
    @property
    def intensity_retained(self) -> Any: ...
    @property
    def formula_source(self) -> Any: ...

class ResidentCandidates:
    """Device-resident packed candidates: owns its device buffers with no
    host copy yet."""

    def read(self, model: Ms2Model) -> PackedCandidateBatch:
        """Perform exactly one device read and return the packed batch."""
    def release(self) -> None:
        """Drop the leased buffers explicitly."""

class Ms2Trainer:
    """The MS2 trainer: the composed model with its AdamW optimizer and the
    resident formula table."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        table: FormulaTable,
        seed: Optional[int] = None,
    ) -> None: ...
    @staticmethod
    def load(path: str, table: FormulaTable) -> Ms2Trainer:
        """Load a checkpoint saved by :meth:`save`."""
    def save(self, path: str) -> None: ...
    def request_report(self) -> None:
        """Ask the next :meth:`step` to report its losses."""
    def step(self, set: ExperimentSet, indices: object) -> Optional[Dict[str, float]]:
        """One optimizer step. Returns the pre-update losses as a dict
        (``step``, ``loss``, ``graph``, ``formula``, ``assign``,
        ``spectra``, ``formula_present``, ``formula_absent``,
        ``gold_not_scored``, ``assign_eligible``, ``assign_partial``,
        ``assign_dropped``, ``assignment_label_overflow``) when a
        report was requested, else ``None``."""
    def teacher_eval(self, set: ExperimentSet, indices: object) -> Dict[str, Any]:
        """Teacher-forced evaluation: a dict of arrays (``nll``, ``q``,
        ``scored_tokens``, ``gold_slot``, ``gold_log_prob``,
        ``molecules``) with the scalar counts."""
    @property
    def step_count(self) -> int: ...
    @property
    def table_sha256(self) -> str: ...
    @property
    def train_config(self) -> TrainConfig: ...
