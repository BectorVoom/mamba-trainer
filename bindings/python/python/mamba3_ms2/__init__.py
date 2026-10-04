"""MS2-to-substructure generation and training, on the device.

The MS2 model turns a batch of mass spectra into structural proposals:
a spectrum encoder, a formula head over a resident formula table and a
graph-action decoder, composed as :class:`Ms2Model`. :class:`Ms2Trainer`
trains the same model with teacher forcing on an :class:`ExperimentSet`
loaded from an export file.

A run, end to end::

    import mamba3_ms2 as ms2

    table = ms2.FormulaTable.from_json("formula_table.json")
    model = ms2.Ms2Model(ms2.ModelConfig(), table, seed=0)
    batch = ms2.SpectrumBatch(n_raw=64, spectrum_id=..., ...)
    out = model.generate(batch, ms2.GenerationConfig(trajectories=8))
    out.validate()

This module and :mod:`mamba3_rl` are two import names of one extension
library: they share one device and one set of counters.
"""

from mamba3_rl._mamba3_rl import (
    ELEMENTS,
    CandidateBatch,
    ChemistryDomain,
    ExperimentSet,
    FormulaTable,
    GenerationConfig,
    ModelConfig,
    Ms2Model,
    Ms2Trainer,
    PackedCandidateBatch,
    ResidentCandidates,
    SpectrumBatch,
    TrainConfig,
    __version__,
    backend,
    candidate_status_names,
    launch_count,
    read_count,
    request_status_names,
    reset_launch_count,
    reset_read_count,
    reset_upload_count,
    synchronize,
    upload_count,
)

__all__ = [
    "ELEMENTS",
    "CandidateBatch",
    "ChemistryDomain",
    "ExperimentSet",
    "FormulaTable",
    "GenerationConfig",
    "ModelConfig",
    "Ms2Model",
    "Ms2Trainer",
    "PackedCandidateBatch",
    "ResidentCandidates",
    "SpectrumBatch",
    "TrainConfig",
    "__version__",
    "backend",
    "candidate_status_names",
    "launch_count",
    "read_count",
    "request_status_names",
    "reset_launch_count",
    "reset_read_count",
    "reset_upload_count",
    "synchronize",
    "upload_count",
]
