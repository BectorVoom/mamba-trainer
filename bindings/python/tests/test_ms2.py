"""The ``mamba3_ms2`` module: MS2 generation and training from Python.

The Rust suites prove the model; these prove the boundary -- that the Python
surface is the Rust one (configs round-trip through the same JSON, defaults
equal the Rust defaults field by field), that arrays go in in whatever layout
NumPy holds them, that validation failures raise the mapped exception with the
Rust message, and that a deterministic scenario run through the binding
reports the same per-step losses as the Rust example driver.
"""

import json
import math
import pathlib
import re
import subprocess

import numpy as np
import pytest

import mamba3_ms2 as ms
import mamba3_rl

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "ms2"
TABLE_PATH = DATA / "formula_table_msgym_v0.json"
OVERFIT_PATH = DATA / "overfit_train.json"

M_H = 1_007_825  # integer hydrogen mass, 1e-6 Da units
M_E = 549  # integer electron mass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def table():
    return ms.FormulaTable.from_json(str(TABLE_PATH))


def table_row_mass():
    rows = json.loads(TABLE_PATH.read_text())["rows"]
    return rows[0][0]


def make_batch(b=2, n_raw=64, peaks=4):
    """A valid batch whose precursor matches the first table row ([M+H]+)."""
    mass = table_row_mass()
    precursor = mass + M_H - M_E
    peak_ids = np.zeros((b, n_raw), dtype=np.uint32)
    peak_ids[:, :peaks] = np.arange(peaks, dtype=np.uint32)
    mz = np.zeros((b, n_raw), dtype=np.uint32)
    mz[:, :peaks] = np.array([12_000_000, 25_000_000, 40_000_000, 59_000_000][:peaks],
                             dtype=np.uint32)
    intensity = np.zeros((b, n_raw), dtype=np.float32)
    intensity[:, :peaks] = 1.0
    fields = dict(
        n_raw=n_raw,
        spectrum_id=np.arange(100, 100 + b, dtype=np.uint64),
        raw_peak_count=np.full(b, peaks, dtype=np.uint32),
        peak_count=np.full(b, peaks, dtype=np.uint32),
        peak_id=peak_ids,
        mz_udalton=mz,
        intensity=intensity,
        mz_uncertainty_udalton=np.full(b, 50, dtype=np.uint32),
        precursor_mz_udalton=np.full(b, precursor, dtype=np.uint32),
        precursor_uncertainty_udalton=np.full(b, 50, dtype=np.uint32),
        adduct=np.full(b, 1, dtype=np.uint16),
        polarity=np.full(b, 1, dtype=np.int8),
        collision_energy_ev=np.zeros(b, dtype=np.float32),
        collision_energy_known=np.zeros(b, dtype=np.uint8),
        energy_count=np.ones(b, dtype=np.uint8),
        fragment_tolerance_ppm_tenths=np.zeros(b, dtype=np.uint16),
        precursor_tolerance_ppm_tenths=np.zeros(b, dtype=np.uint16),
        instrument_class=np.zeros(b, dtype=np.uint8),
    )
    return ms.SpectrumBatch(**fields), fields


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


def test_the_module_is_a_second_import_name_of_one_extension():
    assert ms.Ms2Model.__module__ == "mamba3_ms2"
    assert ms.SpectrumBatch.__module__ == "mamba3_ms2"
    assert ms.read_count is mamba3_rl.read_count
    assert ms.upload_count is mamba3_rl.upload_count
    assert ms.backend() == mamba3_rl.backend()
    assert not hasattr(mamba3_rl, "Ms2Model")
    assert pathlib.Path(ms.__file__).parents[1] == pathlib.Path(mamba3_rl.__file__).parents[1]


def test_every_public_name_is_exported_and_typed():
    extension = mamba3_rl._mamba3_rl
    names = {
        name
        for name in dir(extension)
        if not name.startswith("_")
        and getattr(getattr(extension, name), "__module__", None) == "mamba3_ms2"
    }
    source = (ROOT / "bindings/python/src/ms2.rs").read_text()
    functions = set(re.findall(r"#\[pyfunction\]\s*(?:#\[pyo3[^\]]*\]\s*)?fn (\w+)", source))
    names |= functions
    assert {
        "SpectrumBatch", "ModelConfig", "GenerationConfig", "ChemistryDomain",
        "FormulaTable", "CandidateBatch", "ExperimentSet", "TrainConfig",
        "Ms2Model", "Ms2Trainer",
    } <= names
    stubs = (pathlib.Path(ms.__file__).parent / "__init__.pyi").read_text()
    for name in sorted(names):
        if name == "ELEMENTS":
            assert name in ms.__all__
            continue
        assert name in ms.__all__, f"{name} is not exported from mamba3_ms2"
        assert hasattr(ms, name)
        assert re.search(rf"\b(class|def) {name}\b", stubs), f"{name} has no stub"
    for name in ms.__all__:
        assert hasattr(ms, name)


# ---------------------------------------------------------------------------
# Configs: defaults, attributes, round trips
# ---------------------------------------------------------------------------


def test_model_config_defaults_equal_the_rust_v0_values():
    cfg = ms.ModelConfig()
    assert cfg == ms.ModelConfig.v0()
    # Contract §3.3, the V0 column, field by field.
    assert cfg.schema_version == 2
    assert cfg.version == "ms2-model-v0"
    assert cfg.chemistry == "ms2-chem-v0.1"
    assert cfg.n_peaks == 128
    assert cfg.d_model == 128
    assert cfg.encoder_blocks == 2
    assert cfg.decoder_blocks == 2
    assert cfg.attention_heads == 4
    assert cfg.fourier_features == 16
    assert cfg.max_atoms == 16
    assert cfg.max_ring_closures == 4
    assert cfg.energy_scale_ev == 100.0
    assert cfg.energy_clip_ev == 400.0
    assert cfg.dtype == "f32"
    assert cfg.formula_table["version"] == "ms2-formula-v0"
    assert cfg.formula_table["rows"] == 37_859
    for ssm in (cfg.encoder, cfg.decoder):
        assert ssm["d_model"] == 128
        assert ssm["n_heads"] == 4
        assert ssm["head_dim"] == 64
        assert ssm["d_state"] == 32
        assert ssm["n_groups"] == 4
        assert ssm["conv_kernel"] is None
        assert ssm["discretization"] == "LearnedTrapezoid"
        assert ssm["dynamics"] == "Rotational"
        assert ssm["mode"] == "Siso"
    cfg.validate()
    again = ms.ModelConfig.from_json(cfg.to_json())
    assert again == cfg
    assert again.to_json() == cfg.to_json()
    with pytest.raises(ValueError):
        ms.ModelConfig.from_json("{")
    # Overrides stick; every attribute reads back.
    custom = ms.ModelConfig(
        max_atoms=8, max_ring_closures=2, decoder_blocks=1,
        version="ms2-model-v0", chemistry="ms2-chem-v0.1",
    )
    assert (custom.max_atoms, custom.max_ring_closures, custom.decoder_blocks) == (8, 2, 1)
    custom.validate()
    assert ms.ModelConfig.from_json(custom.to_json()) == custom
    with pytest.raises(ValueError, match="max_atoms"):
        ms.ModelConfig(max_atoms=64).validate()
    with pytest.raises(ValueError, match="chemistry"):
        ms.ModelConfig(chemistry="nope").validate()


def test_generation_config_defaults_and_round_trip():
    cfg = ms.GenerationConfig()
    assert cfg.trajectories == 8
    assert cfg.formulas == 4
    assert cfg.seed == 0
    assert cfg.temperature == 1.0
    assert cfg.max_steps == 22
    assert cfg.max_device_bytes == 2 * 1024 * 1024 * 1024
    assert cfg.formula_rows_visited_max == 2**32 - 1
    assert cfg.formula_rows_scored_max == 4096
    assert cfg.mode == "sampling"
    assert cfg.oracle_formula is False
    assert cfg.control == "none"
    assert cfg.formula_source == "table"
    assert cfg.formula_window == 32
    assert cfg.allocation == "round_robin"
    assert cfg.identity == "trace"
    assert cfg.identity_work_max == 4096
    assert cfg.returned == 0
    cfg.validate()
    cfg.validate(16, 4)
    again = ms.GenerationConfig.from_json(cfg.to_json())
    assert again == cfg
    assert again.to_json() == cfg.to_json()
    custom = ms.GenerationConfig(trajectories=2, formulas=1, seed=7, temperature=0.5)
    assert (custom.trajectories, custom.formulas, custom.seed) == (2, 1, 7)
    assert custom.temperature == pytest.approx(0.5)
    assert ms.GenerationConfig.from_json(custom.to_json()) == custom
    with pytest.raises(ValueError, match="trajectories"):
        ms.GenerationConfig(trajectories=0).validate()
    with pytest.raises(ValueError, match="formula_window"):
        ms.GenerationConfig(formula_window=7).validate()
    with pytest.raises(ValueError, match="temperature"):
        ms.GenerationConfig(temperature=0.0).validate()
    with pytest.raises(NotImplementedError, match="Beam"):
        ms.GenerationConfig(mode="beam").validate()
    with pytest.raises(NotImplementedError, match="Enumerate"):
        ms.GenerationConfig(formula_source="enumerate").validate()


def test_chemistry_domain_defaults_and_round_trip():
    dom = ms.ChemistryDomain()
    assert dom == ms.ChemistryDomain.v0()
    assert dom.schema_version == 1
    assert dom.version == "ms2-chem-v0.1"
    assert dom.mass_scale == 1_000_000
    assert [e["symbol"] for e in dom.elements] == [
        "C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    ]
    assert len(dom.atom_types) == 17
    assert dom.bond_orders == [1, 2, 3]
    assert [a["name"] for a in dom.adducts] == ["[M+H]+", "[M-H]-"]
    assert dom.max_hydrogen_shift == 2
    assert (dom.grammar, dom.traversal, dom.recipe) == (
        "grammar-bfs-v1", "bfs-canon-v1", "q-cut-v1",
    )
    dom.validate()
    again = ms.ChemistryDomain.from_json(dom.to_json())
    assert again == dom
    assert again.to_json() == dom.to_json()
    with pytest.raises(ValueError, match="schema_version"):
        ms.ChemistryDomain(schema_version=9).validate()


def test_train_config_defaults_and_round_trip():
    cfg = ms.TrainConfig()
    assert cfg.batch == 16
    assert cfg.slots == 16
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.formula_weight == pytest.approx(0.2)
    assert cfg.seed == 1
    assert cfg.control == "none"
    assert cfg.grad_clip is None
    assert cfg.gold_formula_conditioning == "row"
    cfg.validate()
    again = ms.TrainConfig.from_json(cfg.to_json())
    assert again == cfg
    assert again.to_json() == cfg.to_json()
    custom = ms.TrainConfig(
        batch=8, lr=1e-3, control="shuffled",
        gold_formula_conditioning="composition", grad_clip=1.0,
    )
    assert custom.batch == 8
    assert custom.control == "shuffled"
    assert custom.gold_formula_conditioning == "composition"
    assert custom.grad_clip == pytest.approx(1.0)
    custom.validate()
    assert ms.TrainConfig.from_json(custom.to_json()) == custom
    with pytest.raises(ValueError, match="batch"):
        ms.TrainConfig(batch=0).validate()
    with pytest.raises(ValueError, match="slots"):
        ms.TrainConfig(slots=17).validate()
    with pytest.raises(ValueError, match="lr"):
        ms.TrainConfig(lr=-1.0).validate()


def test_formula_table_from_path_and_text():
    from_path = ms.FormulaTable.from_json(str(TABLE_PATH))
    from_text = ms.FormulaTable.from_json(TABLE_PATH.read_text())
    assert len(from_path) == from_path.rows > 0
    assert from_text.to_json() == from_path.to_json()
    with pytest.raises(ValueError, match="formula_table"):
        ms.FormulaTable.from_json('{"rows": []}')


# ---------------------------------------------------------------------------
# SpectrumBatch: layouts, dtypes, validation
# ---------------------------------------------------------------------------


def test_batch_layouts_give_identical_json():
    _, fields = make_batch()
    reference = ms.SpectrumBatch(**fields).to_json()
    def strided(v):
        # A non-contiguous view with identical logical values.
        if v.ndim == 2:
            big = np.zeros((v.shape[0], 2 * v.shape[1]), dtype=v.dtype)
            big[:, ::2] = v
            return big[:, ::2]
        big = np.zeros(2 * v.shape[0], dtype=v.dtype)
        big[::2] = v
        return big[::2]

    variants = {
        "fortran": {k: np.asfortranarray(v) if isinstance(v, np.ndarray) else v
                    for k, v in fields.items()},
        "sliced": {k: strided(v) if isinstance(v, np.ndarray) else v
                    for k, v in fields.items()},
        "wider ints": {k: (v.astype(np.int64) if isinstance(v, np.ndarray)
                            and np.issubdtype(v.dtype, np.integer) else v)
                       for k, v in fields.items()},
        "float64 floats": {k: (v.astype(np.float64) if isinstance(v, np.ndarray)
                                and np.issubdtype(v.dtype, np.floating) else v)
                           for k, v in fields.items()},
    }
    assert not variants["fortran"]["peak_id"].flags.c_contiguous
    assert not variants["sliced"]["peak_id"].flags.c_contiguous
    for name, given in variants.items():
        got = ms.SpectrumBatch(**given).to_json()
        assert got == reference, name
        # The caller's arrays are not modified.
        for key, value in given.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    np.asarray(value), np.asarray(fields[key]), err_msg=f"{name}: {key}")


def test_batch_dtype_mismatches_name_the_field():
    _, fields = make_batch()

    def fails(match, exc, **changes):
        given = dict(fields)
        given.update(changes)
        with pytest.raises(exc, match=match):
            ms.SpectrumBatch(**given)

    floats = np.ones((2, 64), dtype=np.float64)
    ints = np.ones((2, 64), dtype=np.int32)
    fails("mz_udalton", TypeError, mz_udalton=floats)
    fails("peak_id", TypeError, peak_id=floats)
    fails("spectrum_id", TypeError, spectrum_id=floats[:2, :1].reshape(2))
    fails("intensity", TypeError, intensity=ints)
    fails("collision_energy_ev", TypeError,
          collision_energy_ev=np.ones(2, dtype=np.int32))
    fails("adduct", ValueError, adduct=np.full(2, 70000, dtype=np.int64))
    fails("peak_id", ValueError, peak_id=np.full((2, 64), 2**40, dtype=np.uint64))
    # A wrong shape:
    fails("mz_udalton", ValueError, mz_udalton=np.zeros((2, 63), dtype=np.uint32))
    # B comes from spectrum_id, so the next field names the mismatch:
    fails("raw_peak_count", ValueError, spectrum_id=np.ones(3, dtype=np.uint64))


def test_batch_validation_failures_raise_with_the_rust_message():
    _, fields = make_batch()

    def fails(match, **changes):
        given = dict(fields)
        given.update(changes)
        batch = ms.SpectrumBatch(**given)
        with pytest.raises(ValueError, match=match):
            batch.validate()

    fails("duplicate spectrum_id", spectrum_id=np.array([7, 7], dtype=np.uint64))
    fails("intensity_scale", intensity_scale=2)
    fails("collision_energy_known",
          collision_energy_known=np.full(2, 2, dtype=np.uint8))
    fails("energy_count", energy_count=np.full(2, 9, dtype=np.uint8))
    fails("fragment_tolerance_ppm_tenths",
          fragment_tolerance_ppm_tenths=np.full(2, 1001, dtype=np.uint16))
    fails("precursor_tolerance_ppm_tenths",
          precursor_tolerance_ppm_tenths=np.full(2, 2000, dtype=np.uint16))
    fails("instrument_class", instrument_class=np.full(2, 5, dtype=np.uint8))
    bad_ids = np.tile(np.arange(64, dtype=np.uint32), (2, 1))
    bad_ids[:, 3] = bad_ids[:, 2]  # a duplicate id among the valid peaks
    fails("strictly increasing", peak_id=bad_ids)
    fails("raw_peak_count",
          raw_peak_count=np.full(2, 2, dtype=np.uint32))  # below peak_count=4
    fails("raw_peak_count",
          peak_id=np.tile(np.arange(10, 74, dtype=np.uint32), (2, 1)))
    def fails_construction(match, **changes):
        given = dict(fields)
        given.update(changes)
        with pytest.raises(ValueError, match=match):
            ms.SpectrumBatch(**given)

    fails_construction("n_raw", n_raw=63)
    fails("schema_version", schema_version=2)

    # Per-spectrum statuses still validate: one bit set per spectrum.
    def statuses(**changes):
        given = dict(fields)
        given.update(changes)
        return ms.SpectrumBatch(**given).validate()

    assert statuses() == [0, 0]
    empty = dict(fields)
    empty["peak_count"] = np.zeros(2, dtype=np.uint32)
    empty["raw_peak_count"] = np.zeros(2, dtype=np.uint32)
    assert statuses(**empty) == [1 << 0, 1 << 0]
    neg = dict(fields)
    neg["intensity"] = np.full((2, 64), -1.0, dtype=np.float32)
    bits = statuses(**neg)
    assert ms.request_status_names(bits[0]) == ["negative_intensity"]
    unknown = dict(fields)
    unknown["adduct"] = np.zeros(2, dtype=np.uint16)
    assert "insufficient_metadata" in ms.request_status_names(statuses(**unknown)[0])
    bad_polarity = dict(fields)
    bad_polarity["polarity"] = np.zeros(2, dtype=np.int8)
    assert "invalid_polarity" in ms.request_status_names(statuses(**bad_polarity)[0])
    nan_ev = dict(fields)
    nan_ev["collision_energy_known"] = np.ones(2, dtype=np.uint8)
    nan_ev["collision_energy_ev"] = np.full(2, np.nan, dtype=np.float32)
    assert "nonfinite_input" in ms.request_status_names(statuses(**nan_ev)[0])
    assert ms.request_status_names(0) == []


def test_batch_json_round_trip():
    batch, _ = make_batch()
    assert batch.batch == 2 and batch.n_raw == 64
    again = ms.SpectrumBatch.from_json(batch.to_json())
    assert again.to_json() == batch.to_json()
    assert again.validate() == batch.validate()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_shapes_dtypes_determinism_and_validate():
    t = table()
    model = ms.Ms2Model(ms.ModelConfig(), t, seed=0)
    batch, _ = make_batch()
    gen = ms.GenerationConfig(trajectories=2, formulas=1, seed=0)
    out = model.generate(batch, gen)
    assert (out.batch, out.trajectories, out.max_steps, out.max_atoms) == (2, 2, 22, 16)
    n = 4
    assert out.spectrum_id.dtype == np.uint64 and out.spectrum_id.shape == (n,)
    assert out.trajectory.dtype == np.uint32 and out.trajectory.shape == (n,)
    assert out.actions.dtype == np.uint32 and out.actions.shape == (n, 22, 4)
    assert out.length.dtype == np.uint32 and out.length.shape == (n,)
    assert out.formula_row.dtype == np.uint32 and out.formula_row.shape == (n,)
    assert out.formula_log_prob.dtype == np.float32 and out.formula_log_prob.shape == (n,)
    assert out.trace_log_prob.dtype == np.float32 and out.trace_log_prob.shape == (n,)
    assert out.open_valence.dtype == np.uint8 and out.open_valence.shape == (n, 16)
    assert out.attachment_partition.dtype == np.uint8
    assert out.status.dtype == np.uint32 and out.status.shape == (n,)
    assert out.evidence_status.dtype == np.uint8
    assert out.identity_resolution.dtype == np.uint8
    assert out.request_status.dtype == np.uint32 and out.request_status.shape == (2,)
    assert out.rows_visited.dtype == np.uint32
    assert out.rows_joined.dtype == np.uint32
    assert out.rows_scored.dtype == np.uint32
    assert out.formula_support_complete.dtype == np.uint8
    assert out.formula_mass_retained.dtype == np.float32
    assert out.peaks_kept.dtype == np.uint32
    assert out.intensity_retained.dtype == np.float32
    assert out.formula_counts.dtype == np.uint16 and out.formula_counts.shape == (n, 10)
    assert out.formula_source.dtype == np.uint8 and out.formula_source.shape == (2,)
    assert out.formula_rank.dtype == np.uint32 and out.formula_rank.shape == (n,)
    out.validate()
    assert isinstance(out.distinct_traces(), list)

    # The same seed gives identical arrays; a different seed differs.
    again = model.generate(batch, ms.GenerationConfig(trajectories=2, formulas=1, seed=0))
    assert again.to_json() == out.to_json()
    other = model.generate(batch, ms.GenerationConfig(trajectories=2, formulas=1, seed=1))
    assert other.to_json() != out.to_json()
    assert not np.array_equal(other.actions, out.actions)
    other.validate()
    # JSON round trip preserves every field.
    assert ms.CandidateBatch.from_json(out.to_json()).to_json() == out.to_json()


def test_generate_packed_equals_pack_of_generate():
    t = table()
    model = ms.Ms2Model(ms.ModelConfig(), t, seed=0)
    batch, _ = make_batch()
    gen = ms.GenerationConfig(trajectories=2, formulas=1, seed=0)
    packed = model.generate_packed(batch, gen)
    packed.validate()
    assert (packed.batch, packed.returned, packed.trajectories) == (2, 2, 2)
    assert packed.score.dtype == np.float32
    # The Rust-side `pack` of `generate` through the binding.
    out = model.generate(batch, gen)
    via = out.pack(2)
    assert via.to_json() == packed.to_json()
    assert ms.PackedCandidateBatch.from_json(packed.to_json()).to_json() == packed.to_json()
    # The resident path defers the same read.
    resident = model.generate_resident(batch, gen)
    deferred = resident.read(model)
    assert deferred.to_json() == packed.to_json()
    resident.release()


# ---------------------------------------------------------------------------
# Trainer: step, reports, evaluation, checkpoints
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def overfit_set():
    if not OVERFIT_PATH.is_file():
        pytest.skip(f"{OVERFIT_PATH} is not present")
    full = ms.ExperimentSet.from_export(str(OVERFIT_PATH), table())
    assert full.spectrum_count > 0 and full.labeled_count > 0
    assert full.molecule_count > 0
    return full.take_labeled(8)


def make_trainer(**overrides):
    settings = dict(batch=8, seed=1, gold_formula_conditioning="composition")
    settings.update(overrides)
    return ms.Ms2Trainer(ms.ModelConfig(), ms.TrainConfig(**settings), table())


def test_experiment_set_counts(overfit_set):
    assert overfit_set.spectrum_count == 8
    assert overfit_set.labeled_count == 8
    assert overfit_set.molecule_count == 8


def test_trainer_step_reports_teacher_eval_and_checkpoints(overfit_set, tmp_path):
    trainer = make_trainer()
    assert trainer.step_count == 0
    indices = list(range(8))
    # No report requested: no read, no dict.
    assert trainer.step(overfit_set, indices) is None
    assert trainer.step_count == 1
    trainer.request_report()
    first = trainer.step(overfit_set, indices)
    assert trainer.step_count == 2
    assert set(first) == {
        "step", "loss", "graph", "formula", "spectra",
        "formula_present", "formula_absent", "gold_not_scored",
    }
    assert first["step"] == 2
    assert first["spectra"] == 8
    assert all(math.isfinite(first[k]) for k in ("loss", "graph", "formula"))

    eval_out = trainer.teacher_eval(overfit_set, indices)
    assert eval_out["nll"].dtype == np.float32
    assert eval_out["q"].dtype == np.float32
    assert eval_out["scored_tokens"].dtype == np.uint32
    assert eval_out["gold_slot"].dtype == np.uint32
    assert eval_out["gold_log_prob"].dtype == np.float32
    assert eval_out["spectra"] == 8 and eval_out["slots"] == 16
    assert eval_out["nll"].shape == (8 * 16,)

    path = str(tmp_path / "ms2.m3ck")
    trainer.save(path)
    loaded = ms.Ms2Trainer.load(path, table())
    assert loaded.step_count == trainer.step_count
    assert loaded.table_sha256 == trainer.table_sha256
    assert loaded.train_config.to_json() == trainer.train_config.to_json()
    again = loaded.teacher_eval(overfit_set, indices)
    np.testing.assert_array_equal(again["nll"], eval_out["nll"])
    np.testing.assert_array_equal(again["gold_slot"], eval_out["gold_slot"])
    with pytest.raises(OSError):
        ms.Ms2Trainer.load(str(tmp_path / "missing.m3ck"), table())


# ---------------------------------------------------------------------------
# Rust/Python parity through the example driver
# ---------------------------------------------------------------------------


def run_rust_experiment(out_path):
    cmd = [
        "cargo", "run", "--release",
        "--no-default-features", "--features", "cpu",
        "--example", "ms2_experiment", "--",
        "--train", str(OVERFIT_PATH),
        "--table", str(TABLE_PATH),
        "--name", "parity",
        "--overfit", "8",
        "--steps", "5",
        "--batch", "8",
        "--lr", "3e-4",
        "--seed", "1",
        "--report-every", "1",
        "--gold-conditioning", "composition",
        "--out", str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1700)
    assert proc.returncode == 0, f"ms2_experiment failed:\n{proc.stderr[-4000:]}"
    return json.loads(out_path.read_text())


def test_parity_with_the_rust_example(overfit_set, tmp_path):
    if not OVERFIT_PATH.is_file():
        pytest.skip(f"{OVERFIT_PATH} is not present")
    out_path = tmp_path / "parity.json"
    report = run_rust_experiment(out_path)
    curve = report["loss_curve"]
    assert len(curve) == 5

    trainer = make_trainer()
    indices = list(range(8))
    got = []
    for _ in range(5):
        trainer.request_report()
        got.append(trainer.step(overfit_set, indices))
    assert trainer.step_count == 5
    for own, rust in zip(got, curve):
        assert own["step"] == rust["step"]
        assert own["spectra"] == rust["spectra"] == 8
        for key in ("loss", "graph", "formula"):
            assert math.isclose(own[key], rust[key], rel_tol=1e-6, abs_tol=1e-9), (
                key, own[key], rust[key])
        assert own["formula_present"] == rust["formula_present"]
        assert own["gold_not_scored"] == rust["gold_not_scored"]


# ---------------------------------------------------------------------------
# Error parity: one failing call per reachable Error variant
# ---------------------------------------------------------------------------


def test_error_parity(overfit_set, tmp_path):
    t = table()
    # Error::Config -> ValueError, with the Rust message.
    _, fields = make_batch()
    with pytest.raises(ValueError, match=r"duplicate spectrum_id 7"):
        ms.SpectrumBatch(**{**fields, "spectrum_id": np.array([7, 7], dtype=np.uint64)}).validate()
    with pytest.raises(ValueError, match=r"ModelConfig::validate: max_atoms 64"):
        ms.ModelConfig(max_atoms=64).validate()
    with pytest.raises(ValueError, match=r"TrainConfig::validate: batch is 0"):
        ms.TrainConfig(batch=0).validate()

    # Error::Json -> ValueError.
    with pytest.raises(ValueError, match="json error"):
        ms.ModelConfig.from_json("{")
    with pytest.raises(ValueError, match=".*"):
        ms.CandidateBatch.from_json('{"schema_version": 2}')

    # Error::Io -> OSError.
    with pytest.raises(OSError, match="No such file|not found|cannot"):
        ms.ExperimentSet.from_export(str(tmp_path / "missing.json"), t)
    with pytest.raises(OSError, match=".*"):
        ms.Ms2Trainer.load(str(tmp_path / "missing.m3ck"), t)

    # Error::Unsupported -> NotImplementedError.
    with pytest.raises(NotImplementedError, match="Beam"):
        ms.GenerationConfig(mode="beam").validate()
    model = ms.Ms2Model(ms.ModelConfig(), t, seed=0)
    batch, _ = make_batch()
    with pytest.raises(NotImplementedError, match="Beam"):
        model.generate(batch, ms.GenerationConfig(
            trajectories=2, formulas=1, mode="beam"))

    # Error::StateDict -> RuntimeError: a checkpoint with a weight removed.
    trainer = make_trainer()
    path = tmp_path / "w.m3ck"
    trainer.save(str(path))
    saved = json.loads(path.read_text())
    first_key = next(iter(saved["weights"]["entries"]))
    del saved["weights"]["entries"][first_key]
    broken = tmp_path / "broken.m3ck"
    broken.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="missing entry"):
        ms.Ms2Trainer.load(str(broken), t)
