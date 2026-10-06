"""Fingerprint sidecars for the completion model (task MC20a).

Conditions the completion model on MIST's 4096-bit fingerprint (`morgan4096`)
instead of functional groups. Three subcommands:

- ``bits``: true fingerprint on-bit lists for every molecule of a molecule
  export file (``tools/ms2/export_casmi_molecules.py`` schema).
- ``noise``: compares MIST predicted probabilities against the true
  fingerprints of the labelled panel compounds (the "noise model" the Rust
  trainer samples from).
- ``panel``: evaluation file in the molecule-export schema with real MIST
  predictions per panel molecule.

Fingerprint definition (vendored
``mist_fp/mist_src/mist/data/featurizers.py``: ``FingerprintFeaturizer`` with
``fp_names=['morgan4096']`` routes to ``_get_morgan_4096``, i.e.
``_get_morgan_fp_base(mol, nbits=4096)`` with the default ``radius=2``, i.e.
``AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)`` applied to
``Chem.MolFromSmiles(smiles)``):

    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)

``bits`` output entries are keyed ``"<key>|<identity_group>"`` (the export's
molecule ``key`` is the 14-char InChIKey, which can repeat across
stereoisomers/rows, so the ``identity_group`` disambiguates); both the train
and validation outputs use this same key format.

Duplicate keys: ``structures.parquet`` itself holds several (inchikey14,
identity_group) groups with distinct SMILES (mostly same-formula tautomer /
protomer variants, 1612 groups in the full table). A JSON object cannot hold
both, so ``bits`` keeps the first molecule in file order per key and reports
``n_molecules`` (rows read), ``n_unique_keys`` (entries written),
``n_duplicate_keys`` (key groups with >1 row) and, among those,
``duplicate_keys_fp_identical`` (groups whose rows all share one fingerprint).
The ``bits`` verification compares ``fingerprint_bits`` against the vendored
class directly on 200 seeded random molecules, independent of this dedup.

``bits`` also writes ``bits_by_molecule``: a list aligned with the molecule
order of the export file (``export["molecules"][i]`` <-> ``bits_by_molecule[i]``),
so keyed-dedup collisions (685 training keys shared by several different
structures with different fingerprints) cannot misattribute a fingerprint.
The Rust trainer looks fingerprints up by this molecule index
(``CompletionExample.source_index``) and asserts the file's molecule count
equals the export's.

Conventions used in ``noise`` (also documented in the output JSON):

- histograms: 20 equal-width bins over [0, 1]; bin ``i`` covers
  ``[i/20, (i+1)/20)``, the last bin includes 1.0
  (``min(int(p * 20), 19)``). Two histogram sets are exported:
  ``hist_pred_given_true_on`` / ``hist_pred_given_true_off`` pool every
  spectrum's prediction (per-spectrum setting), while
  ``hist_pred_given_true_on_molecule`` /
  ``hist_pred_given_true_off_molecule`` pool the per-molecule averaged
  probabilities (one row per molecule, the averaged-panel deployment
  setting). The Rust sampler selects the set with
  ``--fp-noise-level spectrum|molecule`` (default ``spectrum``).
- a predicted bit is "on" at threshold ``t`` when ``p >= t``.
- Tanimoto is ``|A & B| / |A | B|`` (1.0 when both sets are empty).
- precision is ``TP / (TP + FP)`` (1.0 when the prediction is empty),
  recall is ``TP / (TP + FN)`` (1.0 when the truth is empty); reported
  values are macro averages (mean over spectra, resp. over molecules).
- "spectrum level" scores each spectrum's prediction against its own
  spectrum's structure truth (via ``--panel-spectra`` when given, else
  the structures table); "molecule level" first averages the predicted
  probabilities of all spectra of one molecule, then thresholds and
  scores once per molecule.
- truth association: each spectrum's truth is the fingerprint of its own
  structure from ``panel_spectra.parquet`` (``spec_id -> smiles``) when
  ``--panel-spectra`` is given. Panel molecules whose spectra map to
  more than one distinct fingerprint are dropped as ``ambiguous_structure``
  (counted, never silently first-wins).
- ``panel --per-spectrum`` writes one evaluation entry per spectrum (the
  same molecule repeated with that spectrum's own predicted
  probabilities and its own structure, same identity group), because in
  the competition each query is a single spectrum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

FINGERPRINT = "morgan4096"
N_BITS = 4096
RADIUS = 2
DEFINITION = "AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)"
KEY_FORMAT = "<key>|<identity_group>"
N_BINS = 20
THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
TANIMOTO_THRESHOLD = 0.5
SPARSE_MIN_PROB = 0.01
SPARSE_PROB_NDIGITS = 4
VERIFY_N = 200
VERIFY_SEED = 20261006

# Same constants as tools/ms2/export_casmi_molecules.py.
CHEMISTRY = "ms2-chem-v0.1"
SCHEMA_VERSION = 1


def entry_key(key: str, identity_group: int) -> str:
    """Output-dict key for one molecule: ``"<key>|<identity_group>"``."""
    return f"{key}|{int(identity_group)}"


def fingerprint_bits(smiles: str) -> list[int]:
    """Sorted on-bit indices of the vendored ``morgan4096`` fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return sorted(int(i) for i in np.nonzero(arr)[0])


_VENDORED_FEATURIZERS: dict = {}


def vendored_bits(smiles: str, mist_fp_dir: str | Path) -> list[int]:
    """On-bit list from the vendored featurizer class called directly.

    Imports the competition's ``pl_shim`` (stubs for the missing
    ``pytorch_lightning``/``h5py`` training deps) and instantiates
    ``FingerprintFeaturizer(fp_names=['morgan4096'])``. Only used to verify
    ``fingerprint_bits``; the main path never touches vendored code.
    """
    import sys

    d = Path(mist_fp_dir)
    for entry in (str(d), str(d / "mist_src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import pl_shim  # noqa: F401
    from mist.data.featurizers import FingerprintFeaturizer

    key = str(d)
    if key not in _VENDORED_FEATURIZERS:
        _VENDORED_FEATURIZERS[key] = FingerprintFeaturizer(fp_names=[FINGERPRINT])
    arr = _VENDORED_FEATURIZERS[key].featurize_smiles(smiles)
    return sorted(int(i) for i in np.nonzero(arr)[0])


def default_mist_fp_dir(export_path: Path) -> Path:
    """``mist_fp`` lives next to the competition ``data`` dir."""
    return export_path.parent.parent.parent / "mist_fp"


def histogram_counts(probs: np.ndarray, n_bins: int = N_BINS) -> list[int]:
    """Counts of values over ``n_bins`` equal-width bins over [0, 1]."""
    idx = np.minimum((np.asarray(probs, dtype=np.float64) * n_bins).astype(int),
                     n_bins - 1)
    return [int(c) for c in np.bincount(idx, minlength=n_bins)]


def tanimoto(on_pred: set[int], on_true: set[int]) -> float:
    """Tanimoto between two on-bit sets (1.0 when both are empty)."""
    union = len(on_pred | on_true)
    if union == 0:
        return 1.0
    return len(on_pred & on_true) / union


def precision_recall(on_true: set[int], on_pred: set[int]) -> tuple[float, float]:
    """(precision, recall); empty-pred precision and empty-truth recall are 1.0."""
    tp = len(on_true & on_pred)
    precision = tp / len(on_pred) if on_pred else 1.0
    recall = tp / len(on_true) if on_true else 1.0
    return precision, recall


def summarize(values: list[float]) -> dict:
    """Mean/median/10th/90th percentile summary of a list of floats."""
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90))}


def _mask_of(on: set[int], n_bits: int) -> np.ndarray:
    mask = np.zeros(n_bits, dtype=bool)
    if on:
        mask[list(on)] = True
    return mask


def compute_noise(truth: list[set[int]], probs: np.ndarray,
                  mol_index: np.ndarray,
                  thresholds: tuple[float, ...] = THRESHOLDS) -> dict:
    """Noise statistics comparing predicted probabilities to true bit sets.

    ``truth[i]`` is the on-bit set of spectrum ``i``'s own structure,
    ``probs[i]`` its 4096 predicted probabilities, ``mol_index[i]`` the
    molecule position of spectrum ``i``. Pure function (no I/O).

    Two histogram sets are returned: the per-spectrum histograms pool every
    spectrum's prediction; the ``*_molecule`` histograms pool the
    per-molecule averaged probabilities (one row per molecule, from the
    already-computed ``mol_probs``).
    """
    probs = np.asarray(probs, dtype=np.float64)
    mol_index = np.asarray(mol_index)
    n = len(truth)
    n_bits = probs.shape[1] if probs.ndim == 2 else N_BITS
    on_counts = np.array([len(t) for t in truth], dtype=np.float64)
    prob_mass = probs.sum(axis=1) if n else np.zeros(0)
    on_parts = [probs[i, sorted(t)] for i, t in enumerate(truth) if len(t) > 0]
    on_vals = np.concatenate(on_parts) if on_parts else np.zeros(0)
    if n:
        off_parts = [probs[i, np.nonzero(~_mask_of(t, n_bits))[0]]
                     for i, t in enumerate(truth)]
        off_vals = np.concatenate(off_parts) if off_parts else np.zeros(0)
    else:
        off_vals = np.zeros(0)

    spec_tani = [tanimoto(set(np.nonzero(probs[i] >= TANIMOTO_THRESHOLD)[0]), truth[i])
                 for i in range(n)]
    spec_pr = {}
    for thr in thresholds:
        ps, rs = zip(*[precision_recall(
            truth[i], set(np.nonzero(probs[i] >= thr)[0])) for i in range(n)])
        spec_pr[str(thr)] = {"precision": float(np.mean(ps)),
                             "recall": float(np.mean(rs))}

    order = np.argsort(mol_index, kind="stable")
    _, first = np.unique(mol_index[order], return_index=True)
    groups = np.split(order, np.sort(first)[1:]) if n else []
    mol_truth = [truth[g[0]] for g in groups]
    mol_probs = np.array([probs[g].mean(axis=0) for g in groups]) if groups else \
        np.zeros((0, probs.shape[1]))
    mol_tani = [tanimoto(set(np.nonzero(p >= TANIMOTO_THRESHOLD)[0]), t)
                for p, t in zip(mol_probs, mol_truth)]
    mol_pr = {}
    for thr in thresholds:
        if len(groups):
            ps, rs = zip(*[precision_recall(
                t, set(np.nonzero(p >= thr)[0]))
                for p, t in zip(mol_probs, mol_truth)])
            mol_pr[str(thr)] = {"precision": float(np.mean(ps)),
                                "recall": float(np.mean(rs))}
        else:
            mol_pr[str(thr)] = {"precision": 1.0, "recall": 1.0}

    # Molecule-averaged histograms: pool the already-computed per-molecule
    # averaged probabilities (one row per molecule) against the molecule
    # truth. Empty-truth molecules contribute no on-values.
    if len(groups):
        mol_on_parts = [mol_probs[j, sorted(t)] for j, t in enumerate(mol_truth)
                        if len(t) > 0]
        mol_on_vals = np.concatenate(mol_on_parts) if mol_on_parts else np.zeros(0)
        mol_off_parts = [mol_probs[j, np.nonzero(~_mask_of(t, n_bits))[0]]
                         for j, t in enumerate(mol_truth)]
        mol_off_vals = np.concatenate(mol_off_parts) if mol_off_parts else np.zeros(0)
    else:
        mol_on_vals = np.zeros(0)
        mol_off_vals = np.zeros(0)

    return {
        "n_spectra": n,
        "n_molecules": len(groups),
        "mean_true_on_bits_spectrum": float(np.mean(on_counts)) if n else 0.0,
        "mean_true_on_bits_molecule": float(np.mean([len(t) for t in mol_truth]))
        if mol_truth else 0.0,
        "mean_pred_prob_mass_spectrum": float(np.mean(prob_mass)) if n else 0.0,
        "mean_pred_prob_mass_molecule": float(np.mean(mol_probs.sum(axis=1)))
        if len(mol_probs) else 0.0,
        "hist_pred_given_true_on": histogram_counts(on_vals),
        "hist_pred_given_true_off": histogram_counts(off_vals),
        "hist_pred_given_true_on_molecule": histogram_counts(mol_on_vals),
        "hist_pred_given_true_off_molecule": histogram_counts(mol_off_vals),
        "tanimoto_spectrum": summarize(spec_tani),
        "tanimoto_molecule": summarize(mol_tani),
        "precision_recall_spectrum": spec_pr,
        "precision_recall_molecule": mol_pr,
    }


def sparse_probs(mean_probs: np.ndarray,
                 min_prob: float = SPARSE_MIN_PROB) -> list[list]:
    """``[[bit, probability], ...]`` sorted by bit for ``p >= min_prob``."""
    idx = np.nonzero(np.asarray(mean_probs) >= min_prob)[0]
    return [[int(b), round(float(mean_probs[b]), SPARSE_PROB_NDIGITS)]
            for b in sorted(idx.tolist())]


def typed_graph(smiles: str):
    """Typed ``(atoms, bonds)`` via the shared conversion, or None if excluded.

    Same steps as ``export_casmi_molecules``: parse, kekulize, classify
    (out-of-domain molecules are skipped), then ``ref.graph_of``.
    """
    import ms2_reference as ref

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    if ref.classify(mol):
        return None
    return ref.graph_of(mol)


def read_labels(labels_path: Path) -> list[dict]:
    with open(labels_path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_structures(structures_path: Path) -> dict:
    """``inchikey14 -> row`` (first row wins) with the needed columns."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        structures_path,
        columns=["inchikey14", "smiles", "formula", "n_heavy",
                 "identity_group", "fold_identity"])
    out = {}
    for row in table.to_pylist():
        out.setdefault(row["inchikey14"], row)
    return out


def read_panel_spectra(panel_spectra_path: Path | None) -> dict:
    """``spec_id -> {smiles, molecule, formula}`` from ``panel_spectra.parquet``.

    The authoritative per-spectrum structure link: the dry run's labels and
    spectra were built from the panel bundle (see
    ``mist_fp/export_panel_bundle.py``: ``panel.parquet`` holds the 250 panel
    molecules, ``panel_spectra.parquet`` holds one row per spectrum with its
    own ``smiles``). Empty dict when no path is given (legacy structures-table
    path).
    """
    if panel_spectra_path is None:
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(panel_spectra_path,
                          columns=["spec_id", "molecule", "smiles", "formula"])
    out = {}
    for row in table.to_pylist():
        out[str(row["spec_id"])] = row
    return out


def validate_pred_bundle(bundle: dict) -> tuple[list[str], np.ndarray]:
    """Validated ``(names, preds)`` from a prediction pickle.

    Requires ``names`` and ``preds`` with equal lengths, ``preds`` width
    ``N_BITS`` and every probability in ``[0, 1]`` (anything else is an
    error, never a silent ``zip`` truncation).
    """
    if "names" not in bundle or "preds" not in bundle:
        raise ValueError("preds bundle needs 'names' and 'preds'")
    names = [str(x) for x in bundle["names"]]
    probs = np.asarray(bundle["preds"], dtype=np.float64)
    if len(names) != len(probs):
        raise ValueError(
            f"preds bundle holds {len(names)} names for {len(probs)} rows "
            "(lengths must match; zip truncation is refused)")
    if probs.ndim != 2 or probs.shape[1] != N_BITS:
        shape = probs.shape
        raise ValueError(
            f"preds bundle has shape {shape} (needs (*, {N_BITS}))")
    if probs.size and (np.isnan(probs).any() or (probs < 0.0).any()
                       or (probs > 1.0).any()):
        raise ValueError("preds bundle holds probabilities outside [0, 1]")
    return names, probs


def build_bits(molecules: list[dict]) -> tuple[dict, dict, list[list[int]]]:
    """``{entry_key: sorted on-bit list}`` plus dedup statistics plus the
    per-molecule list.

    First molecule in input order wins per ``<key>|<identity_group>``
    (deterministic for a given export file); every bit list is asserted
    sorted and in range. The third return value is ``bits_by_molecule``,
    aligned with the input molecule order (one sorted on-bit list per
    molecule, including duplicates), so callers can look fingerprints up by
    molecule index instead of the lossy key.
    """
    bits: dict[str, list[int]] = {}
    counts: dict[str, list] = {}
    by_molecule: list[list[int]] = []
    for mol in molecules:
        key = entry_key(mol["key"], mol["identity_group"])
        b = fingerprint_bits(mol["smiles"])
        assert b == sorted(b) and all(0 <= x < N_BITS for x in b)
        by_molecule.append(b)
        if key in bits:
            counts[key][0] += 1
            if bits[key] != b:
                counts[key][1] = False
        else:
            bits[key] = b
            counts[key] = [1, True]
    dup = [v for v in counts.values() if v[0] > 1]
    stats = {"n_molecules": len(molecules), "n_unique_keys": len(bits),
             "n_duplicate_keys": len(dup),
             "n_duplicate_molecules": sum(v[0] - 1 for v in dup),
             "duplicate_keys_fp_identical": sum(1 for v in dup if v[1])}
    return bits, stats, by_molecule


def cmd_bits(export_path: Path, out_path: Path,
             mist_fp_dir: Path | None = None) -> dict:
    doc = json.loads(export_path.read_text())
    molecules = doc.get("molecules", [])
    bits, stats, by_molecule = build_bits(molecules)

    mist_fp_dir = mist_fp_dir or default_mist_fp_dir(export_path)
    rng = np.random.default_rng(VERIFY_SEED)
    sample_idx = rng.choice(len(molecules),
                            size=min(VERIFY_N, len(molecules)),
                            replace=False)
    agree = sum(1 for i in sample_idx
                if vendored_bits(molecules[int(i)]["smiles"], mist_fp_dir)
                == fingerprint_bits(molecules[int(i)]["smiles"]))
    if agree != len(sample_idx):
        raise ValueError(
            f"bits verification failed: vendored agreement "
            f"{agree}/{len(sample_idx)} (refusing to write; the export "
            f"would misstate the fingerprint definition)")
    keys_by_molecule = [entry_key(mol["key"], mol["identity_group"])
                        for mol in molecules]
    payload = {"fingerprint": FINGERPRINT, "definition": DEFINITION,
               "rdkit": rdkit.__version__, **stats, "bits": bits,
               "bits_by_molecule": by_molecule,
               "keys_by_molecule": keys_by_molecule}
    text = json.dumps(payload, separators=(",", ":"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    counts = np.array([len(b) for b in bits.values()], dtype=np.float64)
    print(f"{out_path}: {stats['n_molecules']} molecules, "
          f"{stats['n_unique_keys']} unique keys, {len(text) / 1e6:.1f} MB, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    print(f"  mean on-bits {counts.mean():.1f}, "
          f"duplicate keys {stats['n_duplicate_keys']} "
          f"({stats['duplicate_keys_fp_identical']} fp-identical, "
          f"first-in-file-order kept), "
          f"vendored agreement {agree}/{len(sample_idx)} "
          f"(definition: {DEFINITION})")
    return {"mean_on_bits": float(counts.mean()), "agree": agree,
            "verified": len(sample_idx), **stats}


def cmd_noise(preds_path: Path, labels_path: Path,
              structures_path: Path, out_path: Path,
              panel_spectra_path: Path | None = None) -> dict:
    import pickle

    bundle = pickle.load(open(preds_path, "rb"))
    names, probs_all = validate_pred_bundle(bundle)
    by_spec = {row["spec"]: row for row in read_labels(labels_path)}
    structures = read_structures(structures_path)
    panel_spectra = read_panel_spectra(panel_spectra_path)

    truth, probs, mol_of, skipped = [], [], [], Counter()
    compound_of_row: dict[str, int] = {}
    for name, prob in zip(names, probs_all):
        row = by_spec.get(name)
        if row is None:
            skipped["no_label"] += 1
            continue
        # Authoritative truth: the spectrum's own structure from the panel
        # bundle when available; else the structures table (first wins).
        spec_link = panel_spectra.get(name)
        if spec_link is not None:
            smiles = spec_link["smiles"]
        else:
            struct = structures.get(row["compound"])
            if struct is None:
                skipped["not_in_structures"] += 1
                continue
            smiles = struct["smiles"]
        try:
            on = set(fingerprint_bits(smiles))
        except ValueError:
            skipped["unparsable_smiles"] += 1
            continue
        if row["compound"] not in compound_of_row:
            compound_of_row[row["compound"]] = len(compound_of_row)
        truth.append(on)
        probs.append(prob)
        mol_of.append(compound_of_row[row["compound"]])
    stats = compute_noise(truth, np.array(probs),
                          np.array(mol_of, dtype=int))
    payload = {"fingerprint": FINGERPRINT, "definition": DEFINITION,
               "rdkit": rdkit.__version__, "n_bits": N_BITS,
               "n_bins": N_BINS,
               "bin_edges": [i / N_BINS for i in range(N_BINS + 1)],
               "bin_convention": "bin i covers [i/20, (i+1)/20); last bin "
                                 "includes 1.0",
               "thresholds": list(THRESHOLDS),
               "threshold_convention": "bit on when p >= threshold",
               "tanimoto_threshold": TANIMOTO_THRESHOLD,
               "aggregation": "spectrum level scores each spectrum vs its "
                              "molecule truth; molecule level averages "
                              "probabilities over a molecule's spectra first; "
                              "precision/recall are macro (mean) averages; hist_*_molecule pool the per-molecule averaged probabilities (averaged-panel setting, selected in Rust with --fp-noise-level molecule)",
                "noise_levels": ["spectrum", "molecule"],
               "skipped_spectra": dict(skipped),
               **stats}
    text = json.dumps(payload, separators=(",", ":"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"{out_path}: {stats['n_spectra']} spectra, "
          f"{stats['n_molecules']} molecules, {len(text) / 1e3:.1f} kB")
    print(f"  mean true on-bits spectrum {stats['mean_true_on_bits_spectrum']:.1f} / "
          f"molecule {stats['mean_true_on_bits_molecule']:.1f}; "
          f"mean pred mass spectrum {stats['mean_pred_prob_mass_spectrum']:.1f} / "
          f"molecule {stats['mean_pred_prob_mass_molecule']:.1f}")
    for level in ("spectrum", "molecule"):
        t = stats[f"tanimoto_{level}"]
        print(f"  tanimoto@{TANIMOTO_THRESHOLD} {level}: mean {t['mean']:.4f} "
              f"median {t['median']:.4f} p10 {t['p10']:.4f} p90 {t['p90']:.4f}")
        for thr, pr in stats[f"precision_recall_{level}"].items():
            print(f"  {level} thr {thr}: precision {pr['precision']:.4f} "
                  f"recall {pr['recall']:.4f}")
    if skipped:
        print(f"  skipped spectra: {dict(skipped)}")
    return payload


def cmd_panel(preds_path: Path, labels_path: Path, structures_path: Path,
              out_path: Path, max_heavy: int = 32,
              panel_spectra_path: Path | None = None,
              per_spectrum: bool = False) -> dict:
    import pickle

    bundle = pickle.load(open(preds_path, "rb"))
    names, probs_all = validate_pred_bundle(bundle)
    by_spec: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        by_spec.setdefault(name, []).append(i)
    label_rows = read_labels(labels_path)
    spec_row = {row["spec"]: row for row in label_rows}
    structures = read_structures(structures_path)

    panel_spec = read_panel_spectra(panel_spectra_path)
    compounds: dict[str, str] = {}
    for row in label_rows:
        compounds.setdefault(row["compound"], row["formula"])
    # Spectra of each compound that carry a prediction.
    compound_spec_names: dict[str, list[str]] = {}
    for row in label_rows:
        for _ in by_spec.get(row["spec"], []):
            compound_spec_names.setdefault(row["compound"], []).append(row["spec"])
    # Distinct predicted rows per compound (for the missing policy count).
    compound_pred_rows: dict[str, list[int]] = {}
    for compound in compounds:
        seen: list[int] = []
        for name, row in spec_row.items():
            if row["compound"] == compound:
                seen.extend(by_spec.get(name, []))
        compound_pred_rows[compound] = seen

    def spectrum_smiles(name: str, row: dict, struct: dict) -> str:
        link = panel_spec.get(name)
        if link is not None:
            return link["smiles"]
        return struct["smiles"]

    def append_entry(compound: str, smiles: str, struct: dict, prob_vec: np.ndarray,
                     n_spectra: int, spec_id=None) -> bool:
        try:
            true_bits = sorted(set(fingerprint_bits(smiles)))
        except ValueError:
            skipped["unparsable_smiles"] += 1
            return False
        graph = typed_graph(smiles)
        if graph is None:
            skipped["graph_excluded"] += 1
            return False
        atoms, bonds = graph
        entry = {"key": compound, "smiles": smiles,
                 "identity_group": int(struct["identity_group"]),
                 "fold_identity": int(struct["fold_identity"]),
                 "atoms": atoms, "bonds": [list(b) for b in bonds],
                 "fp_true": true_bits,
                 "fp_pred_mean": sparse_probs(prob_vec),
                 "spectra": n_spectra, "formula": compounds[compound]}
        if spec_id is not None:
            entry["spec_id"] = spec_id
        panel_molecules.append(entry)
        return True

    panel_molecules = []
    skipped = Counter()
    folds: Counter = Counter()
    heavy_ok = 0
    missing_predictions = 0
    for compound in sorted(compounds):
        struct = structures.get(compound)
        if struct is None:
            skipped["not_in_structures"] += 1
            continue
        folds[int(struct["fold_identity"])] += 1
        if int(struct["n_heavy"]) > max_heavy:
            skipped["too_many_atoms"] += 1
            continue
        heavy_ok += 1
        pred_idx = compound_pred_rows.get(compound, [])
        if per_spectrum:
            # One entry per spectrum: the molecule repeated with that
            # spectrum's own predicted probabilities and its own structure
            # (same identity group), because each competition query is a
            # single spectrum.
            spec_names = compound_spec_names.get(compound, [])
            if not spec_names:
                missing_predictions += 1
                skipped["missing_predictions"] += 1
                continue
            for name in sorted(spec_names):
                row = spec_row[name]
                smiles = spectrum_smiles(name, row, struct)
                for i in by_spec.get(name, []):
                    append_entry(compound, smiles, struct, probs_all[i], 1,
                                 spec_id=name)
            continue
        # Molecule-averaged entry: associate every spectrum with its own
        # structure; a compound whose spectra map to more than one distinct
        # fingerprint stays ambiguous and is dropped (counted).
        spec_names = compound_spec_names.get(compound, [])
        fp_of: dict[str, list[int]] = {}
        for name in spec_names:
            try:
                fp_of[name] = sorted(set(
                    fingerprint_bits(spectrum_smiles(name, spec_row[name], struct))))
            except ValueError:
                fp_of[name] = []
        distinct = {tuple(v) for v in fp_of.values()}
        if len(distinct) > 1:
            skipped["ambiguous_structure"] += 1
            continue
        smiles = spectrum_smiles(spec_names[0], spec_row[spec_names[0]], struct) \
            if spec_names else struct["smiles"]
        idx = pred_idx
        # Explicit missing-prediction policy: no predicted row means an empty
        # (all-epsilon) fingerprint, counted instead of silently kept.
        if not idx:
            missing_predictions += 1
            skipped["missing_predictions"] += 1
        mean_p = probs_all[idx].mean(axis=0) if idx else np.zeros(N_BITS)
        append_entry(compound, smiles, struct, mean_p, len(idx))
    payload = {"schema_version": SCHEMA_VERSION, "chemistry": CHEMISTRY,
               "rdkit": rdkit.__version__,
               "source": "MIST dry-run panel predictions + folds/structures.parquet"
                         + (" + panel_spectra.parquet per-spectrum structures"
                            if panel_spectra_path is not None else "")
                         + (" (per-spectrum entries)" if per_spectrum else ""),
               "per_spectrum": per_spectrum,
               "missing_predictions": missing_predictions,
               "fingerprint": FINGERPRINT, "definition": DEFINITION,
               "max_heavy": max_heavy,
               "panel_identity_groups": sorted(
                   {m["identity_group"] for m in panel_molecules}),
               "skipped_panel_molecules": dict(skipped),
               "molecules": panel_molecules}
    text = json.dumps(payload, separators=(",", ":"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    n_panel = len(compounds)
    n_found = sum(folds.values())
    print(f"{out_path}: {len(panel_molecules)} molecules, "
          f"{len(text) / 1e6:.1f} MB")
    print(f"  panel compounds {n_panel}, in structures {n_found}, "
          f"folds {dict(sorted(folds.items()))}, "
          f"at most {max_heavy} heavy atoms {heavy_ok}, "
          f"typed graph {len(panel_molecules)}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return {"n_panel": n_panel, "n_found": n_found,
            "folds": dict(sorted(folds.items())), "heavy_ok": heavy_ok,
            "written": len(panel_molecules),
            "per_spectrum": per_spectrum,
            "missing_predictions": missing_predictions,
            "ambiguous": skipped.get("ambiguous_structure", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bits", help="true fingerprint bits for an export file")
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mist-fp", type=Path, default=None,
                   help="competition mist_fp dir (default: derived from --export)")

    p = sub.add_parser("noise", help="prediction-vs-truth noise statistics")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--panel-spectra", type=Path, default=None,
                   help="panel_spectra.parquet for per-spectrum truth")

    p = sub.add_parser("panel", help="evaluation file with real MIST predictions")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-heavy", type=int, default=32)
    p.add_argument("--panel-spectra", type=Path, default=None,
                   help="panel_spectra.parquet for per-spectrum structures")
    p.add_argument("--per-spectrum", action="store_true",
                   help="one evaluation entry per spectrum")

    args = ap.parse_args()
    if args.cmd == "bits":
        cmd_bits(args.export, args.out, args.mist_fp)
    elif args.cmd == "noise":
        cmd_noise(args.preds, args.labels, args.structures, args.out,
                  args.panel_spectra)
    elif args.cmd == "panel":
        cmd_panel(args.preds, args.labels, args.structures, args.out,
                  args.max_heavy, args.panel_spectra, args.per_spectrum)


if __name__ == "__main__":
    main()
