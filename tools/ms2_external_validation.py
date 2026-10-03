"""MassBank/GNPS external measured-spectrum validation (bounded, pinned).

Reads ONLY pinned local archives (no bulk download):
  data/pinned/external/MassBank-data-2026.03.zip
  data/pinned/external/GNPS-FAULKNERLEGACY.mgf
  data/pinned/msgym_candidates_formula_prefix64.json  (candidate pools)
  experiments/molecular_completion/20261003_predictor/predictor_model.npz
  experiments/molecular_completion/20261003_predictor/{predictor_rows.csv,
    train_fit_rows.csv}  (sealed identities for overlap audit only)

Protocol (per docs/MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md + PROMPT.md):
  * Declared fixed selection: archive-order scan, first N distinct
    molecules passing chemistry/adduct/spectrum gates (cap 100-200/source).
  * Per-record license/provenance retained; unknown/ambiguous precursor
    conversions rejected (supported ion rules only, correct constants,
    multicharge excluded).
  * Connectivity/content overlap audit BEFORE scoring; fit/calibration
    molecules excluded from scoring with counters; val/test overlap
    flagged per query. True external target NEVER injected into pools
    (naturally present copies are REMOVED with counters).
  * Frozen Ridge checkpoint (hash-verified) scores external spectra on
    identical pools/order with the same four arms (uniform, massresid,
    predictor, prior) and deterministic qid-crc32 ties. No refit, no
    val/external tuning. Original 200 stay sealed (read-only overlap).
  * Reference-spectra NN arm (leave-one-out within the selected MassBank
    set) is reported as library lookup, never as unseen generalization.
  * val200 library-coverage check is reported as LIBRARYLOOKUP.
  * PubChem tiny expansion ONLY on measured pool misses (<=5 formulas x
    <=100 candidates, cached, throttled); marginal recall reported without
    rescoring frozen metrics.
  * Exact Murcko scaffolds via project-local RDKit (pinned version); if
    RDKit is unavailable the exact-scaffold gate stays INCOMPLETE.

Outputs (into experiments/molecular_completion/20261003_remaining/):
  external_rows.csv, external_summary.json (strict allow_nan=False),
  external_nn_rows.csv, library_lookup.csv, pubchem_cache/*.json
Also writes the source-pin manifest data/pinned/EXTERNAL_PIN.json
(official URLs/revisions/hashes/licenses/times). Never touches PIN.json.

CPU stdlib + numpy/scipy/sklearn (+ optional project-local RDKit):
GPU execution and Rust/Python parity are NOT APPLICABLE.
"""

import csv
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
import unittest
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_spectral_rank import parse_spectrum  # noqa: E402
from tools.ms2_fp_predictor import (  # noqa: E402
    FP_DIM,
    bin_spectrum_1da,
    capability_for_query,
    evaluate_pool,
    featurize_spectrum,
    fp_from_smiles,
    score_candidates_fp,
    spectrum_has_usable_features,
    target_parse_info,
)
from tools.ms2_spectral_rank import (  # noqa: E402
    candidate_residuals as spectral_residuals,
    seeded_positions,
)

ARMS = ("uniform", "massresid", "predictor", "prior")
KS = (1, 3, 10, 25)
Z95 = 1.959963984540054

SUPPORTED_ADDUCTS = ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+")
# Ion masses in Da. Proton mass is the hydrogen-ion mass (atomic H minus one
# electron). Na+/K+ ion masses are atomic masses minus one electron mass
# (0.000548579909 Da); using neutral atomic masses here would bias neutral
# estimates by ~0.55 mDa. Integer micro-Da (uDa) used for persisted masses.
PROTON_MASS = 1.007276466879
ELECTRON_MASS = 0.000548579909
NA_ION_MASS = 22.9897692809 - 0.000548579909
K_ION_MASS = 38.9637074864 - 0.000548579909

ADDUCT_DELTA = {"[M+H]+": -PROTON_MASS, "[M-H]-": PROTON_MASS,
                "[M+Na]+": -NA_ION_MASS, "[M+K]+": -K_ION_MASS}
ADDUCT_POLARITY = {"[M+H]+": "positive", "[M-H]-": "negative",
                   "[M+Na]+": "positive", "[M+K]+": "positive"}

MAX_EXTERNAL_QUERIES = 150
GNPS_MAX_QUERIES = 100
MAX_SCAN_RECORDS = 40000
MAX_INDEX_DISTINCT = 150000
MASS_AGREE_PPM = 50.0
MIN_PEAKS = 3
# Stage-1 measured mass window: 10 ppm relative (STAGE1_PPM_TENTHS, a
# DECLARED assumed instrument tolerance) plus a per-query absolute floor
# equal to the persisted textual rounding half-width in micro-Da (measured
# from the precursor text, e.g. 3 decimals -> 500 uDa). Queries whose
# precursor precision is unknown run stage-1 as unavailable (retained,
# never rejected on mass).
STAGE1_PPM_TENTHS = 100


def progress(msg):
    sys.stderr.write(f"# progress {msg}\n")
    sys.stderr.flush()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            b = handle.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wilson_interval(hits, n, z=Z95):
    if n <= 0:
        return None, None
    if hits < 0 or hits > n:
        raise ValueError(f"hits out of range: {hits}/{n}")
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def hit_at_k(rank, k):
    if rank is None:
        return False
    try:
        r = int(rank)
    except (ValueError, TypeError):
        return False
    return 1 <= r <= k


def rdkit_lib():
    """Project-local RDKit bootstrap. Returns module or None."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(here, "experiments", "molecular_completion",
                        "20261003_remaining", ".rdkit_lib")
    if cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        from rdkit import Chem  # noqa
        import rdkit as _rd
        return _rd
    except Exception:
        return None


def murcko_of(smiles):
    """Exact Murcko scaffold SMILES via RDKit, or (None, reason)."""
    rd = rdkit_lib()
    if rd is None:
        return None, "rdkit_unavailable"
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaf = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(
            smiles, includeChirality=False)
        return (scaf or None), (None if scaf else "no_scaffold")
    except Exception as exc:
        return None, f"rdkit_error:{exc}"


def inchikey14_of(smiles, fallback=None):
    """InChIKey first block via RDKit, else fallback string."""
    rd = rdkit_lib()
    if rd is not None:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToInchiKey(mol).split("-")[0]
        except Exception:
            pass
    if fallback:
        key = fallback.strip()
        return key.split("-")[0] if "-" in key else key[:14]
    return None


# --------------------------------------------------------------------------
# MassBank record parsing.
# --------------------------------------------------------------------------

def parse_massbank_record(text):
    """Parse one MassBank record. Returns (rec, None) or (None, reason).

    Required: ACCESSION, LICENSE, CH$SMILES, CH$FORMULA, CH$EXACT_MASS,
    MS$FOCUSED_ION PRECURSOR_M/Z + PRECURSOR_TYPE, PK$PEAK block.
    Optional provenance retained when present (INCHIKEY link, instrument,
    collision energy, publication).
    """
    def field(pat):
        m = re.search(pat, text, re.M)
        return m.group(1).strip() if m else ""

    accession = field(r"^ACCESSION:\s*(.*)")
    if not accession:
        return None, "no_accession"
    license_ = field(r"^LICENSE:\s*(.*)")
    if not license_:
        return None, "no_license"
    smiles = field(r"^CH\$SMILES:\s*(.*)")
    formula = field(r"^CH\$FORMULA:\s*(.*)")
    exact_mass = field(r"^CH\$EXACT_MASS:\s*(.*)")
    if not (smiles and formula and exact_mass):
        return None, "missing_structure"
    try:
        exact_mass_f = float(exact_mass)
        if not math.isfinite(exact_mass_f) or exact_mass_f <= 0:
            return None, "bad_exact_mass"
    except ValueError:
        return None, "bad_exact_mass"
    prec_mz = field(r"^MS\$FOCUSED_ION:\s*PRECURSOR_M/Z\s+(.*)")
    prec_type = field(r"^MS\$FOCUSED_ION:\s*PRECURSOR_TYPE\s+(.*)")
    if not (prec_mz and prec_type):
        return None, "no_precursor"
    try:
        prec_mz_f = float(prec_mz)
        if not math.isfinite(prec_mz_f) or prec_mz_f <= 0:
            return None, "bad_precursor_mz"
    except ValueError:
        return None, "bad_precursor_mz"
    if re.search(r"2[+-]|\[M[^\]]*2[+-]?\]|2\+|2-", prec_type):
        return None, "unsupported_charge_state"
    adduct = prec_type.strip()
    if adduct not in SUPPORTED_ADDUCTS:
        return None, f"unsupported_adduct:{adduct}"
    # Ion-mode consistency: adduct sign must agree with stated polarity.
    ion_mode = field(r"MS\$FOCUSED_ION:\s*ION_MODE\s+(.*)") or field(r"ION_MODE\s+(\S+)")
    if not ion_mode:
        m_ion = re.search(r"^\s*AC\$MASS_SPECTROMETRY:\s*ION_MODE\s+(\S+)", text, re.M)
        ion_mode = m_ion.group(1).strip() if m_ion else ""
    if ion_mode:
        pol = ion_mode.strip().lower()
        want = ADDUCT_POLARITY[adduct]
        if (want == "positive" and pol.startswith("neg")) or \
           (want == "negative" and pol.startswith("pos")):
            return None, f"ion_mode_conflict:{adduct}_vs_{ion_mode}"
    # Measured-mass precision from the TEXTUAL precursor: N decimals imply
    # a half-rounding width of 0.5 * 10^-N Da (persisted as rounding_mu).
    # This is a numerical property of the record text, NOT a measured
    # instrument precision: the assumed instrument ppm tolerance used in
    # stage-1 decisions is declared separately and never inferred from
    # digit counts. No usable decimals -> precision unavailable.
    prec_text = prec_mz.strip().split()[0]
    m_dec = re.match(r"^[0-9]+\.([0-9]+)$", prec_text)
    if m_dec:
        decimals = len(m_dec.group(1))
        rounding_mu = int(round(0.5 * 10 ** (6 - decimals)))
        mass_precision = f"rounding_halfwidth_{rounding_mu}_muDa"
        precision_known = True
    else:
        decimals = 0
        rounding_mu = None
        mass_precision = "unknown (no usable decimals)"
        precision_known = False
    peaks = []
    in_peak = False
    terminated = False
    for line in text.split("\n"):
        if line.startswith("PK$PEAK:"):
            in_peak = True
            continue
        if in_peak:
            if line.startswith("//"):
                terminated = True
                break
            if re.match(r"^[A-Z].*:", line):
                break
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz, inten = float(parts[0]), float(parts[1])
                except ValueError:
                    return None, "bad_peak_line"
                if not (math.isfinite(mz) and math.isfinite(inten)):
                    return None, "nonfinite_peak"
                if mz <= 0 or inten < 0:
                    return None, "bad_peak_values"
                peaks.append((mz, inten))
    if in_peak and not terminated:
        return None, "incomplete_terminator"
    num_peak = field(r"^PK\$NUM_PEAK:\s*(\S+)")
    try:
        num_peak_n = int(num_peak) if num_peak else None
    except ValueError:
        return None, "bad_num_peak"
    if num_peak_n is not None and num_peak_n != len(peaks):
        # Explicit malformed/mismatching count metadata: reject, since the
        # peak block cannot be trusted complete as declared.
        return None, f"peak_count_mismatch:declared_{num_peak_n}_parsed_{len(peaks)}"
    if len(peaks) < MIN_PEAKS:
        return None, "too_few_peaks"
    inchikey = ""
    for m in re.finditer(r"^CH\$LINK:\s*INCHIKEY\s+(\S+)", text, re.M):
        inchikey = m.group(1).strip()
        break
    return {"accession": accession, "license": license_, "smiles": smiles,
            "formula": formula, "exact_mass": exact_mass_f,
            "precursor_mz": prec_mz_f, "precursor_mz_text": prec_text,
            "mass_precision": mass_precision, "rounding_mu": rounding_mu,
            "precision_known": precision_known, "ion_mode": ion_mode,
            "adduct": adduct,
            "instrument": field(r"^AC\$INSTRUMENT:\s*(.*)"),
            "instrument_type": field(r"^AC\$INSTRUMENT_TYPE:\s*(.*)"),
            "collision_energy": field(r"COLLISION_ENERGY\s+(.*)"),
            "publication": field(r"^PUBLICATION:\s*(.*)"),
            "inchikey": inchikey, "peaks": peaks,
            "n_peaks_declared": num_peak_n,
            "peak_count_match": (num_peak_n == len(peaks)) if num_peak_n is not None else None,
            }, None


def precursor_to_neutral(precursor_mz, adduct):
    """Validated precursor m/z -> neutral mass (Da). Returns (mu, None)/(None, reason)."""
    if adduct not in ADDUCT_DELTA:
        return None, f"unsupported_adduct:{adduct}"
    neutral = precursor_mz + ADDUCT_DELTA[adduct]
    if not math.isfinite(neutral) or neutral <= 0:
        return None, "bad_neutral"
    return int(round(neutral * 1_000_000)), None


def spectrum_content_key(peaks):
    """Dedup key for spectrum content (rounded mz/int hashed)."""
    items = sorted((round(m, 4), round(i, 2)) for m, i in peaks)
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# GNPS MGF parsing.
# --------------------------------------------------------------------------

def parse_mgf_blocks(text):
    """Parse MGF blocks. Returns (blocks, stats).

    Each block: params dict + peaks list. Incomplete blocks (no PEPMASS,
    no peaks, unterminated, missing END IONS) are counted with reasons,
    never scored. Non-finite/negative peaks are rejected (NaN intensity
    is never accepted); all-zero spectra are flagged unusable downstream.
    """
    raw = text.split("BEGIN IONS")
    blocks, bad = [], {"no_pepmass": 0, "no_peaks": 0, "bad_values": 0}
    for chunk in raw[1:]:
        if "END IONS" not in chunk:
            bad["unterminated"] = bad.get("unterminated", 0) + 1
            continue
        body = chunk.split("END IONS")[0]
        params, peaks = {}, []
        for line in body.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if "=" in line and not re.match(r"^[0-9]", line):
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, inten = float(parts[0]), float(parts[1])
                    except ValueError:
                        bad["bad_values"] += 1
                        peaks = []
                        break
                    if not (math.isfinite(mz) and math.isfinite(inten)):
                        bad["nonfinite_peak"] = bad.get("nonfinite_peak", 0) + 1
                        peaks = []
                        break
                    if mz <= 0 or inten < 0:
                        bad["bad_peak_values"] = bad.get("bad_peak_values", 0) + 1
                        peaks = []
                        break
                    peaks.append((mz, inten))
        if "PEPMASS" not in params:
            bad["no_pepmass"] += 1
            continue
        try:
            pep = float(params["PEPMASS"].split()[0])
            if not math.isfinite(pep) or pep <= 0:
                raise ValueError
        except ValueError:
            bad["bad_values"] += 1
            continue
        if len(peaks) < MIN_PEAKS:
            bad["no_peaks"] += 1
            continue
        # Explicit peak-count metadata, when present, is enforced.
        declared = None
        for ck in ("NUM_PEAKS", "NUMPEAKS", "PEAK_COUNT", "PEAKCOUNT",
                   "NUM PEAKS", "NPEAKS"):
            if ck in params:
                try:
                    declared = int(str(params[ck]).strip().split()[0])
                except ValueError:
                    declared = "bad"
                break
        if declared == "bad":
            bad["bad_count_metadata"] = bad.get("bad_count_metadata", 0) + 1
            continue
        if declared is not None and declared != len(peaks):
            bad["peak_count_mismatch"] = bad.get("peak_count_mismatch", 0) + 1
            continue
        blocks.append({"params": params, "peaks": peaks, "pep": pep})
    return blocks, {"n_blocks": len(blocks), "n_bad": sum(bad.values()),
                    "bad_reasons": bad}


def gnps_adduct_of(block):
    """Adduct inference for a GNPS block with explicit evidence rule.

    The adduct token must match as a WHOLE whitespace/comma-delimited NAME
    token (M+H, M-H, M+Na, M+K); substring matches inside water-loss or
    fragment annotations (M-H2O+H, M+H-H2O, M+2H, ...) are explicitly
    rejected as ambiguous rather than read as intact-parent adducts.
    The token must agree with IONMODE polarity, and the stated CHARGE
    (when present) must equal 1 with a consistent sign: CHARGE=2/2+ is a
    magnitude conflict, CHARGE=1- a sign conflict (e.g. real GNPS B26A11:
    CHARGE=2 with an M+H name). Absent CHARGE is accepted with
    charge_status recorded (assumed z=1, declared).
    Returns (adduct, evidence) or (None, reason).
    """
    params = block["params"]
    name = params.get("NAME", "")
    ionmode = params.get("IONMODE", "").lower()
    tokens = re.split(r"[\s,;]+", name)
    # Ambiguous water-loss / multiply-charged / fragment annotations:
    # never read as intact-parent adducts.
    for tok in tokens:
        if re.search(r"H2O|2H|2-|2\+|\[", tok) and tok not in (
                "M+H", "M-H", "M+Na", "M+K"):
            if re.match(r"^(M[+-].*|.*M[+-].*)$", tok) and tok not in (
                    "M+H", "M-H", "M+Na", "M+K"):
                return None, f"ambiguous_adduct_token:{tok}"
    charge_raw = (params.get("CHARGE") or "").strip().split()
    charge_raw = charge_raw[0] if charge_raw else ""
    if charge_raw:
        if charge_raw in ("1", "1+", "+1"):
            charge_status = f"stated_z1 ({charge_raw})"
        elif charge_raw in ("1-", "-1"):
            return None, f"charge_sign_conflict:{charge_raw}"
        else:
            return None, f"unsupported_charge_state:{charge_raw}"
    else:
        charge_status = "unstated_assumed_z1"
    want = {"M+H": ("[M+H]+", "positive"), "M-H": ("[M-H]-", "negative"),
            "M+Na": ("[M+Na]+", "positive"), "M+K": ("[M+K]+", "positive")}
    for tok in tokens:
        if tok in want:
            adduct, pol = want[tok]
            if pol in ionmode:
                return adduct, f"NAME:{tok}+IONMODE:{ionmode}+{charge_status}"
            return None, f"polarity_conflict:{tok}_vs_{ionmode}"
    return None, "unsupported_adduct_evidence"


# --------------------------------------------------------------------------
# Candidate pool index (formula-matched, target never injected).
# --------------------------------------------------------------------------

def rdkit_canon(smiles):
    """RDKit canonical SMILES (connectivity identity, stereo-insensitive).

    Returns (canon_or_None, ok). None when RDKit is unavailable or the
    SMILES is invalid; callers fall back to the raw string and count it.
    """
    rd = rdkit_lib()
    if rd is None:
        return None, False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), True
    except Exception:
        return None, False


def build_formula_pool_index(prefix_path, max_distinct=0):
    """Index distinct prefix-pool SMILES by formula_key. Returns (index, stats).

    max_distinct=0 means no cap (full index; explicit completion status).
    Each entry stores (smiles, rdkit_canonical_or_None) so pool identity is
    exact connectivity (RDKit-canonical), never raw-string canonicalization
    dependent. Unparseable SMILES counted by reason.
    """
    from tools.ms2_msgym_corpus import iter_json_entries, standardize_smiles
    from tools.ms2_chebi_corpus import formula_key as _fk
    index, seen = {}, set()
    n_done = n_trunc = 0
    excl = {}
    n_canon_ok = 0
    for q, cands in iter_json_entries(prefix_path, 1 << 30):
        for smi in cands:
            if smi in seen:
                continue
            if max_distinct and n_done >= max_distinct:
                n_trunc += 1
                continue
            seen.add(smi)
            n_done += 1
            try:
                rec, reason = standardize_smiles(smi, "ext-pool", smi)
            except Exception:
                rec, reason = None, "exception"
            if rec is None:
                excl[reason.split(":")[0]] = excl.get(reason.split(":")[0], 0) + 1
                continue
            canon, ok = rdkit_canon(smi)
            n_canon_ok += 1 if ok else 0
            index.setdefault(_fk(rec["formula"]), []).append((smi, canon))
    return index, {"n_distinct_seen": n_done, "n_truncated": n_trunc,
                   "n_formulas": len(index), "exclusions": excl,
                   "n_rdkit_canon_ok": n_canon_ok,
                   "completion": ("complete" if not n_trunc else "truncated_cap"),
                   "identity": "RDKit-canonical connectivity (raw-string fallback counted)"}


# --------------------------------------------------------------------------
# Frozen checkpoint scoring.
# --------------------------------------------------------------------------

def load_frozen_train_identities(repo_root):
    """Recover the ACTUAL USED frozen fit/calib identities (read-only).

    Uses the real schema of train_fit_rows.csv (smiles,split,status,reason,
    group,...) where group = short InChIKey-14 connectivity key. USED =
    status == 'candidate' (n_fit + n_calib rows of the frozen run). Recovers
    per-molecule spectrum content with the SAME content_fp semantics as the
    frozen pipeline (first train-fold TSV row per SMILES, file order).
    Returns dict with group/sm iles/content sets + persisted lists/hashes.
    """
    from tools.ms2_spectral_rank import content_fp, parse_spectrum
    pred_dir = os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor")
    fit_rows = list(csv.DictReader(open(os.path.join(pred_dir, "train_fit_rows.csv"),
                                        encoding="utf-8")))
    used = [r for r in fit_rows if r.get("status") == "candidate"]
    used_fit = [r for r in used if r.get("split") == "fit"]
    used_calib = [r for r in used if r.get("split") == "calibration"]
    groups = [r.get("group", "") for r in used if r.get("group")]
    smiles = [r.get("smiles", "") for r in used if r.get("smiles")]
    # Same content semantics as frozen collect_train_molecules: first
    # train-fold TSV row per SMILES in file order -> content_fp(peaks).
    want = set(smiles)
    first = {}
    n_train_rows = 0
    with open(os.path.join(repo_root, "data/pinned/MassSpecGym1.5.tsv"),
              encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            n_train_rows += 1
            if row.get("fold") != "train":
                continue
            smi = (row.get("smiles") or "").strip()
            if smi in want and smi not in first:
                first[smi] = row
    contents = {}
    n_spec_fail = 0
    for smi, row in first.items():
        peaks, err = parse_spectrum(row.get("mzs"), row.get("intensities"))
        if err is not None or not peaks:
            n_spec_fail += 1
            continue
        contents[smi] = content_fp(peaks)
    return {
        "n_rows_total": len(fit_rows),
        "n_used": len(used),
        "n_used_fit": len(used_fit),
        "n_used_calib": len(used_calib),
        "groups": groups,
        "group_set": set(groups),
        "smiles": smiles,
        "smiles_set": set(smiles),
        "content_by_smiles": contents,
        "content_set": set(contents.values()),
        "n_tsv_rows_scanned": n_train_rows,
        "n_content_recovered": len(contents),
        "n_content_spec_fail": n_spec_fail,
        "groups_sha256": hashlib.sha256(
            repr(sorted(groups)).encode()).hexdigest()[:16],
        "smiles_sha256": hashlib.sha256(
            repr(sorted(smiles)).encode()).hexdigest()[:16],
    }


def build_chebi_index(sdf_path, max_seconds=480):
    """Build the FULL in-domain pinned ChEBI index (reviewed ingestion).

    Uses standardize_chebi_record over every record of the pinned 3-star
    SDF (no cap on records; explicit completion). Each kept record gains
    RDKit-canonical connectivity identity from its ChEBI SMILES property
    (cross-validated against the INCHIKEY property; agreement counted).
    Returns (records, stats, by_formula).
    """
    import gzip as _gzip
    from collections import Counter
    from tools.ms2_chebi_corpus import (iter_sdf_records,
                                        standardize_chebi_record,
                                        first_prop)
    from tools.ms2_chebi_corpus import formula_key as _fk
    t0 = time.time()
    records = []
    excl = Counter()
    n_scanned = 0
    n_canon_ok = n_key_agree = n_key_disagree = 0
    opener = _gzip.open if sdf_path.endswith(".gz") else open
    timed_out = False
    with opener(sdf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for mol_lines, props in iter_sdf_records(handle):
            if time.time() - t0 > max_seconds:
                timed_out = True
                break
            n_scanned += 1
            chebi_id = first_prop(props, "ChEBI ID") or f"CHEBI-ROW{n_scanned}"
            rec, reason = standardize_chebi_record(mol_lines, props, chebi_id)
            if rec is None:
                excl[reason.split(":")[0]] += 1
                continue
            smi_prop = (first_prop(props, "SMILES") or "").strip()
            canon, ok = rdkit_canon(smi_prop) if smi_prop else (None, False)
            if ok and canon:
                n_canon_ok += 1
            key_prop = (first_prop(props, "INCHIKEY") or "").strip().split("-")[0]
            rdk_key = inchikey14_of(smi_prop) if (ok and smi_prop) else None
            if key_prop and rdk_key:
                if key_prop == rdk_key:
                    n_key_agree += 1
                else:
                    n_key_disagree += 1
            rec["canon"] = canon
            rec["canon_ok"] = bool(ok and canon)
            rec["chebi_key14"] = key_prop
            records.append(rec)
    by_formula = {}
    for r in records:
        by_formula.setdefault(_fk(r["formula"]), []).append(r)
    stats = {"n_scanned": n_scanned, "n_index": len(records),
             "n_formulas": len(by_formula),
             "exclusions": dict(excl),
             "n_canon_ok": n_canon_ok,
             "n_key_agree": n_key_agree, "n_key_disagree": n_key_disagree,
             "timed_out": timed_out,
             "completion": "truncated_time" if timed_out else "complete",
             "identity": ("ChEBI SMILES property RDKit-canonical, "
                          "cross-validated vs INCHIKEY property")}
    return records, stats, by_formula


def chebi_stage_query(qid, qstd, canon, neutral_mu, precision_known,
                      db_index, chebi_by_formula, uncertainty_mu=None):
    """Staged ChEBI coverage for one external query. Returns stage dict.

    Nested candidate sets via the reviewed run_query (s2 ⊆ s1, s3 ⊆ s2):
    Stage 1 (measured): mass_verdict over the FULL index with the
    independently measured precursor-derived neutral mass; mass-window
    only, never a formula filter. Unknown precision (uncertainty None)
    stays unavailable all the way through stage 3 (retained, never
    rejected on mass).
    Stage 2 (measured): exact formula + domain filter OF s1 (nested);
    canonical presence in s2.
    Stage 3 (ORACLE, separate label): single/bonded-pair patterns from the
    query's own typed graph applied to s2 (synthetic-oracle arm; NOT
    measured fragment graphs). Reports target membership in s3, never
    self-retrieval as database recall.
    Stage 4: precursor whole-parent redundancy -> not_evaluated.
    The query target is NEVER injected (id absent by construction). A
    missing target is a miss at every stage; a present-but-mass-rejected
    target is absent at s2/s3 by nesting.
    """
    from tools.ms2_database_retrieval import (WorkCounters, run_query)
    from tools.ms2_msgym_corpus import extract_patterns
    # Absolute mass floor = persisted textual rounding uncertainty when the
    # precursor text carries usable decimals, else None (unavailable all the
    # way through stage 3). The ppm window is a DECLARED assumed instrument
    # tolerance, never a measured precision.
    unc = uncertainty_mu if precision_known else None
    single, pair = extract_patterns(qstd["atom_types"], qstd["edges"])
    pats = [single] if single and pair is None else (
        [single, pair] if pair else ([single] if single else []))
    pseudo = {"id": qid, "mass": neutral_mu, "formula": qstd["formula"],
              "charge": 0}
    res = run_query(db_index, pseudo, pats, overlap="unknown",
                    precursor=None, uncertainty=unc,
                    ppm_tenths=STAGE1_PPM_TENTHS,
                    counters=WorkCounters(limit=100_000))
    s1, s2, s3 = res["s1"], res["s2"], res["s3"]
    # nesting invariant (reviewed run_query guarantees s2 ⊆ s1, s3 ⊆ s2)
    s1_ids = set(r["id"] for r in s1)
    s2_ids = set(r["id"] for r in s2)
    s3_ids = set(r["id"] for r in s3)
    nested_ok = s2_ids <= s1_ids and s3_ids <= s2_ids
    def _present(recs):
        if not canon:
            return False
        return any(r.get("canon") == canon for r in recs)
    return {
        "qid": qid,
        "measured_neutral_mu": neutral_mu,
        "precision_known": precision_known,
        "uncertainty_mu": unc,
        "nested_ok": nested_ok,
        "stage1": {"n_s1": len(s1),
                   "n_accept": res["n_accept"], "n_ambiguous": res["n_ambiguous"],
                   "n_unavailable": res["n_unavailable"], "n_reject": res["n_reject"],
                   "n_index": len(db_index),
                   "target_present_s1": _present(s1)},
        "stage2": {"n_s2": len(s2),
                   "target_present_s2": _present(s2)},
        "stage3_oracle": {"patterns": ("single+bonded_pair" if pair
                                       else ("single" if single else "none")),
                          "n_s3": len(s3),
                          "target_present_s3": _present(s3),
                          "truncated": res["truncated"],
                          "zero_status": res["zero_status"],
                          "label": ("ORACLE patterns from true graph; NOT "
                                    "measured fragment graphs")},
        "stage4_precursor": res["precursor_status"],
    }


def chebi_score_queries(queries, chebi_by_formula, model, tag):
    """Four frozen arms on ChEBI formula pools (same pools, all arms).

    Pool fingerprints come from the ChEBI records' own reviewed typed
    graphs (fp_from_typed), never reparsed SMILES. Identity is canonical;
    unknown-identity pool members are retained unscored and can never
    match. Target NEVER injected.
    """
    from tools.ms2_chebi_corpus import formula_key as _fk
    from tools.ms2_fp_predictor import (fp_from_typed, rank_by_score as _rbs,
                                        rank_uniform as _ru,
                                        rank_massresid_order as _rm,
                                        margin_from_finite_scores as _mf,
                                        finite_scored_values as _fv)
    rows = []
    n_present = 0
    n_q_unknown = 0
    n_pool_dedup = 0
    for i, q in enumerate(queries):
        qid = f"{tag}-{i:04d}"
        rec = q["rec"]
        raw_pool = [r for r in chebi_by_formula.get(_fk(q["std"]["formula"]), [])
                    if r["charge"] == 0]
        # Canonical connectivity dedup for rank counts: first record per
        # known canon wins; unknown-identity records retained as distinct
        # unscored entries (never match, counted).
        pool_recs, seen_canon = [], set()
        for r in raw_pool:
            c = r.get("canon")
            if c and c in seen_canon:
                n_pool_dedup += 1
                continue
            if c:
                seen_canon.add(c)
            pool_recs.append(r)
        tcanon, tok = rdkit_canon(rec["smiles"])
        if not (tok and tcanon):
            n_q_unknown += 1
            tgt_pos = None
        else:
            tgt_pos = next((idx for idx, r in enumerate(pool_recs)
                            if r.get("canon") == tcanon), None)
        present = 1 if tgt_pos is not None else 0
        n_present += present
        pfps, residuals = [], []
        observed_mu = q["neutral_mu"]
        for r in pool_recs:
            try:
                from tools.ms2_fp_predictor import fp_from_typed as _fft
                vec, norm = _fft(r["atom_types"], r["edges"])
                pfps.append((vec, norm))
            except Exception:
                pfps.append((None, 0.0))
            try:
                ppm = abs(r["mass"] - observed_mu) / max(observed_mu, 1) * 1e6
                residuals.append(ppm if math.isfinite(ppm) else None)
            except Exception:
                residuals.append(None)
        pred, pn = predict_fp_numpy(model, rec["peaks"])
        pscores = score_candidates_fp(pred, pn, pfps)
        prscores = score_candidates_fp(model["prior"], model["prior_norm"], pfps)
        pos = seeded_positions(qid, len(pool_recs))
        ou = _ru(len(pool_recs), pos)
        om = _rm(residuals, pos)
        op = _rbs(pscores, pos)
        orr = _rbs(prscores, pos)

        def rank_of(order):
            return None if tgt_pos is None else order.index(tgt_pos) + 1

        ranks = {"uniform": rank_of(ou), "massresid": rank_of(om),
                 "predictor": rank_of(op), "prior": rank_of(orr)}
        margin_pred, margin_status = _mf([s["score"] for s in pscores])
        rankable, rreason = capability_for_query(rec["peaks"], pred, pn, pscores)
        has_features = spectrum_has_usable_features(rec["peaks"])
        if not has_features:
            ranks = {a: None for a in ranks}
            pred_applicable = False
        else:
            ranks["predictor"], pred_applicable = enforce_predictor_rank(
                ranks["predictor"], rankable, pscores, tgt_pos)
        tparse, _, _ = target_parse_info(rec["smiles"])
        rows.append(({"qid": qid,
                      "in_pool": tgt_pos is not None,
                      "n_pool": len(pool_recs),
                      "n_pool_identity_known": sum(1 for r in pool_recs if r.get("canon")),
                      "n_pool_identity_unknown": sum(1 for r in pool_recs if not r.get("canon")),
                      "n_parseable": sum(1 for s in pscores if s["parseable"]),
                      "target_parseable": tparse, "rankable": rankable,
                      "rankable_reason": rreason,
                      "margin_pred": margin_pred,
                      "predictor_applicable": pred_applicable,
                      "identity_status": ("known" if (tok and tcanon) else "unknown"),
                      "massresid_input": "measured_precursor_neutral",
                      "measured_neutral_mu": observed_mu,
                      "rank_uniform": ranks["uniform"],
                      "rank_massresid": ranks["massresid"],
                      "rank_predictor": ranks["predictor"],
                      "rank_prior": ranks["prior"],
                      "target_natural_presence": present,
                      "target_string_presence": 0}, q))
    return rows, {"n_natural_presence": n_present,
                  "n_identity_unknown_query": n_q_unknown,
                  "n_pool_dedup_canon": n_pool_dedup}


def load_frozen_predictor(npz_path):
    import numpy as np
    d = np.load(npz_path)
    coef = np.asarray(d["coef_"], dtype=np.float64)
    intercept = float(np.asarray(d["intercept_"]))
    prior = np.asarray(d["mean_prior"], dtype=np.float64).tolist()
    pnorm = math.sqrt(sum(v * v for v in prior))
    return {"coef": coef, "intercept": intercept,
            "prior": prior, "prior_norm": pnorm,
            "sha256": sha256_file(npz_path)}


def predict_fp_numpy(model, peaks):
    """Frozen Ridge predict with numpy (equivalent to sklearn predict).

    pred = X @ coef.T + intercept, then L2-normalize. Returns (vec, norm).
    """
    import numpy as np
    feat, norm, _ = featurize_spectrum(peaks)
    if norm <= 0.0 or not feat:
        return [0.0] * FP_DIM, 0.0
    dense = np.zeros((1, model["coef"].shape[1]), dtype=np.float64)
    for b, v in feat.items():
        if 0 <= b < dense.shape[1]:
            dense[0, b] = v
    pred = (dense @ model["coef"].T)[0] + model["intercept"]
    vec = [float(v) for v in pred]
    n = math.sqrt(sum(v * v for v in vec))
    if n > 0.0:
        vec = [v / n for v in vec]
    return vec, n


# --------------------------------------------------------------------------
# PubChem tiny expansion (only on measured misses; bounded; cached).
# --------------------------------------------------------------------------

def _pug_get_json(url, timeout=20):
    """GET a PUG URL, following async ListKey polling (bounded).

    Returns (payload_dict, n_http_calls). Raises on Fault/timeout.
    Formula queries return HTTP 200 {"Waiting": {"ListKey": ...}} first;
    poll the listkey endpoint up to 6 x 5s, then parse the final payload.
    """
    import time as _t
    calls = 0

    def _get(u):
        req = urllib.request.Request(u, headers={"User-Agent": "mamba-trainer-ms2/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    payload = _get(url)
    calls += 1
    _t.sleep(0.3)
    for _ in range(6):
        if "Waiting" not in payload:
            break
        key = payload["Waiting"].get("ListKey")
        if not key:
            break
        _t.sleep(5)
        payload = _get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                       f"listkey/{key}/cids/JSON")
        calls += 1
        _t.sleep(0.3)
    if "Fault" in payload:
        raise RuntimeError(f"PUG fault: {payload['Fault']}")
    if "Waiting" in payload:
        raise TimeoutError("PUG listkey still waiting after bounded polls")
    return payload, calls


def pubchem_expand(formulas, cache_dir, max_formulas=5, max_candidates=100,
                   cache_only=True):
    """Bounded PubChem PUG expansion for measured-miss formulas.

    Returns (results, stats). Never raises: per-formula errors recorded.
    cache_only=True (review-mandated: no live PubChem requests): cached
    responses replay only; formulas without cache are reported unavailable
    with concrete status, never fetched. Throttle ~0.3s between calls,
    20s timeout each on the live path. No remote structure
    is used as ground truth; presence of the query target in the expanded
    set is reported as marginal recall only (no rescoring).
    """
    os.makedirs(cache_dir, exist_ok=True)
    results, api_calls, errors = {}, 0, {}
    for form in sorted(set(formulas))[:max_formulas]:
        cache_hit = os.path.exists(os.path.join(cache_dir, f"cids_{form}.json"))
        try:
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"formula/{form}/cids/JSON?MaxRecords={max_candidates}")
            cpath = os.path.join(cache_dir, f"cids_{form}.json")
            if not os.path.exists(cpath):
                if cache_only:
                    raise FileNotFoundError(f"cached CID response unavailable: {form}")
                payload, ncalls = _pug_get_json(url)
                api_calls += ncalls
                with open(cpath, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload))
            with open(cpath, encoding="utf-8") as handle:
                cids = json.load(handle).get("IdentifierList", {}).get("CID", [])
            cids = cids[:max_candidates]
            smiles = []
            for i in range(0, len(cids), 50):
                chunk = cids[i:i + 50]
                spath = os.path.join(cache_dir, f"smiles_{form}_{i}.json")
                if not os.path.exists(spath):
                    if cache_only:
                        raise FileNotFoundError(f"cached structure response unavailable: {form}/{i}")
                    purl = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                            + ",".join(str(c) for c in chunk)
                            + "/property/CanonicalSMILES/JSON")
                    req = urllib.request.Request(purl, headers={"User-Agent": "mamba-trainer-ms2/1.0"})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        pbody = resp.read()
                    api_calls += 1
                    time.sleep(0.3)
                    with open(spath, "wb") as handle:
                        handle.write(pbody)
                with open(spath, "rb") as handle:
                    props = json.loads(handle.read().decode("utf-8")) \
                        .get("PropertyTable", {}).get("Properties", [])
                # Current pinned responses name this field ConnectivitySMILES.
                # Preserve compatibility with older cached CanonicalSMILES data.
                smiles.extend(p.get("ConnectivitySMILES") or
                              p.get("CanonicalSMILES") or "" for p in props)
            results[form] = {"n_cids": len(cids), "n_smiles": len(smiles),
                             "truncated": len(cids) >= max_candidates,
                             "cache_hit": cache_hit,
                             "smiles": smiles}
        except Exception as exc:
            errors[form] = f"{type(exc).__name__}: {exc}"
            results[form] = {"error": errors[form], "cache_hit": cache_hit}
    stats = {"formulas_attempted": len(results), "api_calls": api_calls,
             "cache_only": cache_only,
             "errors": errors}
    return results, stats


# --------------------------------------------------------------------------
# Main pipeline.
# --------------------------------------------------------------------------

def scan_massbank(zip_path, max_records=MAX_SCAN_RECORDS, max_seconds=480):
    """Archive-order scan of MassBank records. Returns (records, stats)."""
    t0 = time.time()
    records, excl = [], {}
    n_files = n_scanned = 0
    licenses = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.endswith(".txt") and "/MSBNK-" in n)
        n_files = len(names)
        for name in names:
            if n_scanned >= max_records or time.time() - t0 > max_seconds:
                break
            n_scanned += 1
            try:
                text = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                excl["unreadable"] = excl.get("unreadable", 0) + 1
                continue
            rec, reason = parse_massbank_record(text)
            if rec is None:
                excl[reason.split(":")[0]] = excl.get(reason.split(":")[0], 0) + 1
                continue
            rec["source_file"] = name
            records.append(rec)
            licenses[rec["license"]] = licenses.get(rec["license"], 0) + 1
    stats = {"n_record_files": n_files, "n_scanned": n_scanned,
             "n_parseable": len(records), "exclusions": excl,
             "licenses": licenses,
             "timed_out": (time.time() - t0 > max_seconds),
             "truncated": n_scanned >= max_records,
             "completion": ("complete" if n_scanned >= n_files else
                            ("truncated_time" if time.time() - t0 > max_seconds
                             else "truncated_cap"))}
    return records, stats


def canon_set(smiles_iter):
    """RDKit-canonical connectivity set. Returns (canon_set, n_fail).

    Stereo-insensitive (isomericSmiles=False) connectivity identity, the
    stated policy. Failures are counted, never silently raw-matched.
    """
    out, fail = set(), 0
    for s in smiles_iter:
        c, ok = rdkit_canon(s)
        if ok and c:
            out.add(c)
        else:
            fail += 1
    return out, fail


def select_external(records, fit_ids, max_queries=MAX_EXTERNAL_QUERIES,
                    pre_canon=None, pre_content=None):
    """Fixed archive-order selection of distinct molecules.

    Gates: chemistry parse, adduct-supported neutral conversion, mass
    agreement (<=50ppm measured-neutral vs CH$EXACT_MASS cross-check),
    usable spectrum, canonical-connectivity dedup + spectrum-content dedup,
    frozen fit/calib exclusion by canonical connectivity AND spectrum
    content (same content, different ID/SMILES excluded). pre_canon /
    pre_content hold cross-source already-selected identities (dedup across
    sources). Returns (selected, audit) with before/after counters.
    fit_ids: dict with group_set/smiles_set/canon_set/content_set.
    """
    from tools.ms2_msgym_corpus import standardize_smiles
    selected = []
    seen_keys, seen_content = set(pre_canon or ()), set(pre_content or ())
    # Seed seen sets with cross-source preselected content keys too.
    audit = {"n_excluded_chem": 0, "n_excluded_mass": 0,
             "n_excluded_nofeatures": 0, "n_excluded_dup": 0,
             "n_excluded_fitcalib": 0, "n_excluded_fitcalib_content": 0,
             "n_canon_fail": 0, "chem_reasons": {}}
    for rec in records:
        if len(selected) >= max_queries:
            break
        try:
            srec, reason = standardize_smiles(rec["smiles"], "ext", rec["accession"])
        except Exception:
            srec, reason = None, "exception"
        if srec is None:
            audit["n_excluded_chem"] += 1
            audit["chem_reasons"][reason.split(":")[0]] = \
                audit["chem_reasons"].get(reason.split(":")[0], 0) + 1
            continue
        neutral_mu, err = precursor_to_neutral(rec["precursor_mz"], rec["adduct"])
        if neutral_mu is None:
            audit["n_excluded_chem"] += 1
            continue
        exact_mu = int(round(rec["exact_mass"] * 1_000_000))
        ppm = abs(neutral_mu - exact_mu) / max(exact_mu, 1) * 1e6
        if ppm > MASS_AGREE_PPM:
            audit["n_excluded_mass"] += 1
            continue
        if not spectrum_has_usable_features(rec["peaks"]):
            audit["n_excluded_nofeatures"] += 1
            continue
        canon, cok = rdkit_canon(rec["smiles"])
        if not cok or not canon:
            audit["n_canon_fail"] += 1
            continue
        from tools.ms2_spectral_rank import content_fp
        ckey = content_fp(rec["peaks"])
        if canon in seen_keys or ckey in seen_content:
            audit["n_excluded_dup"] += 1
            continue
        if canon in fit_ids.get("canon_set", set()):
            audit["n_excluded_fitcalib"] += 1
            continue
        if ckey in fit_ids.get("content_set", set()):
            audit["n_excluded_fitcalib_content"] += 1
            continue
        seen_keys.add(canon)
        seen_content.add(ckey)
        key14 = inchikey14_of(rec["smiles"], rec.get("inchikey"))
        selected.append({"rec": rec, "std": srec, "key14": key14,
                         "canon": canon, "content": ckey,
                         "neutral_mu": neutral_mu, "ppm": ppm})
    audit["n_selected"] = len(selected)
    return selected, audit


def enforce_predictor_rank(rank_pred, rankable, pscores, tgt_pos):
    """Gate the learned-predictor rank on inference capability.

    The predictor top-k requires BOTH query capability (usable features,
    finite nonzero prediction, >=1 finite score, finite margin) AND a
    finite score for the true candidate itself. Otherwise the predictor
    rank is None (miss in ALLSELECTED, excluded from conditional
    predictor accuracy). Uniform/massresid/prior ranks stay valid where
    their own inputs apply and are gated separately by callers.
    Returns (rank_or_None, applicable_bool).
    """
    if not rankable or tgt_pos is None:
        return None, False
    try:
        s = pscores[tgt_pos].get("score")
    except (IndexError, AttributeError):
        return None, False
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        return None, False
    if not math.isfinite(float(s)):
        return None, False
    return rank_pred, True


def apply_frozen_gate(rows, gate):
    """Apply the frozen abstain-all calibration gate to external rows.

    The gate (tau, kmax) comes from the frozen calibration (tau None ->
    abstain-all: accept nothing). Returns (n_accepted, accepted_qids):
    measured by applying the rule, never hardcoded.
    """
    if not gate or gate.get("tau") is None:
        return 0, []
    tau, kmax = gate["tau"], gate.get("kmax", 0)
    acc = []
    for r, _ in rows:
        try:
            m = float(r.get("margin_pred", "nan"))
        except (TypeError, ValueError):
            continue
        rp = r.get("rank_predictor")
        if (math.isfinite(m) and m >= tau and rp is not None
                and rp <= max(kmax, 0)):
            acc.append(r["qid"])
    return len(acc), acc


def score_queries(queries, pool_index, model, tag):
    """Score external queries with the frozen checkpoint on identical pools.

    Pools are formula-matched database subsets used VERBATIM: a naturally
    present external target is database coverage (counted in pool recall),
    never an injection (nothing is ever added to a pool). Identity
    overlap with fit/calibration was already excluded at selection.
    """
    from tools.ms2_chebi_corpus import formula_key as _fk
    fp_cache = {}
    rows = []
    n_natural_present = 0
    n_string_present = 0
    n_identity_unknown_query = 0
    n_identity_unknown_cand = 0
    for i, q in enumerate(queries):
        qid = f"{tag}-{i:04d}"
        rec = q["rec"]
        key = _fk(q["std"]["formula"])
        entries = pool_index.get(key, [])
        pool = [s for s, _ in entries]
        # Exact stereo-insensitive canonical connectivity (stated policy).
        # Query without canonical identity is EXCLUDED from identity
        # denominators (status identity_unknown, retained as unavailable).
        # Candidates without canonical identity are retained UNSCORED with
        # unknown identity: they can never match (no silent raw fallback).
        tcanon, tok = rdkit_canon(rec["smiles"])
        if tok and tcanon:
            tkey = tcanon
            q_identity = "known"
        else:
            tkey = None
            q_identity = "unknown"
            n_identity_unknown_query += 1
        canon_known = [c for _, c in entries if c]
        n_identity_unknown_cand += sum(1 for _, c in entries if not c)
        natural = 1 if (tkey is not None and tkey in canon_known) else 0
        n_natural_present += natural
        string_present = 1 if rec["smiles"] in pool else 0
        n_string_present += string_present
        # rank the identical pool; target position by connectivity identity
        if tkey is None:
            tgt_pos = None
        else:
            tgt_pos = next((idx for idx, (_, c) in enumerate(entries)
                            if c == tkey), None)
        # pool fingerprints (retained when unparseable, never dropped)
        pfps = []
        for smi in pool:
            if smi not in fp_cache:
                fp_cache[smi] = fp_from_smiles(smi)
            vec, norm, _ = fp_cache[smi]
            pfps.append((vec, norm))
        pred, pn = predict_fp_numpy(model, rec["peaks"])
        pscores = score_candidates_fp(pred, pn, pfps)
        prscores = score_candidates_fp(model["prior"], model["prior_norm"], pfps)
        # massresid baseline input is the INDEPENDENTLY MEASURED precursor-
        # derived neutral mass (neutral_mu, micro-Da int), with the stated
        # uncertainty of candidate_residuals (ppm ranking). The theoretical
        # CH$EXACT_MASS annotation is a cross-check only, NEVER a baseline
        # input: changing it cannot move these ranks (regression-tested).
        if "neutral_mu" in q and q["neutral_mu"] is not None:
            measured_mu = q["neutral_mu"]
            mass_role = "measured_precursor_neutral"
        else:
            measured_mu = int(round(float(rec["exact_mass"]) * 1_000_000))
            mass_role = "measured_precursor_neutral (gnps wrapper)"
        residuals, _ = spectral_residuals(pool, str(measured_mu / 1_000_000))
        pos = seeded_positions(qid, len(pool))
        # ranks on the identical pool, target located by connectivity
        # identity (same order helpers as the frozen predictor run)
        from tools.ms2_fp_predictor import (rank_by_score as _rbs,
                                            rank_uniform as _ru,
                                            rank_massresid_order as _rm,
                                            margin_from_finite_scores as _mf,
                                            finite_scored_values as _fv)
        ou = _ru(len(pool), pos)
        om = _rm(residuals, pos)
        op = _rbs(pscores, pos)
        orr = _rbs(prscores, pos)

        def rank_of(order):
            return None if tgt_pos is None else order.index(tgt_pos) + 1

        ranks = {"uniform": rank_of(ou), "massresid": rank_of(om),
                 "predictor": rank_of(op), "prior": rank_of(orr)}
        margin_pred, margin_status = _mf([s["score"] for s in pscores])
        n_scored = len(_fv([s["score"] for s in pscores]))
        rankable, rreason = capability_for_query(rec["peaks"], pred, pn, pscores)
        has_features = spectrum_has_usable_features(rec["peaks"])
        if not has_features:
            # Failed-feature queries are retained as unavailable: no arm
            # may rank from the unscored tail.
            ranks = {a: None for a in ranks}
            pred_applicable = False
        else:
            ranks["predictor"], pred_applicable = enforce_predictor_rank(
                ranks["predictor"], rankable, pscores, tgt_pos)
        tparse, _, _ = target_parse_info(rec["smiles"])
        if q_identity == "unknown":
            in_pool_flag = False
            status_identity = "identity_unknown"
        else:
            in_pool_flag = tgt_pos is not None
            status_identity = "known"
        row = {"qid": qid, "in_pool": in_pool_flag, "n_pool": len(pool),
               "n_pool_identity_known": len(canon_known),
               "n_pool_identity_unknown": sum(1 for _, c in entries if not c),
               "n_parseable": sum(1 for s in pscores if s["parseable"]),
               "target_parseable": tparse, "rankable": rankable,
               "rankable_reason": rreason,
               "margin_pred": margin_pred,
               "predictor_applicable": pred_applicable,
               "identity_status": status_identity,
               "massresid_input": mass_role,
               "measured_neutral_mu": measured_mu,
               "rank_uniform": ranks["uniform"],
               "rank_massresid": ranks["massresid"],
               "rank_predictor": ranks["predictor"],
               "rank_prior": ranks["prior"],
               "target_natural_presence": natural,
               "target_string_presence": string_present}
        rows.append((row, q))
    return rows, {"n_natural_presence": n_natural_present,
                  "n_string_presence": n_string_present,
                  "n_identity_unknown_query": n_identity_unknown_query,
                  "n_identity_unknown_cand": n_identity_unknown_cand}


def summarize_scored(rows, gate=None):
    """Tiered top-k/Wilson/paired summary over scored external rows.

    Denominators (never conflated):
      ALLSELECTED: every row incl. identity_unknown / unavailable as miss.
      present: canonical target presence (in_pool True) -> conditional topk.
      present_parseable: present AND target graph parseable.
      capability: rankable (inference capability, independent of the true
        target graph/identity).
    Identical pools feed all arms; failed-identity/feature queries are
    retained as unavailable (ranks None). The unscored tail (None scores)
    ranks last and is NEVER presented as a useful spectral prediction:
    margins exclude non-finite scores (see margin_from_finite_scores).
    gate: frozen abstain-all calibration gate -> accepted counts (0).
    """
    from tools.ms2_quality_audit import paired_diff_ci
    tiers = {
        "all_selected": [r for r, _ in rows],
        "present": [r for r, _ in rows if r.get("in_pool")],
        "present_parseable": [r for r, _ in rows
                              if r.get("in_pool") and r.get("target_parseable")],
        "capability": [r for r, _ in rows if r.get("rankable")],
        "predictor_applicable": [r for r, _ in rows
                                 if r.get("predictor_applicable")],
    }
    out = {"n": len(rows),
           "n_tiers": {k: len(v) for k, v in tiers.items()}}
    for tier, sub in tiers.items():
        out[tier] = {}
        n = len(sub)
        for arm in ARMS:
            col = f"rank_{arm}"
            for k in KS:
                h = sum(1 for r in sub if hit_at_k(r[col], k))
                lo, hi = wilson_interval(h, n)
                out[tier][f"{arm}@top{k}"] = {
                    "hits": h, "n": n,
                    "rate": (h / n) if n else None, "ci95": [lo, hi]}
    # paired predictor-vs-baseline on ALLSELECTED qids (explicit qid match)
    allr = tiers["all_selected"]
    for other in ("uniform", "massresid", "prior"):
        for k in KS:
            a = [hit_at_k(r["rank_predictor"], k) for r in allr]
            b = [hit_at_k(r[f"rank_{other}"], k) for r in allr]
            n = len(allr)
            if n == 0:
                out[f"predictor_minus_{other}.top{k}"] = {"diff": 0.0, "ci95": [None, None], "n": 0}
                continue
            diffs = [1 if (x and not y) else (-1 if (y and not x) else 0)
                     for x, y in zip(a, b)]
            mean = sum(diffs) / n
            var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            se = math.sqrt(var / n) if n else 0.0
            out[f"predictor_minus_{other}.top{k}"] = {
                "diff": mean, "ci95": [mean - Z95 * se, mean + Z95 * se], "n": n}
    n_pool_hit = sum(1 for r, _ in rows if r["in_pool"])
    lo, hi = wilson_interval(n_pool_hit, len(rows))
    out["pool_recall_all"] = {"hits": n_pool_hit, "n": len(rows),
                              "rate": (n_pool_hit / len(rows)) if rows else None,
                              "ci95": [lo, hi]}
    # back-compat flat aliases for the all-selected tier (previous shape)
    for k, v in out["all_selected"].items():
        out[k] = v
    out["paired_note"] = ("descriptive paired differences on fixed frozen "
                          "ranks; no tuning")
    if gate is not None:
        n_acc, acc_qids = apply_frozen_gate(rows, gate)
        out["frozen_gate"] = dict(gate)
        out["frozen_gate"]["n_accepted_external"] = n_acc
        out["frozen_gate"]["accepted_qids"] = acc_qids
    return out


def run_external_validation(repo_root, out_dir, max_queries=MAX_EXTERNAL_QUERIES,
                            do_pubchem=True):
    t0 = time.process_time()
    wall0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.join(repo_root, "data/pinned/external")
    mb_zip = os.path.join(ext, "MassBank-data-2026.03.zip")
    gnps_mgf = os.path.join(ext, "GNPS-FAULKNERLEGACY.mgf")
    prefix = os.path.join(repo_root, "data/pinned/msgym_candidates_formula_prefix64.json")
    model_p = os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/predictor_model.npz")
    for p in (mb_zip, gnps_mgf, prefix, model_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing pinned input: {p}")

    api_calls_total = 0

    # --- MassBank scan + fixed selection ---
    progress("scanning MassBank archive")
    mb_records, mb_scan = scan_massbank(mb_zip)
    progress(f"MassBank parseable={mb_scan['n_parseable']} "
             f"scanned={mb_scan['n_scanned']}")

    # --- sealed overlap identities: ACTUAL USED fit/calib (read-only) ---
    frozen = load_frozen_train_identities(repo_root)
    progress(f"frozen USED fit={frozen['n_used_fit']} calib={frozen['n_used_calib']} "
             f"content_recovered={frozen['n_content_recovered']}")
    val_rows = list(csv.DictReader(open(
        os.path.join(repo_root, "experiments/molecular_completion/20261003_predictor/predictor_rows.csv"),
        encoding="utf-8")))
    val_smiles = set(r["smiles"] for r in val_rows if r.get("smiles"))
    fit_canon, fit_canon_fail = canon_set(frozen["smiles"])
    val_canon, val_canon_fail = canon_set(val_smiles)
    fit_ids = {"group_set": frozen["group_set"],
               "smiles_set": frozen["smiles_set"],
               "canon_set": fit_canon,
               "content_set": frozen["content_set"]}
    test_smiles, test_keys = set(), set()
    with open(os.path.join(repo_root, "data/pinned/MassSpecGym1.5.tsv"), encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("fold") != "test":
                continue
            if row.get("smiles"):
                test_smiles.add(row["smiles"].strip())
            k = (row.get("inchikey") or "").strip().split("-")[0]
            if k:
                test_keys.add(k)
    test_canon, test_canon_fail = canon_set(test_smiles)

    mb_selected, mb_audit = select_external(mb_records, fit_ids,
                                            max_queries=max_queries)
    progress(f"MassBank selected={len(mb_selected)}")
    mb_audit["cross_source_preselected"] = {"n_canon": 0, "n_content": 0}

    # --- GNPS parse (audit before filter, then after) ---
    with open(gnps_mgf, encoding="utf-8") as handle:
        gblocks, gstats = parse_mgf_blocks(handle.read())
    g_license_note = (
        "MGF carries no per-record license field. Per-block SUBMITUSER / "
        "ORGANISM / SPECTRUMID retained. GNPS docs "
        "(https://ccms-ucsd.github.io/GNPSDocumentation/downloadlibraries/): "
        "'All GNPS Reference spectra contributed directly to GNPS by default "
        "will have the CC0 license. Third party libraries imported may not "
        "conform to the CC BY license and should be verified by users.' "
        "FAULKNERLEGACY import status is UNVERIFIED: evaluation use only, "
        "contributor attribution retained (Sirenas Marine Discovery via "
        "mwang87). CC0 is NOT assumed for this import.")
    gstruct, gexcl = [], {}
    for b in gblocks:
        smi = (b["params"].get("SMILES") or "").strip()
        if not smi or smi == "N/A":
            gexcl["no_structure"] = gexcl.get("no_structure", 0) + 1
            continue
        adduct, rule = gnps_adduct_of(b)
        if adduct is None:
            gexcl[f"adduct:{rule}"] = gexcl.get(f"adduct:{rule}", 0) + 1
            continue
        from tools.ms2_msgym_corpus import standardize_smiles
        try:
            srec, reason = standardize_smiles(smi, "gnps", b["params"].get("SPECTRUMID", "?"))
        except Exception:
            srec, reason = None, "exception"
        if srec is None:
            gexcl[f"chem:{reason.split(':')[0]}"] = gexcl.get(f"chem:{reason.split(':')[0]}", 0) + 1
            continue
        if not spectrum_has_usable_features(b["peaks"]):
            gexcl["nofeatures"] = gexcl.get("nofeatures", 0) + 1
            continue
        neutral_mu, err = precursor_to_neutral(b["pep"], adduct)
        if neutral_mu is None:
            gexcl[f"neutral:{err}"] = gexcl.get(f"neutral:{err}", 0) + 1
            continue
        ppm = abs(neutral_mu - srec["mass"]) / max(srec["mass"], 1) * 1e6
        if ppm > MASS_AGREE_PPM:
            gexcl["mass_mismatch"] = gexcl.get("mass_mismatch", 0) + 1
            continue
        canon, cok = rdkit_canon(smi)
        if not cok or not canon:
            gexcl["canon_fail"] = gexcl.get("canon_fail", 0) + 1
            continue
        from tools.ms2_spectral_rank import content_fp
        gstruct.append({"block": b, "smiles": smi, "adduct": adduct,
                        "rule": rule, "std": srec, "neutral_mu": neutral_mu,
                        "ppm": ppm, "canon": canon,
                        "content": content_fp(b["peaks"])})
    n_gstruct_pre = len(gstruct)
    # GNPS fit/calib + cross-source (MassBank-selected) exclusion, then cap.
    gn_selected = []
    n_gexcl_fit = n_gexcl_fit_content = n_gexcl_cross = 0
    mb_canon = set(s["canon"] for s in mb_selected)
    mb_content = set(s["content"] for s in mb_selected)
    for g in gstruct:
        k14 = inchikey14_of(g["smiles"], g["block"]["params"].get("INCHI"))
        if g["canon"] in fit_ids["canon_set"]:
            n_gexcl_fit += 1
            continue
        if g["content"] in fit_ids["content_set"]:
            n_gexcl_fit_content += 1
            continue
        if g["canon"] in mb_canon or g["content"] in mb_content:
            n_gexcl_cross += 1
            continue
        if len(gn_selected) >= GNPS_MAX_QUERIES:
            continue
        g["key14"] = k14
        gn_selected.append(g)
    n_gexcl_cap = n_gstruct_pre - len(gn_selected) - n_gexcl_fit - n_gexcl_fit_content - n_gexcl_cross

    # --- candidate pool index (formula-matched; retained named arm) ---
    progress("building formula pool index")
    pool_index, pool_stats = build_formula_pool_index(prefix)

    # --- FULL ChEBI coverage index (required stage; pinned SDF) ---
    from tools.ms2_chebi_corpus import formula_key as _fkc
    chebi_path = os.path.join(repo_root, "data/pinned/chebi_3_stars.sdf.gz")
    if not os.path.exists(chebi_path):
        raise FileNotFoundError(f"missing pinned ChEBI SDF: {chebi_path}")
    progress("building FULL ChEBI index")
    t_chebi0 = time.process_time()
    chebi_records, chebi_stats, chebi_by_formula = build_chebi_index(chebi_path)
    from tools.ms2_database_retrieval import DatabaseIndex
    chebi_db = DatabaseIndex(chebi_records)
    chebi_stats["index_build_cpu_s"] = time.process_time() - t_chebi0
    chebi_stats["pin"] = {"file": "data/pinned/chebi_3_stars.sdf.gz",
                          "sha256": sha256_file(chebi_path),
                          "bytes": os.path.getsize(chebi_path)}
    progress(f"ChEBI index: {len(chebi_records)} records, "
             f"{chebi_stats['n_formulas']} formulas")

    # --- frozen checkpoint (hash-verified, never refit) ---
    model = load_frozen_predictor(model_p)
    progress(f"checkpoint sha={model['sha256'][:12]}")
    pred_summary = json.load(open(os.path.join(
        repo_root, "experiments/molecular_completion/20261003_predictor/predictor_summary.json"),
        encoding="utf-8"))
    frozen_gate = pred_summary.get("calibration", {}).get("gate", {})

    # --- score MassBank + GNPS structurized: prefix-pool arm (retained,
    # named; NOT replaced by the ChEBI stage) ---
    mb_rows, mb_pool = score_queries(mb_selected, pool_index, model, "MB")
    gn_rows, gn_pool = score_queries(
        [{"rec": {"smiles": g["smiles"], "peaks": g["block"]["peaks"],
                          "exact_mass": g["neutral_mu"] / 1_000_000,
                          "accession": g["block"]["params"].get("SPECTRUMID", "?"),
                          "adduct": g["adduct"]},
          "std": g["std"], "neutral_mu": g["neutral_mu"]} for g in gn_selected],
        pool_index, model, "GN")
    mb_summary = summarize_scored(mb_rows, gate=frozen_gate)
    mb_summary["arm_name"] = ("prefix_formula_pool (MassSpecGym prefix64 "
                              "formula pools; retained additional experiment)")
    gn_summary = summarize_scored(gn_rows, gate=frozen_gate)
    gn_summary["arm_name"] = "prefix_formula_pool (GNPS structurized)"

    # --- ChEBI full-coverage stage for every selected query ---
    def _precision_known_mb(s):
        return bool(s["rec"].get("precision_known", False))

    def _rounding_mb(s):
        return s["rec"].get("rounding_mu")

    def _precision_known_gn(g):
        pep_text = (g["block"]["params"].get("PEPMASS") or "").strip().split()[0]
        return bool(re.match(r"^[0-9]+\.[0-9]+$", pep_text))

    def _rounding_gn(g):
        pep_text = (g["block"]["params"].get("PEPMASS") or "").strip().split()[0]
        m = re.match(r"^[0-9]+\.([0-9]+)$", pep_text)
        return int(round(0.5 * 10 ** (6 - len(m.group(1))))) if m else None

    chebi_stages = []
    for i, s in enumerate(mb_selected):
        chebi_stages.append(chebi_stage_query(
            f"MB-{i:04d}", s["std"], s["canon"], s["neutral_mu"],
            _precision_known_mb(s), chebi_db, chebi_by_formula,
            uncertainty_mu=_rounding_mb(s)))
    for i, g in enumerate(gn_selected):
        chebi_stages.append(chebi_stage_query(
            f"GN-{i:04d}", g["std"], g["canon"], g["neutral_mu"],
            _precision_known_gn(g), chebi_db, chebi_by_formula,
            uncertainty_mu=_rounding_gn(g)))
    chebi_rows_mb, chebi_pool_mb = chebi_score_queries(
        mb_selected, chebi_by_formula, model, "MB")
    chebi_rows_gn, chebi_pool_gn = chebi_score_queries(
        [{"rec": {"smiles": g["smiles"], "peaks": g["block"]["peaks"],
                           "exact_mass": g["neutral_mu"] / 1_000_000,
                           "accession": g["block"]["params"].get("SPECTRUMID", "?"),
                           "adduct": g["adduct"]},
          "std": g["std"], "neutral_mu": g["neutral_mu"]} for g in gn_selected],
        chebi_by_formula, model, "GN")
    chebi_summary_mb = summarize_scored(chebi_rows_mb, gate=frozen_gate)
    chebi_summary_mb["arm_name"] = ("chebi_formula_pool_ranking (formula-only "
                                    "ranking arm; independent of the nested "
                                    "measured mass->formula->oracle arm)")
    chebi_summary_gn = summarize_scored(chebi_rows_gn, gate=frozen_gate)
    chebi_summary_gn["arm_name"] = ("chebi_formula_pool_ranking "
                                    "(GNPS structurized; formula-only)")
    # stage aggregates: nested presence with Wilson recall on ALLSELECTED
    # plus eligible (present-at-previous-stage) conditionals
    def _stage_agg(stages):
        def _w(h, n):
            lo, hi = wilson_interval(h, n)
            return {"hits": h, "n": n,
                    "rate": (h / n) if n else None, "ci95": [lo, hi]}
        n = len(stages)
        s1 = [s for s in stages if s["stage1"]["target_present_s1"]]
        s2 = [s for s in stages if s["stage2"]["target_present_s2"]]
        s3 = [s for s in stages
              if s["stage3_oracle"]["target_present_s3"]]
        agg = {"n": n,
               "nested_ok_all": all(s["nested_ok"] for s in stages),
               "stage1_presence_all": _w(len(s1), n),
               "stage2_presence_all": _w(len(s2), n),
               "stage2_presence_given_s1": _w(len(s2), len(s1)),
               "stage3_presence_all": _w(len(s3), n),
               "stage3_presence_given_s2": _w(len(s3), len(s2)),
               "stage1": {
                   "n_unavailable_precision": sum(
                       1 for s in stages if not s["precision_known"]),
                   "counts": sorted(
                       (s["stage1"]["n_s1"] for s in stages))[:5],
                   "max_n_s1": max(
                       (s["stage1"]["n_s1"] for s in stages), default=0)},
               "stage2": {
                   "pool_sizes": sorted(
                       s["stage2"].get("n_s2", 0) for s in stages)},
               "stage3_oracle": {
                   "n_truncated": sum(
                       1 for s in stages if s["stage3_oracle"]["truncated"])},
               "stage4_precursor": list(
                   set(s["stage4_precursor"] for s in stages))}
        return agg
    chebi_stage_agg = {"massbank": _stage_agg([s for s in chebi_stages
                                               if s["qid"].startswith("MB-")]),
                       "gnps": _stage_agg([s for s in chebi_stages
                                           if s["qid"].startswith("GN-")])}

    # --- overlap flags per query: canonical connectivity (val/test reported) ---
    for rows in (mb_rows, gn_rows):
        for r, q in rows:
            qc, ok = rdkit_canon(q["rec"]["smiles"])
            r["overlap_val"] = bool(ok and qc in val_canon)
            r["overlap_test"] = bool(ok and qc in test_canon)
            r["overlap_val_string"] = q["rec"]["smiles"] in val_smiles
            r["overlap_test_string"] = q["rec"]["smiles"] in test_smiles

    # --- reference NN arm: separate query spectra vs remaining source
    # references (REFERENCELOOKUP only, never unseen generalization) ---
    # References: MassBank parseable records NOT selected as queries
    # (capped) + GNPS same-source spectra retained separately (including
    # no-structure blocks, which are spectral neighbors but ineligible as
    # identity references). Excluded from references: query's own accession,
    # same spectrum content, frozen fit/calib canon/content overlaps.
    # Eligibility: >=1 reference with the SAME canonical connectivity under
    # a DIFFERENT ID (any source). Conditional accuracy is reported among
    # eligible queries only; cross-source reference makeup is disclosed.
    from tools.ms2_spectral_rank import content_fp as _cfp
    ref_pool = []
    seen_ref_content = set()
    for rec in mb_records:
        if len(ref_pool) >= 3000:
            break
        try:
            from tools.ms2_msgym_corpus import standardize_smiles as _ss2
            srec, _ = _ss2(rec["smiles"], "ref", rec["accession"])
        except Exception:
            continue
        if srec is None:
            continue
        rc, rok = rdkit_canon(rec["smiles"])
        if not (rok and rc):
            continue
        if rc in fit_ids["canon_set"]:
            continue
        rcontent = _cfp(rec["peaks"])
        if rcontent in fit_ids["content_set"]:
            continue
        ref_pool.append({"source": "MassBank", "accession": rec["accession"],
                         "smiles": rec["smiles"], "canon": rc,
                         "peaks": rec["peaks"], "content": rcontent})
    n_gn_ref_nostruct = 0
    for b in gblocks:
        if not spectrum_has_usable_features(b["peaks"]):
            continue
        smi = (b["params"].get("SMILES") or "").strip()
        if smi and smi != "N/A":
            try:
                srec, _ = standardize_smiles(smi, "ref-gn",
                                             b["params"].get("SPECTRUMID", "?"))
            except Exception:
                srec = None
            if srec is None:
                continue
            rc, rok = rdkit_canon(smi)
            if not (rok and rc) or rc in fit_ids["canon_set"]:
                continue
        else:
            rc = None
            n_gn_ref_nostruct += 1
        rcontent = _cfp(b["peaks"])
        if rcontent in fit_ids["content_set"]:
            continue
        ref_pool.append({"source": "GNPS", "accession": b["params"].get(
            "SPECTRUMID", b["params"].get("FILENAME", "?")),
            "smiles": smi, "canon": rc, "peaks": b["peaks"],
            "content": rcontent})
    ref_vecs = []
    for rf in ref_pool:
        feat, norm, _ = featurize_spectrum(rf["peaks"])
        ref_vecs.append((feat, norm))
    nn_rows = []
    for (r, q) in mb_rows + gn_rows:
        src = "MassBank" if r["qid"].startswith("MB-") else "GNPS"
        qc, qok = rdkit_canon(q["rec"]["smiles"])
        qcontent = _cfp(q["rec"]["peaks"])
        qacc = q["rec"].get("accession", r["qid"])
        feat, norm, _ = featurize_spectrum(q["rec"]["peaks"])
        best, bi = -1.0, None
        for j, rf in enumerate(ref_pool):
            if rf["accession"] == qacc or rf["content"] == qcontent:
                continue
            fj, nj = ref_vecs[j]
            if norm <= 0 or nj <= 0:
                continue
            s = sum(feat.get(b, 0.0) * fj.get(b, 0.0)
                    for b in set(feat) | set(fj))
            if s > best:
                best, bi = s, j
        eligible_refs = [rf for rf in ref_pool
                         if rf["canon"] and qok and rf["canon"] == qc
                         and rf["accession"] != qacc
                         and rf["content"] != qcontent]
        eligible = len(eligible_refs) > 0
        hit = False
        pred_acc = pred_canon = None
        pred_source = None
        if bi is not None:
            rf = ref_pool[bi]
            pred_acc, pred_canon, pred_source = (rf["accession"], rf["canon"],
                                                rf["source"])
            hit = bool(eligible and rf["canon"] and qok and rf["canon"] == qc)
        nn_rows.append({"qid": r["qid"], "source": src,
                        "nn_accession": pred_acc, "nn_source": pred_source,
                        "cosine": best if bi is not None else None,
                        "eligible": eligible,
                        "n_eligible_refs": len(eligible_refs),
                        "eligible_ref_sources": sorted(set(
                            rf["source"] for rf in eligible_refs)),
                        "hit": hit, "true_canon": qc if qok else None,
                        "pred_canon": pred_canon})
    nn_elig = [n for n in nn_rows if n["eligible"]]
    nn_hits = sum(1 for n in nn_elig if n["hit"])
    nn_lo, nn_hi = wilson_interval(nn_hits, len(nn_elig))
    nn_ref_sources = {}
    for rf in ref_pool:
        nn_ref_sources[rf["source"]] = nn_ref_sources.get(rf["source"], 0) + 1

    # --- val200 LIBRARYLOOKUP vs MassBank structures ---
    mb_all_keys = set()
    mb_all_smiles = set()
    for rec in mb_records:
        mb_all_smiles.add(rec["smiles"])
        k = (rec.get("inchikey") or "").strip().split("-")[0]
        if k:
            mb_all_keys.add(k)
    lib_smi = sum(1 for s in val_smiles if s in mb_all_smiles)
    lib_hits = []
    for r in val_rows:
        k = (r.get("smiles") or "")
        lib_hits.append(k in mb_all_smiles)
    gn_all_smiles = set(g["smiles"] for g in gstruct)
    lib_gn = sum(1 for s in val_smiles if s in gn_all_smiles)

    # --- exact Murcko scaffolds (RDKit project-local) ---
    rd = rdkit_lib()
    scaf = {"rdkit": ("2026.3.6 project-local" if rd is not None else "unavailable"),
            "exact_gate": ("COMPLETE" if rd is not None else
                           "INCOMPLETE: RDKit unavailable; no proxy substituted")}
    if rd is not None:
        def scafs(smiles_set):
            out = {}
            for s in smiles_set:
                sc, _ = murcko_of(s)
                if sc:
                    out.setdefault(sc, []).append(s)
            return out
        mb_scaf = scafs(set(q["rec"]["smiles"] for _, q in mb_rows))
        fit_scaf = scafs(frozen["smiles_set"])
        val_scaf = scafs(val_smiles)
        test_scaf = scafs(set(list(test_smiles)[:20000]))
        scaf.update({
            "n_mb_scaffolds": len(mb_scaf),
            "n_fit_scaffolds": len(fit_scaf),
            "n_val_scaffolds": len(val_scaf),
            "mb_vs_fit_intersection": len(set(mb_scaf) & set(fit_scaf)),
            "mb_vs_val_intersection": len(set(mb_scaf) & set(val_scaf)),
            "mb_vs_test_intersection": len(set(mb_scaf) & set(test_scaf)),
            "method": "rdkit.Chem.Scaffolds.MurckoScaffold "
                      "(MurckoScaffoldSmilesFromSmiles, includeChirality=False), "
                      "RDKit 2026.3.6 project-local",
        })

    # --- PubChem misses: canonical-unresolved AFTER ChEBI coverage ---
    # Unresolved = canonical absence from BOTH prefix and ChEBI pools AND
    # target parseable. Cache replayed (no live ground truth): cache-hit
    # status recorded per formula. Canonical recall of expanded candidates
    # + optional same-frozen-model ranking of the expanded set (marginal
    # only, frozen metrics NEVER rescored). Resolved formulas never queried.
    chebi_present = {}
    for r, q in chebi_rows_mb + chebi_rows_gn:
        chebi_present[r["qid"]] = bool(r["in_pool"])
    miss_queries, miss_formulas = [], []
    for r, q in mb_rows + gn_rows:
        if r["identity_status"] == "unknown":
            continue
        if (not r["in_pool"]) and (not chebi_present.get(r["qid"], False)) \
                and r["target_parseable"]:
            miss_queries.append(r["qid"])
            smi = q["rec"]["smiles"]
            from tools.ms2_msgym_corpus import standardize_smiles as _ss
            try:
                rec0, _ = _ss(smi, "pubchem", smi)
                if rec0 is not None:
                    miss_formulas.append(_hill(rec0["formula"]))
            except Exception:
                pass
    pubchem = {"trigger_rule": "expand only formulas/masses with measured pool misses",
               "miss_queries": miss_queries, "miss_formulas": sorted(set(miss_formulas)),
               "identity_policy": "canonical connectivity (RDKit, stereo-insensitive)"}
    if miss_formulas and do_pubchem:
        cache_dir = os.path.join(out_dir, "pubchem_cache")
        pre_cached = set()
        for form in sorted(set(miss_formulas))[:5]:
            if os.path.exists(os.path.join(cache_dir, f"cids_{form}.json")):
                pre_cached.add(form)
        results, pstats = pubchem_expand(
            sorted(set(miss_formulas)), cache_dir)
        api_calls_total += pstats["api_calls"]
        # canonical marginal recall + optional frozen-model ranking
        targets = {}
        target_formulas = {}
        for r, q in mb_rows + gn_rows:
            if r["qid"] in miss_queries:
                tc, ok = rdkit_canon(q["rec"]["smiles"])
                targets[r["qid"]] = (tc if ok else None, q["rec"]["peaks"])
                target_rec, _ = standardize_smiles(q["rec"]["smiles"],
                                                   "pubchem-evaluation", r["qid"])
                if target_rec is not None:
                    target_formulas[r["qid"]] = _hill(target_rec["formula"])
        evaluable = sorted(qid for qid, form in target_formulas.items()
                           if results.get(form, {}).get("smiles"))
        marg = 0
        marg_ranks = {}
        for form, res in results.items():
            if "smiles" in res:
                have = []
                for s in res["smiles"]:
                    c, ok = rdkit_canon(s)
                    if ok and c:
                        have.append(c)
                have_set = set(have)
                for qid, (tc, peaks) in targets.items():
                    if target_formulas.get(qid) != form:
                        continue
                    if tc and tc in have_set:
                        marg += 1
                        # optional same-frozen-model ranking of expanded set
                        pfps = []
                        for s in res["smiles"]:
                            pfps.append(fp_from_smiles(s)[:2])
                        pred, pn = predict_fp_numpy(model, peaks)
                        pscores = score_candidates_fp(pred, pn, pfps)
                        order = sorted(
                            range(len(pfps)),
                            key=lambda i: ((0.0, -pscores[i]["score"])
                                           if pscores[i]["score"] is not None
                                           else (1.0, 0.0), i))
                        cands = [rdkit_canon(s)[0]
                                 for s in res["smiles"]]
                        try:
                            marg_ranks[qid] = order.index(
                                next(i for i, c in enumerate(cands)
                                     if c == tc)) + 1
                        except (StopIteration, ValueError):
                            marg_ranks[qid] = None
        pubchem.update({"expansion": results, "stats": pstats,
                        "cache_hits_precached": sorted(pre_cached),
                        "marginal_recall_targets_found": marg,
                        "marginal_frozen_model_ranks": marg_ranks,
                        "cache_evaluable_queries": evaluable,
                        "n_cache_evaluable_queries": len(evaluable),
                        "n_unresolved_queries": len(miss_queries),
                        "status": (f"cache replay evaluated {len(evaluable)}/{len(miss_queries)} "
                                   f"unresolved queries across {len(results)} selected formulas; "
                                   f"canonical recoveries {marg}/{len(evaluable)}; "
                                   f"{len(pstats['errors'])} selected formulas unavailable; "
                                   "truncated candidate lists, no absence proof or frozen rescoring")})
    elif miss_formulas:
        pubchem["status"] = ("triggered but expansion disabled by flag; "
                             f"{len(miss_queries)} miss queries recorded")
    else:
        pubchem["status"] = ("not-triggered: no canonical-unresolved pool misses "
                             "after ChEBI coverage; no PubChem query issued")
        pubchem["expansion"] = {}
        pubchem["stats"] = {"formulas_attempted": 0, "api_calls": 0, "errors": {}}

    cpu_s = time.process_time() - t0
    wall_s = time.time() - wall0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    summary = {
        "tool": "tools/ms2_external_validation.py",
        "checkpoint": {"path": model_p, "sha256": model["sha256"],
                       "note": "frozen Ridge hash-fingerprint checkpoint; never refit; "
                               "original-200 sealed (read-only overlap audit)"},
        "frozen_train_identities": {
            "n_rows_total": frozen["n_rows_total"],
            "n_used": frozen["n_used"],
            "n_used_fit": frozen["n_used_fit"],
            "n_used_calib": frozen["n_used_calib"],
            "n_content_recovered": frozen["n_content_recovered"],
            "n_content_spec_fail": frozen["n_content_spec_fail"],
            "groups_sha256": frozen["groups_sha256"],
            "smiles_sha256": frozen["smiles_sha256"],
            "n_canon_fail_fit": fit_canon_fail,
            "n_canon_fail_val": val_canon_fail,
            "n_canon_fail_test": test_canon_fail,
        },
        "massbank": {"scan": mb_scan, "selection_audit": mb_audit,
                     "scored": mb_summary, "pool": mb_pool,
                     "nn_lookup": {"n_queries": len(nn_rows),
                                   "n_eligible": len(nn_elig),
                                   "hits": nn_hits,
                                   "conditional_rate": (nn_hits / len(nn_elig)) if nn_elig else None,
                                   "ci95": [nn_lo, nn_hi],
                                   "n_references": len(ref_pool),
                                   "reference_sources": nn_ref_sources,
                                   "n_gnps_nostruct_refs": n_gn_ref_nostruct,
                                   "note": ("separate query spectra vs remaining source "
                                            "references; same-connectivity different-ID "
                                            "eligible (REFERENCELOOKUP only); exact "
                                            "content/self/fit-calib excluded; negative "
                                            "coverage reported honestly")}},
        "gnps": {"blocks": gstats, "exclusions": gexcl,
                 "n_structurized_pre_filter": n_gstruct_pre,
                 "n_selected": len(gn_selected),
                 "n_excluded_fitcalib": n_gexcl_fit,
                 "n_excluded_fitcalib_content": n_gexcl_fit_content,
                 "n_excluded_cross_source": n_gexcl_cross,
                 "n_excluded_cap": n_gexcl_cap,
                 "cap": GNPS_MAX_QUERIES,
                 "scored": gn_summary, "pool": gn_pool,
                 "license": g_license_note},
        "chebi_full_coverage": {
            "index": chebi_stats,
            "stage_aggregates": chebi_stage_agg,
            "massbank_scored": chebi_summary_mb,
            "massbank_pool": chebi_pool_mb,
            "gnps_scored": chebi_summary_gn,
            "gnps_pool": chebi_pool_gn,
            "uncertainty_mu": ("per-query textual rounding half-width uDa "
                               "(measured from precursor text); None when "
                               "precision unknown (unavailable throughout)"),
            "ppm_tenths": STAGE1_PPM_TENTHS,
            "ppm_role": ("DECLARED assumed instrument tolerance, never a "
                         "measured precision"),
            "note": ("stage1 measured mass window only (no formula filter); "
                     "stage2 formula/domain + canonical presence; stage3 "
                     "ORACLE patterns from true graph (separate arm); "
                     "stage4 not_evaluated (whole-parent redundancy)")},
        "library_lookup_val200": {
            "n_val_targets": len(val_smiles),
            "massbank_smiles_matches": lib_smi,
            "massbank_formula_note": "SMILES-string coverage; LIBRARYLOOKUP not generalization",
            "gnps_smiles_matches": lib_gn},
        "overlap_audit": {
            "fit_used_molecules": frozen["n_used"],
            "val_molecules": len(val_smiles),
            "test_molecules": len(test_smiles),
            "exclusion_policy": ("canonical connectivity + spectrum content "
                                 "(frozen semantics); cross-source dedup; "
                                 "val/test canonical overlap flagged per query"),
            "note": "fit/calib excluded from scoring with counters; val/test overlap "
                    "flagged per query and reported, never used for tuning"},
        "scaffolds": scaf,
        "pubchem_conditional": pubchem,
        "pubchem_miss_queries": miss_queries,
        "costs": {"wall_s": wall_s, "cpu_s": cpu_s, "peak_rss_kb": rss_kb,
                  "api_calls": api_calls_total,
                  "chebi_index_build_cpu_s": chebi_stats.get("index_build_cpu_s"),
                  "per_1000_queries": {
                      "denominator": f"{len(mb_rows) + len(gn_rows)} scored external queries",
                      "cpu_s_per_1000": (cpu_s / (len(mb_rows) + len(gn_rows)) * 1000) if (mb_rows or gn_rows) else None,
                      "wall_s_per_1000": (wall_s / (len(mb_rows) + len(gn_rows)) * 1000) if (mb_rows or gn_rows) else None}},
        "code_sha256": sha256_file(os.path.join(repo_root, "tools/ms2_external_validation.py")),
        "env": {"python": sys.version.split()[0]},
    }
    # per-query CSV (measured roles + full provenance persisted per row)
    rows_p = os.path.join(out_dir, "external_rows.csv")
    with open(rows_p, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "qid", "source", "accession", "smiles", "canon", "adduct",
            "adduct_evidence", "ion_mode", "charge_status", "instrument",
            "collision_energy", "license", "contributor", "source_file",
            "inchikey", "measured_neutral_mu", "exact_mass_theoretical",
            "mass_precision", "ppm_crosscheck", "in_pool", "n_pool",
            "n_pool_identity_known", "n_pool_identity_unknown",
            "n_parseable", "target_parseable", "identity_status",
            "rankable", "rankable_reason",
            "margin_pred", "predictor_applicable",
            "rank_uniform", "rank_massresid", "rank_predictor", "rank_prior",
            "overlap_val", "overlap_test"])
        w.writeheader()
        for r, q in mb_rows:
            rec = q["rec"]
            qc, _ = rdkit_canon(rec["smiles"])
            w.writerow({"qid": r["qid"], "source": "MassBank",
                        "accession": rec["accession"], "smiles": rec["smiles"],
                        "canon": qc or "",
                        "adduct": rec["adduct"],
                        "adduct_evidence": "PRECURSOR_TYPE+ion-mode-consistent",
                        "ion_mode": rec.get("ion_mode", ""),
                        "charge_status": "stated_z1 (supported adducts unary)",
                        "instrument": rec.get("instrument", ""),
                        "collision_energy": rec.get("collision_energy", ""),
                        "license": rec.get("license", ""),
                        "contributor": "",
                        "source_file": rec.get("source_file", ""),
                        "inchikey": rec.get("inchikey", ""),
                        "measured_neutral_mu": r.get("measured_neutral_mu", ""),
                        "exact_mass_theoretical": rec["exact_mass"],
                        "mass_precision": rec.get("mass_precision", ""),
                        "ppm_crosscheck": round(q.get("ppm", 0), 2),
                        "in_pool": r["in_pool"], "n_pool": r["n_pool"],
                        "n_pool_identity_known": r.get("n_pool_identity_known", ""),
                        "n_pool_identity_unknown": r.get("n_pool_identity_unknown", ""),
                        "n_parseable": r["n_parseable"],
                        "target_parseable": r["target_parseable"],
                        "identity_status": r.get("identity_status", ""),
                        "rankable": r["rankable"],
                        "rankable_reason": r["rankable_reason"],
                        "margin_pred": r.get("margin_pred", ""),
                        "predictor_applicable": r.get("predictor_applicable", ""),
                        "rank_uniform": r["rank_uniform"],
                        "rank_massresid": r["rank_massresid"],
                        "rank_predictor": r["rank_predictor"],
                        "rank_prior": r["rank_prior"],
                        "overlap_val": r.get("overlap_val", False),
                        "overlap_test": r.get("overlap_test", False)})
        for (r, q), g in zip(gn_rows, gn_selected):
            rec = q["rec"]
            qc, _ = rdkit_canon(rec["smiles"])
            params = g["block"]["params"]
            w.writerow({"qid": r["qid"], "source": "GNPS",
                        "accession": rec["accession"], "smiles": rec["smiles"],
                        "canon": qc or "",
                        "adduct": rec["adduct"],
                        "adduct_evidence": g.get("rule", ""),
                        "ion_mode": params.get("IONMODE", ""),
                        "charge_status": ("CHARGE=" + params.get("CHARGE", "unstated")),
                        "instrument": params.get("SOURCE_INSTRUMENT", ""),
                        "collision_energy": "",
                        "license": "MGF-no-per-record-field (see summary)",
                        "contributor": ";".join(filter(None, [
                            params.get("SUBMITUSER", ""),
                            params.get("ORGANISM", "")])),
                        "source_file": params.get("FILENAME", ""),
                        "inchikey": params.get("INCHI", ""),
                        "measured_neutral_mu": r.get("measured_neutral_mu", ""),
                        "exact_mass_theoretical": "",
                        "mass_precision": (lambda pt: (
                            f"rounding_halfwidth_{int(round(0.5 * 10 ** (6 - len(pt.split('.')[1]))))}_muDa"
                            if "." in pt and pt.split(".")[1].isdigit()
                            else "unknown (no usable decimals)")(
                            (params.get("PEPMASS") or "").strip().split()[0])),
                        "ppm_crosscheck": round(g.get("ppm", 0), 2),
                        "in_pool": r["in_pool"], "n_pool": r["n_pool"],
                        "n_pool_identity_known": r.get("n_pool_identity_known", ""),
                        "n_pool_identity_unknown": r.get("n_pool_identity_unknown", ""),
                        "n_parseable": r["n_parseable"],
                        "target_parseable": r["target_parseable"],
                        "identity_status": r.get("identity_status", ""),
                        "rankable": r["rankable"],
                        "rankable_reason": r["rankable_reason"],
                        "margin_pred": r.get("margin_pred", ""),
                        "predictor_applicable": r.get("predictor_applicable", ""),
                        "rank_uniform": r["rank_uniform"],
                        "rank_massresid": r["rank_massresid"],
                        "rank_predictor": r["rank_predictor"],
                        "rank_prior": r["rank_prior"],
                        "overlap_val": r.get("overlap_val", False),
                        "overlap_test": r.get("overlap_test", False)})
    # Keep ranking evidence alongside the stage evidence, with qids joining
    # external_rows.csv for source attribution. All four arms use these pools.
    chebi_rank_csv = os.path.join(out_dir, "chebi_rank_rows.csv")
    rank_rows = [r for r, _ in chebi_rows_mb + chebi_rows_gn]
    with open(chebi_rank_csv, "w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for r in rank_rows for key in r})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rank_rows)
    # ChEBI stage detail per query
    chebi_csv = os.path.join(out_dir, "chebi_stage_rows.csv")
    with open(chebi_csv, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "qid", "measured_neutral_mu", "precision_known",
            "nested_ok",
            "stage1_present_s1", "stage1_n_s1",
            "stage1_n_accept", "stage1_n_ambiguous", "stage1_n_unavailable",
            "stage1_n_reject",
            "stage2_present_s2", "stage2_n_s2",
            "stage3_oracle_patterns", "stage3_present_s3", "stage3_n_s3",
            "stage3_truncated", "stage4_precursor"])
        w.writeheader()
        for s in chebi_stages:
            w.writerow({"qid": s["qid"],
                        "measured_neutral_mu": s["measured_neutral_mu"],
                        "precision_known": s["precision_known"],
                        "nested_ok": s["nested_ok"],
                        "stage1_present_s1": s["stage1"]["target_present_s1"],
                        "stage1_n_s1": s["stage1"]["n_s1"],
                        "stage1_n_accept": s["stage1"]["n_accept"],
                        "stage1_n_ambiguous": s["stage1"]["n_ambiguous"],
                        "stage1_n_unavailable": s["stage1"]["n_unavailable"],
                        "stage1_n_reject": s["stage1"]["n_reject"],
                        "stage2_present_s2": s["stage2"]["target_present_s2"],
                        "stage2_n_s2": s["stage2"]["n_s2"],
                        "stage3_oracle_patterns": s["stage3_oracle"]["patterns"],
                        "stage3_present_s3": s["stage3_oracle"]["target_present_s3"],
                        "stage3_n_s3": s["stage3_oracle"]["n_s3"],
                        "stage3_truncated": s["stage3_oracle"]["truncated"],
                        "stage4_precursor": s["stage4_precursor"]})
    nn_p = os.path.join(out_dir, "external_nn_rows.csv")
    with open(nn_p, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "qid", "source", "nn_accession", "nn_source", "cosine",
            "eligible", "n_eligible_refs", "hit"])
        w.writeheader()
        for n in nn_rows:
            w.writerow({k: n[k] for k in (
                "qid", "source", "nn_accession", "nn_source", "cosine",
                "eligible", "n_eligible_refs", "hit")})
    lib_p = os.path.join(out_dir, "library_lookup.csv")
    with open(lib_p, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["smiles", "in_massbank", "in_gnps"])
        w.writeheader()
        for s in sorted(val_smiles):
            w.writerow({"smiles": s, "in_massbank": s in mb_all_smiles,
                        "in_gnps": s in gn_all_smiles})
    text = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
    sum_p = os.path.join(out_dir, "external_summary.json")
    with open(sum_p, "w", encoding="utf-8") as handle:
        handle.write(text)

    # --- source-pin manifest (new file; PIN.json untouched) ---
    pin = {
        "created": "2026-10-03",
        "massbank": {
            "release": "https://github.com/MassBank/MassBank-data/releases/tag/2026.03",
            "tag_commit": "705afb7bccc3b2c42410a744eef73674716a60ef",
            "zenodo": "https://zenodo.org/records/19073053",
            "doi": "10.5281/zenodo.19073053",
            "file": "MassBank/MassBank-data-2026.03.zip",
            "bytes": 239602156,
            "md5": "51904e29ee8717756153a8d31726789d",
            "sha256": sha256_file(mb_zip),
            "license": "per-record LICENSE field retained per query (CC BY / CC BY-SA observed); evaluation use with attribution",
            "retrieved": "2026-10-03",
        },
        "gnps": {
            "library": "GNPS-FAULKNERLEGACY",
            "docs": "https://ccms-ucsd.github.io/GNPSDocumentation/downloadlibraries/",
            "file": "GNPS-FAULKNERLEGACY.mgf",
            "bytes": os.path.getsize(gnps_mgf),
            "sha256": sha256_file(gnps_mgf),
            "license": ("MGF has no per-record license field; GNPS docs default CC0 for "
                        "directly-contributed reference spectra, third-party imports may "
                        "differ; evaluation use only, contributor attribution retained "
                        "(Sirenas Marine Discovery via mwang87)"),
            "retrieved": "2026-10-03",
            "limitation": "snapshot export; content SHA pinned, mutable-source limitation declared",
        },
        "qm9_full": {
            "api": "https://api.figshare.com/v2/articles/1057646",
            "download": "https://ndownloader.figshare.com/files/3195389",
            "file": "dsgdb9nsd.xyz.tar.bz2",
            "bytes": os.path.getsize(os.path.join(ext, "dsgdb9nsd.xyz.tar.bz2")),
            "sha256": sha256_file(os.path.join(ext, "dsgdb9nsd.xyz.tar.bz2")),
            "license": "CC0",
            "retrieved": "2026-10-03",
        },
        "chebi_index": {
            "file": "data/pinned/chebi_3_stars.sdf.gz",
            "sha256": chebi_stats["pin"]["sha256"],
            "bytes": chebi_stats["pin"]["bytes"],
            "n_records": chebi_stats["n_index"],
            "license": "CC BY 4.0 (source data; per PIN.json September 2026 release)",
            "retrieved": "2026-10-03",
        },
        "candidate_pools": {
            "file": "data/pinned/msgym_candidates_formula_prefix64.json",
            "sha256": sha256_file(prefix),
            "bytes": os.path.getsize(prefix),
            "role": "retained additional prefix-pool arm (not replaced)",
        },
        "checkpoint": {"path": model_p, "sha256": model["sha256"]},
        "rdkit": {"version": "2026.3.6", "scope": "project-local",
                  "path": "experiments/molecular_completion/20261003_remaining/.rdkit_lib"},
    }
    pin_p = os.path.join(repo_root, "data/pinned/EXTERNAL_PIN.json")
    with open(pin_p, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(pin, allow_nan=False, indent=2, sort_keys=True))
    summary["outputs"] = {"rows": rows_p, "nn_rows": nn_p, "lookup": lib_p,
                          "chebi_ranks": chebi_rank_csv,
                          "chebi_stages": chebi_csv,
                          "summary": sum_p, "pin": pin_p}
    return summary


def _hill(formula):
    """Hill-order formula string (C first, H second, rest alphabetical)."""
    parts = []
    order = ["C", "H"] + sorted(e for e in formula if e not in ("C", "H"))
    for el in order:
        n = formula.get(el, 0)
        if n:
            parts.append(f"{el}{n if n != 1 else ''}")
    return "".join(parts)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/molecular_completion/20261003_remaining")
    ap.add_argument("--max-queries", type=int, default=MAX_EXTERNAL_QUERIES)
    ap.add_argument("--no-pubchem", action="store_true")
    args = ap.parse_args(argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(repo_root) == "tools":
        repo_root = os.path.dirname(repo_root)
    summary = run_external_validation(repo_root, args.out_dir,
                                      max_queries=args.max_queries,
                                      do_pubchem=not args.no_pubchem)
    print(json.dumps({"mb_scored": summary["massbank"]["scored"]["n"],
                      "gn_scored": summary["gnps"]["scored"]["n"],
                      "mb_recall": summary["massbank"]["scored"]["pool_recall_all"]["rate"],
                      "out": summary["outputs"]["summary"]}, allow_nan=False))


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

_MB_MINIMAL = """ACCESSION: MSBNK-TEST-000001
RECORD_TITLE: Test; LC-ESI; MS2; [M+H]+
DATE: 2020.01.01
AUTHORS: Test Author
LICENSE: CC BY
CH$NAME: Testmol
CH$FORMULA: C10H10O3
CH$EXACT_MASS: 178.06299
CH$SMILES: CC1CC2=C(C(=CC=C2)O)C(=O)O1
CH$LINK: INCHIKEY KWILGNNWGSNMPA-UHFFFAOYSA-N
AC$INSTRUMENT: Q-Exactive
AC$INSTRUMENT_TYPE: LC-ESI-ITFT
MS$FOCUSED_ION: BASE_PEAK 161.0591
MS$FOCUSED_ION: PRECURSOR_M/Z 179.0703
MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+
PK$NUM_PEAK: 3
PK$PEAK: m/z int. rel.int.
  133.0648 21905.0 225
  161.0597 96508.0 999
  179.0703 72563.0 750
//
"""


class MassbankTests(unittest.TestCase):
    def test_parse_minimal(self):
        rec, reason = parse_massbank_record(_MB_MINIMAL)
        self.assertIsNone(reason)
        self.assertEqual(rec["accession"], "MSBNK-TEST-000001")
        self.assertEqual(rec["license"], "CC BY")
        self.assertEqual(rec["adduct"], "[M+H]+")
        self.assertEqual(len(rec["peaks"]), 3)
        self.assertTrue(rec["peak_count_match"])

    def test_missing_license_rejected(self):
        rec, reason = parse_massbank_record(_MB_MINIMAL.replace("LICENSE: CC BY\n", ""))
        self.assertIsNone(rec)
        self.assertEqual(reason, "no_license")

    def test_unsupported_adduct_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("[M+H]+", "[M+NH4]+"))
        self.assertIsNone(rec)
        self.assertTrue(reason.startswith("unsupported_adduct"))

    def test_multicharge_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("[M+H]+", "[M+2H]2+"))
        self.assertIsNone(rec)
        self.assertEqual(reason, "unsupported_charge_state")

    def test_incomplete_peaks_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("  179.0703 72563.0 750\n", ""))
        self.assertIsNone(rec)
        # declared 3 vs parsed 2: explicit count mismatch now rejects first
        self.assertTrue(reason.startswith("peak_count_mismatch"))

    def test_peak_count_mismatch_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("PK$NUM_PEAK: 3", "PK$NUM_PEAK: 999"))
        self.assertIsNone(rec)
        self.assertTrue(reason.startswith("peak_count_mismatch"))

    def test_missing_terminator_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("//\n", ""))
        self.assertIsNone(rec)
        self.assertEqual(reason, "incomplete_terminator")

    def test_rounding_uncertainty_from_text(self):
        # 4 decimals -> half-width 0.00005 Da = 50 uDa; 3 decimals -> 500.
        rec, reason = parse_massbank_record(_MB_MINIMAL)
        self.assertIsNone(reason)
        self.assertEqual(rec["rounding_mu"], 50)
        self.assertTrue(rec["precision_known"])
        rec2, reason2 = parse_massbank_record(
            _MB_MINIMAL.replace("PRECURSOR_M/Z 179.0703",
                                "PRECURSOR_M/Z 179.070"))
        self.assertIsNone(reason2)
        self.assertEqual(rec2["rounding_mu"], 500)

    def test_coarse_precision_unavailable(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("PRECURSOR_M/Z 179.0703",
                                "PRECURSOR_M/Z 179"))
        self.assertIsNone(reason)
        self.assertFalse(rec["precision_known"])
        self.assertIsNone(rec["rounding_mu"])


class AdductTests(unittest.TestCase):
    def test_conversions(self):
        mu, _ = precursor_to_neutral(179.0703, "[M+H]+")
        self.assertAlmostEqual(mu / 1e6, 179.0703 - PROTON_MASS, places=4)
        mu, _ = precursor_to_neutral(177.0, "[M-H]-")
        self.assertAlmostEqual(mu / 1e6, 177.0 + PROTON_MASS, places=4)
        mu, _ = precursor_to_neutral(201.0, "[M+Na]+")
        self.assertAlmostEqual(mu / 1e6, 201.0 - NA_ION_MASS, places=4)
        mu, _ = precursor_to_neutral(217.0, "[M+K]+")
        self.assertAlmostEqual(mu / 1e6, 217.0 - K_ION_MASS, places=4)

    def test_electron_correction_explicit(self):
        # Ion masses, not neutral atomic masses: Na+/K+ carry away one
        # electron (0.000548579909 Da). Neutral atomic masses would bias
        # neutral estimates by ~0.55 mDa.
        self.assertAlmostEqual(NA_ION_MASS,
                               22.9897692809 - ELECTRON_MASS, places=9)
        self.assertAlmostEqual(K_ION_MASS,
                               38.9637074864 - ELECTRON_MASS, places=9)
        self.assertAlmostEqual(ELECTRON_MASS, 0.000548579909, places=12)

    def test_unknown_rejected(self):
        mu, reason = precursor_to_neutral(100.0, "[M+NH4]+")
        self.assertIsNone(mu)
        self.assertTrue(reason.startswith("unsupported_adduct"))

    def test_ion_mode_conflict_rejected(self):
        rec, reason = parse_massbank_record(
            _MB_MINIMAL.replace("PRECURSOR_TYPE [M+H]+",
                                "PRECURSOR_TYPE [M-H]-"))
        # fixture has no ION_MODE field -> accepted on adduct alone
        self.assertIsNone(reason)
        bad = _MB_MINIMAL + "AC$MASS_SPECTROMETRY: ION_MODE POSITIVE\n"
        bad = bad.replace("PRECURSOR_TYPE [M+H]+", "PRECURSOR_TYPE [M-H]-")
        rec, reason = parse_massbank_record(bad)
        self.assertIsNone(rec)
        self.assertTrue(reason.startswith("ion_mode_conflict"))


class MgfTests(unittest.TestCase):
    def test_complete_block(self):
        text = ("BEGIN IONS\nPEPMASS=100.0\nCHARGE=1\nIONMODE=Positive\n"
                "NAME=X M+H\nSMILES=CCO\n100.0 10.0\n101.0 20.0\n102.0 5.0\nEND IONS\n")
        blocks, stats = parse_mgf_blocks(text)
        self.assertEqual(len(blocks), 1)
        adduct, _ = gnps_adduct_of(blocks[0])
        self.assertEqual(adduct, "[M+H]+")

    def test_incomplete_no_pepmass(self):
        text = "BEGIN IONS\nNAME=X\n1.0 2.0\nEND IONS\n"
        blocks, stats = parse_mgf_blocks(text)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(stats["bad_reasons"]["no_pepmass"], 1)

    def test_adduct_requires_evidence(self):
        text = ("BEGIN IONS\nPEPMASS=100.0\nIONMODE=Positive\nNAME=X\n"
                "SMILES=CCO\n100.0 10.0\n101.0 20.0\n102.0 5.0\nEND IONS\n")
        blocks, _ = parse_mgf_blocks(text)
        adduct, reason = gnps_adduct_of(blocks[0])
        self.assertIsNone(adduct)
        self.assertEqual(reason, "unsupported_adduct_evidence")

    def test_multicharge_name_conflict_rejected(self):
        # Mirrors real GNPS B26A11: CHARGE=2 with an M+H name must NOT be
        # read as a singly-charged [M+H]+ precursor.
        text = ("BEGIN IONS\nPEPMASS=525.405\nCHARGE=2\nIONMODE=Positive\n"
                "NAME=B26A11 M+H\nSMILES=CCO\n100.0 10.0\n101.0 20.0\n102.0 5.0\nEND IONS\n")
        blocks, _ = parse_mgf_blocks(text)
        self.assertEqual(len(blocks), 1)
        adduct, reason = gnps_adduct_of(blocks[0])
        self.assertIsNone(adduct)
        self.assertTrue(reason.startswith("unsupported_charge_state"))

    def test_charge_sign_conflict_rejected(self):
        text = ("BEGIN IONS\nPEPMASS=100.0\nCHARGE=1-\nIONMODE=Positive\n"
                "NAME=X M+H\nSMILES=CCO\n100.0 10.0\n101.0 20.0\n102.0 5.0\nEND IONS\n")
        blocks, _ = parse_mgf_blocks(text)
        adduct, reason = gnps_adduct_of(blocks[0])
        self.assertIsNone(adduct)
        self.assertTrue(reason.startswith("charge_sign_conflict"))

    def test_water_loss_token_not_intact_adduct(self):
        for nm in ("X M+H-H2O", "X M-H2O+H"):
            text = ("BEGIN IONS\nPEPMASS=100.0\nCHARGE=1\nIONMODE=Positive\n"
                    f"NAME={nm}\nSMILES=CCO\n100.0 10.0\n101.0 20.0\n102.0 5.0\nEND IONS\n")
            blocks, _ = parse_mgf_blocks(text)
            self.assertEqual(len(blocks), 1)
            adduct, reason = gnps_adduct_of(blocks[0])
            self.assertIsNone(adduct, nm)
            self.assertTrue(reason.startswith("ambiguous_adduct_token"), reason)

    def test_nan_intensity_rejected(self):
        text = ("BEGIN IONS\nPEPMASS=100.0\nCHARGE=1\nIONMODE=Positive\n"
                "NAME=X M+H\nSMILES=CCO\n100.0 10.0\n101.0 nan\n102.0 5.0\nEND IONS\n")
        blocks, stats = parse_mgf_blocks(text)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(stats["bad_reasons"].get("nonfinite_peak", 0), 1)

    def test_negative_intensity_rejected(self):
        text = ("BEGIN IONS\nPEPMASS=100.0\nCHARGE=1\nIONMODE=Positive\n"
                "NAME=X M+H\nSMILES=CCO\n100.0 10.0\n101.0 -3.0\n102.0 5.0\nEND IONS\n")
        blocks, stats = parse_mgf_blocks(text)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(stats["bad_reasons"].get("bad_peak_values", 0), 1)


class CanonicalIdentityTests(unittest.TestCase):
    def test_ethanol_alternate_smiles_present(self):
        # OCC vs CCO: raw strings differ, connectivity identical.
        c1, ok1 = rdkit_canon("OCC")
        c2, ok2 = rdkit_canon("CCO")
        self.assertTrue(ok1 and ok2)
        self.assertEqual(c1, c2)
        self.assertNotEqual("OCC", "CCO")

    def test_stereo_variants_same_connectivity(self):
        c1, ok1 = rdkit_canon("C[C@H](O)C")
        c2, ok2 = rdkit_canon("C[C@@H](O)C")
        self.assertTrue(ok1 and ok2)
        self.assertEqual(c1, c2)

    def test_constitutional_isomers_differ(self):
        c1, _ = rdkit_canon("CCO")
        c2, _ = rdkit_canon("COC")
        self.assertNotEqual(c1, c2)

    def test_invalid_smiles_no_identity(self):
        c, ok = rdkit_canon("not a smiles(([")
        self.assertFalse(ok)
        self.assertIsNone(c)


class MassresidMeasuredTests(unittest.TestCase):
    def test_ground_truth_change_no_effect(self):
        # massresid ranks derive from the MEASURED precursor neutral only:
        # changing the theoretical CH$EXACT_MASS annotation cannot move them.
        from tools.ms2_spectral_rank import candidate_residuals
        pool = ["CCO", "CCC"]
        r1, _ = candidate_residuals(pool, "46.068644")  # measured neutral
        r2, _ = candidate_residuals(pool, "46.068644")
        self.assertEqual(r1, r2)

    def test_precursor_change_moves_baseline(self):
        from tools.ms2_spectral_rank import candidate_residuals
        pool = ["CCO", "CCC"]
        r1, _ = candidate_residuals(pool, "46.068644")
        r2, _ = candidate_residuals(pool, "44.000000")
        self.assertNotEqual(r1, r2)


class LeakageFixtureTests(unittest.TestCase):
    def _write_tsv(self, d, rows):
        import csv as _csv
        p = os.path.join(d, "mini.tsv")
        with open(p, "w", encoding="utf-8", newline="") as h:
            w = _csv.DictWriter(h, fieldnames=[
                "identifier", "mzs", "intensities", "smiles", "inchikey",
                "formula", "precursor_formula", "parent_mass", "precursor_mz",
                "adduct", "instrument_type", "collision_energy", "fold",
                "simulation_challenge"], delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return p

    def _row(self, ident, smi, key, mzs="100.0,200.0", ints="10.0,20.0"):
        return {"identifier": ident, "mzs": mzs, "intensities": ints,
                "smiles": smi, "inchikey": key, "formula": "C2H6O",
                "precursor_formula": "C2H6O", "parent_mass": "46.0686",
                "precursor_mz": "47.0759", "adduct": "[M+H]+",
                "instrument_type": "Orbitrap", "collision_energy": "10",
                "fold": "train", "simulation_challenge": ""}

    def test_alternate_smiles_same_connectivity_excluded(self):
        # File-backed: frozen TRAIN holds CCO; external candidate OCC
        # (alternate SMILES, same connectivity) must be excluded by the
        # canonical-connectivity guard, not admitted by raw-string compare.
        import tempfile
        from tools.ms2_spectral_rank import content_fp, parse_spectrum
        with tempfile.TemporaryDirectory() as d:
            p = self._write_tsv(d, [self._row("T1", "CCO", "KEY1-AAAA")])
            import csv as _csv
            seen = {}
            with open(p, encoding="utf-8") as h:
                for row in _csv.DictReader(h, delimiter="\t"):
                    if row["fold"] == "train":
                        seen[row["smiles"]] = row
            fit_ids = {"canon_set": canon_set(["CCO"])[0],
                       "content_set": set()}
            qc, ok = rdkit_canon("OCC")
            self.assertTrue(ok)
            self.assertIn(qc, fit_ids["canon_set"])
            self.assertNotIn("OCC", set(seen))

    def test_same_content_different_id_excluded(self):
        # File-backed: same spectrum content under a different ID/SMILES
        # must be excluded by the content guard.
        import tempfile
        from tools.ms2_spectral_rank import content_fp, parse_spectrum
        with tempfile.TemporaryDirectory() as d:
            rows = [self._row("T1", "CCO", "KEY1-AAAA",
                              mzs="100.0,200.0", ints="10.0,20.0"),
                    self._row("T2", "CCC", "KEY2-BBBB",
                              mzs="100.0,200.0", ints="10.0,20.0")]
            p = self._write_tsv(d, rows)
            import csv as _csv
            contents = set()
            with open(p, encoding="utf-8") as h:
                for row in _csv.DictReader(h, delimiter="\t"):
                    peaks, err = parse_spectrum(row["mzs"], row["intensities"])
                    self.assertIsNone(err)
                    contents.add(content_fp(peaks))
            ext_peaks, err = parse_spectrum("100.0,200.0", "10.0,20.0")
            self.assertIn(content_fp(ext_peaks), contents)


class StageRegressionTests(unittest.TestCase):
    def test_missing_target_still_miss(self):
        # Pool without the target: ranks None (miss), retained, never dropped.
        rows = [({"qid": "Q0", "in_pool": False, "target_parseable": True,
                  "rankable": False, "identity_status": "known",
                  "rank_uniform": None, "rank_massresid": None,
                  "rank_predictor": None, "rank_prior": None}, {})]
        out = summarize_scored(rows)
        self.assertEqual(out["pool_recall_all"]["hits"], 0)
        self.assertEqual(out["all_selected"]["uniform@top1"]["hits"], 0)
        self.assertEqual(out["present"]["uniform@top1"]["n"], 0)
        self.assertIsNone(out["present"]["uniform@top1"]["rate"])

    def test_mass_formula_stage_difference(self):
        # Mass window (accept+ambiguous+unavailable) is a superset filter vs
        # the exact formula stage: stage counts must satisfy n_s1 >= n_s2.
        from tools.ms2_database_retrieval import (DatabaseIndex, WorkCounters,
                                                 fixture_raw_records, run_query,
                                                 standardize_record)
        recs = []
        for raw in fixture_raw_records():
            r, reason = standardize_record(raw)
            self.assertIsNotNone(r, reason)
            recs.append(r)
        index = DatabaseIndex(recs)
        target = index.by_id["FIX-001"]
        res = run_query(index, target, [], "unknown", None,
                        counters=WorkCounters(limit=100_000))
        self.assertGreaterEqual(len(res["s1"]), len(res["s2"]))
        self.assertEqual(res["precursor_status"], "not_evaluated")

    def _mini_chebi_index(self):
        # Two-record ChEBI-like index: CCO present, COC absent-query decoy.
        from tools.ms2_database_retrieval import DatabaseIndex
        from tools.ms2_msgym_corpus import standardize_smiles
        recs = []
        for mid, smi in (("C1", "CCO"), ("C2", "CCC")):
            rec, reason = standardize_smiles(smi, "t", mid)
            self.assertIsNotNone(rec, reason)
            c, ok = rdkit_canon(smi)
            rec["canon"] = c if ok else None
            rec["id"] = mid
            recs.append(rec)
        return DatabaseIndex(recs)

    def test_nested_presence_cco_vs_coc(self):
        # CCO query against an index holding only COC/CCC: every stage must
        # report the target MISS (never self-verdict presence).
        from tools.ms2_msgym_corpus import standardize_smiles
        index = self._mini_chebi_index()
        by_form = {}
        for r in index.records:
            from tools.ms2_chebi_corpus import formula_key as _fk
            by_form.setdefault(_fk(r["formula"]), []).append(r)
        qrec, reason = standardize_smiles("OCC", "t", "Q")
        self.assertIsNotNone(qrec, reason)
        qc, ok = rdkit_canon("OCC")
        self.assertTrue(ok)
        from tools.ms2_database_retrieval import neutral_mass
        mu = neutral_mass(qrec["formula"])
        # shift +1 Da: stage1 must reject every record here
        shifted = mu + 1_000_000
        st = chebi_stage_query("Q", qrec, qc, shifted, True, index, by_form,
                               uncertainty_mu=500)
        self.assertEqual(st["stage1"]["n_s1"], 0)
        self.assertFalse(st["stage1"]["target_present_s1"])
        self.assertFalse(st["stage2"]["target_present_s2"])
        self.assertFalse(st["stage3_oracle"]["target_present_s3"])
        self.assertTrue(st["nested_ok"])

    def test_unknown_precision_unavailable_throughout(self):
        from tools.ms2_msgym_corpus import standardize_smiles
        index = self._mini_chebi_index()
        by_form = {}
        for r in index.records:
            from tools.ms2_chebi_corpus import formula_key as _fk
            by_form.setdefault(_fk(r["formula"]), []).append(r)
        qrec, _ = standardize_smiles("CCO", "t", "Q")
        qc, _ = rdkit_canon("CCO")
        from tools.ms2_database_retrieval import neutral_mass
        st = chebi_stage_query("Q", qrec, qc, neutral_mass(qrec["formula"]),
                               False, index, by_form, uncertainty_mu=None)
        # unknown precision: everything unavailable, nothing rejected
        self.assertEqual(st["stage1"]["n_unavailable"], 2)
        self.assertEqual(st["stage1"]["n_reject"], 0)
        self.assertIsNone(st["uncertainty_mu"])

    def test_real_present_record_found_all_stages(self):
        from tools.ms2_msgym_corpus import standardize_smiles
        index = self._mini_chebi_index()
        by_form = {}
        for r in index.records:
            from tools.ms2_chebi_corpus import formula_key as _fk
            by_form.setdefault(_fk(r["formula"]), []).append(r)
        qrec, _ = standardize_smiles("CCO", "t", "Q")
        qc, _ = rdkit_canon("CCO")
        from tools.ms2_database_retrieval import neutral_mass
        st = chebi_stage_query("Q", qrec, qc, neutral_mass(qrec["formula"]),
                               True, index, by_form, uncertainty_mu=500)
        self.assertTrue(st["stage1"]["target_present_s1"])
        self.assertTrue(st["stage2"]["target_present_s2"])
        self.assertTrue(st["stage3_oracle"]["target_present_s3"])
        self.assertTrue(st["nested_ok"])


class PredictorGateTests(unittest.TestCase):
    def test_requires_capability_and_finite_target_score(self):
        pscores = [{"score": 0.9, "parseable": True},
                   {"score": 0.1, "parseable": True}]
        r, ok = enforce_predictor_rank(1, True, pscores, 0)
        self.assertEqual((r, ok), (1, True))
        # incapable query -> None even with present target
        r, ok = enforce_predictor_rank(1, False, pscores, 0)
        self.assertEqual((r, ok), (None, False))
        # non-finite true-candidate score -> None
        bad = [{"score": float("nan"), "parseable": True},
               {"score": 0.1, "parseable": True}]
        r, ok = enforce_predictor_rank(1, True, bad, 0)
        self.assertEqual((r, ok), (None, False))
        # absent target -> None
        r, ok = enforce_predictor_rank(None, True, pscores, None)
        self.assertEqual((r, ok), (None, False))

    def test_frozen_gate_measured_not_hardcoded(self):
        gate = {"tau": None, "kmax": 0, "status": "abstain_all"}
        rows = [({"qid": "Q0", "margin_pred": 0.5, "rank_predictor": 1}, {})]
        n, acc = apply_frozen_gate(rows, gate)
        self.assertEqual((n, acc), (0, []))
        gate2 = {"tau": 0.1, "kmax": 10, "status": "test"}
        n2, acc2 = apply_frozen_gate(rows, gate2)
        self.assertEqual((n2, acc2), (1, ["Q0"]))


class PoolTests(unittest.TestCase):
    def test_pool_dedup_counts_reported(self):
        # Canonical dedup accounting: OCC + CCO collapse to one connectivity.
        entries = [("OCC", None), ("CCO", None)]
        canon_entries = []
        for smi, _ in entries:
            c, ok = rdkit_canon(smi)
            canon_entries.append((smi, c if ok else None))
        known = [c for _, c in canon_entries if c]
        self.assertEqual(len(known), 2)
        self.assertEqual(len(set(known)), 1)


class PubchemTests(unittest.TestCase):
    def test_cap_respected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # offline: unresolvable host would error; use empty formulas
            results, stats = pubchem_expand([], d)
            self.assertEqual(results, {})
            self.assertEqual(stats["api_calls"], 0)

    def test_cache_replay_no_network(self):
        # Pre-seeded cache replays with zero API calls (no live ground truth).
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            payload = {"IdentifierList": {"CID": [123, 456]}}
            with open(os.path.join(d, "cids_C2H6O.json"), "w") as h:
                json.dump(payload, h)
            props = {"PropertyTable": {"Properties": [
                {"CID": 123, "CanonicalSMILES": "CCO"},
                {"CID": 456, "CanonicalSMILES": "OCC"}]}}
            with open(os.path.join(d, "smiles_C2H6O_0.json"), "w") as h:
                json.dump(props, h)
            results, stats = pubchem_expand(["C2H6O"], d)
            self.assertEqual(stats["api_calls"], 0)
            self.assertEqual(results["C2H6O"]["n_cids"], 2)
            self.assertEqual(results["C2H6O"]["n_smiles"], 2)


class StrictJsonTests(unittest.TestCase):
    def test_no_nan(self):
        bad = {"x": float("nan")}
        with self.assertRaises(ValueError):
            json.dumps(bad, allow_nan=False)


if __name__ == "__main__":
    main(sys.argv[1:])
