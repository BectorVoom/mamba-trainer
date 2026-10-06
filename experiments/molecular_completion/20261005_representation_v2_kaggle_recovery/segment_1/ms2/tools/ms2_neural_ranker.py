"""T4 neural ranking pilot; no stability, Mamba-completion, or FPNet claims.

Reuses the frozen Ridge experiment's selection and leakage exclusions. Graph
encoding has the existing restricted neutral typed-graph domain. Missing graph
scores stay in a seeded tail and every selected query stays in the denominator.
"""
import argparse
import copy
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ms2_fp_predictor import collect_train_molecules, select_val
from tools.ms2_msgym_corpus import standardize_smiles
from tools.ms2_spectral_rank import content_fp, parse_spectrum, seeded_positions, sha256_file, short_key

ELEMENTS = ("C", "N", "O", "F", "Cl", "Br", "I", "S")
N_BINS = 2000
N_BONDS = 4  # single, double, triple, and the shared parser's aromatic marker


def spectrum_features(peaks, precursor):
    """FPNet-style filter/cap/sqrt, then fixed 1-Da bins (not FPNet itself)."""
    if not math.isfinite(precursor) or precursor <= 0:
        raise ValueError("precursor_mz must be finite and positive")
    if any(not math.isfinite(m) or not math.isfinite(i) or i < 0 for m, i in peaks):
        raise ValueError("peaks must be finite with nonnegative intensities")
    kept = [(m, i, j) for j, (m, i) in enumerate(peaks) if 0 < m <= precursor + 2]
    maximum = max((i for _, i, _ in kept), default=0)
    result = np.zeros(N_BINS, dtype=np.float32)
    if maximum <= 0:
        return result
    kept = [(m, i / maximum, j) for m, i, j in kept if i / maximum >= 1e-3]
    for m, i, j in sorted(kept, key=lambda p: (-p[1], p[2]))[:160]:
        if m < N_BINS:
            result[int(m)] = max(result[int(m)], math.sqrt(i))
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def metadata(row, adduct_ids):
    precursor = float(row["precursor_mz"])
    adduct = row["adduct"]
    polarity = 1 if adduct.endswith("+") else -1 if adduct.endswith("-") else 0
    # Do not interpret percentages or normalized CE as eV.
    energy_text = row.get("collision_energy", "").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:eV)?", energy_text)
    energy = float(match.group(1)) if match else 0.0
    count_text = row.get("energy_count", "").strip()
    count = int(count_text) if count_text else 0
    if not 0 <= count <= 8:
        raise ValueError("energy_count must be in [0, 8]")
    values = [precursor / 2000, polarity, energy / 100, float(match is not None), count / 8]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("metadata must be finite")
    return np.asarray(values, dtype=np.float32), adduct_ids.get(adduct, 0)


def graph(smiles):
    record, reason = standardize_smiles(smiles, "neural-pilot", smiles)
    if record is None:
        return None
    atoms = record["atom_types"]
    x = np.zeros((len(atoms), len(ELEMENTS) + 2), dtype=np.float32)
    for index, (element, hydrogens, valence) in enumerate(atoms):
        x[index, ELEMENTS.index(element)] = 1
        x[index, -2:] = hydrogens / 4, valence / 4
    adjacency = np.zeros((N_BONDS, len(atoms), len(atoms)), dtype=np.float32)
    for a, b, order in record["edges"]:
        adjacency[order - 1, a, b] = adjacency[order - 1, b, a] = 1
    return x, adjacency


def pack_graphs(graphs, device):
    if not graphs or any(g is None for g in graphs):
        raise ValueError("pack_graphs requires supported, nonempty graphs")
    n = max(len(g[0]) for g in graphs)
    x = np.zeros((len(graphs), n, len(ELEMENTS) + 2), dtype=np.float32)
    a = np.zeros((len(graphs), N_BONDS, n, n), dtype=np.float32)
    mask = np.zeros((len(graphs), n, 1), dtype=np.float32)
    for j, (nodes, bonds) in enumerate(graphs):
        k = len(nodes)
        x[j, :k], a[j, :, :k, :k], mask[j, :k] = nodes, bonds, 1
    return tuple(torch.as_tensor(v, device=device) for v in (x, a, mask))


class Ranker(nn.Module):
    def __init__(self, n_adducts, width=128):
        super().__init__()
        self.adduct = nn.Embedding(n_adducts, 16)
        self.spectrum = nn.Sequential(nn.Linear(N_BINS + 21, width), nn.GELU(), nn.Linear(width, width))
        self.atom = nn.Linear(len(ELEMENTS) + 2, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.bond_layers = nn.ModuleList(nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(N_BONDS)) for _ in range(3))
        self.pool = nn.Linear(width * 2, width)
        self.forward_spectrum = nn.Sequential(nn.Linear(width + 21, width), nn.GELU(), nn.Linear(width, N_BINS), nn.Softplus())

    def encode_graph(self, packed):
        nodes, bonds, mask = packed
        h = F.gelu(self.atom(nodes)) * mask
        for own, messages in zip(self.self_layers, self.bond_layers):
            total = sum(bonds[:, b] @ layer(h) for b, layer in enumerate(messages))
            h = F.gelu(own(h) + total) * mask
        summed = h.sum(1)
        return self.pool(torch.cat([summed / mask.sum(1).clamp_min(1), summed], dim=-1))

    def context(self, meta, ids):
        return torch.cat([meta, self.adduct(ids)], -1)

    def encode_spectrum(self, spec, meta, ids):
        return F.normalize(self.spectrum(torch.cat([spec, self.context(meta, ids)], -1)), dim=-1)

    def predict_spectrum(self, graphs, meta, ids):
        return F.normalize(self.forward_spectrum(torch.cat([graphs, self.context(meta, ids)], -1)), dim=-1)


def tensors(molecules, ids, device):
    features, metas, adducts = [], [], []
    for m in molecules:
        meta, aid = metadata(m["row"], ids)
        features.append(spectrum_features(m["peaks"], float(m["row"]["precursor_mz"])))
        metas.append(meta)
        adducts.append(aid)
    return (torch.as_tensor(np.stack(features), device=device),
            torch.as_tensor(np.stack(metas), device=device),
            torch.as_tensor(adducts, device=device))


def loss(model, batch, ids, device, auxiliary):
    spec, meta, aid = tensors(batch, ids, device)
    g = model.encode_graph(pack_graphs([graph(m["smiles"]) for m in batch], device))
    s = model.encode_spectrum(spec, meta, aid)
    logits = s @ F.normalize(g, dim=-1).T / 0.1
    # Same connectivity variants must not be false negatives.
    positive = torch.tensor([[a["group"] == b["group"] for b in batch] for a in batch], device=device)
    target = positive.float() / positive.sum(1, keepdim=True)
    contrast = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
    forward = (1 - (model.predict_spectrum(g, meta, aid) * spec).sum(-1)).mean()
    return contrast + auxiliary * forward


def train(fit, calibration, ids, device, args, auxiliary):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Ranker(len(ids) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best, best_loss, history = None, math.inf, []
    for epoch in range(args.epochs):
        model.train()
        order = list(fit)
        random.shuffle(order)
        values = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            value = loss(model, batch, ids, device, auxiliary)
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite training loss")
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            values.append(value.item())
        model.eval()
        with torch.no_grad():
            chunks = [calibration[i:i + args.batch_size] for i in range(0, len(calibration), args.batch_size)]
            valid = [loss(model, b, ids, device, auxiliary).item() for b in chunks if len(b) >= 2]
        if not values or not valid or not all(math.isfinite(v) for v in valid):
            raise ValueError("need at least two fit and calibration molecules and finite losses")
        metric = float(np.mean(valid))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(values)), "calibration_loss": metric})
        print(json.dumps(history[-1]), flush=True)
        if metric < best_loss:
            best, best_loss = copy.deepcopy(model.state_dict()), metric
    model.load_state_dict(best)
    model.eval()
    return model, history


def order_scores(scores, positions):
    return sorted(range(len(scores)), key=lambda i: (scores[i] is None, -(scores[i] or 0), positions[i]))


@torch.no_grad()
def evaluate(model, pools, ids, device, batch_size):
    rows = []
    for qid, query in pools.items():
        pool = query["distinct"]
        positions = seeded_positions(qid, len(pool))
        peaks, error = parse_spectrum(query["row"]["mzs"], query["row"]["intensities"])
        scores = [None] * len(pool)
        if error is None:
            batch = [{"row": query["row"], "peaks": peaks}]
            spec, meta, aid = tensors(batch, ids, device)
            if spec.norm() > 0:
                embedded = model.encode_spectrum(spec, meta, aid)
                supported = [(i, graph(s)) for i, s in enumerate(pool)]
                supported = [(i, g) for i, g in supported if g is not None]
                for start in range(0, len(supported), batch_size):
                    chunk = supported[start:start + batch_size]
                    g = F.normalize(model.encode_graph(pack_graphs([g for _, g in chunk], device)), dim=-1)
                    values = (embedded @ g.T).flatten().cpu().tolist()
                    for (index, _), score in zip(chunk, values):
                        if not math.isfinite(score):
                            raise RuntimeError("nonfinite candidate score")
                        scores[index] = score
        order = order_scores(scores, positions)
        target = pool.index(query["smiles"]) if query["smiles"] in pool else None
        uniform = sorted(range(len(pool)), key=lambda i: positions[i])
        rows.append({"qid": qid, "n_pool": len(pool), "n_scored": sum(s is not None for s in scores),
                     "rank": order.index(target) + 1 if target is not None else None,
                     "uniform_rank": uniform.index(target) + 1 if target is not None else None})
    return rows


def bootstrap_difference(a, b, seed, repeats=2000):
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap needs equal nonempty samples")
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repeats)]
    return {"difference": float(delta.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable: select a Colab GPU runtime")
        name = torch.cuda.get_device_name()
        if args.require_t4 and "T4" not in name:
            raise RuntimeError(f"Requested T4, allocated {name}")
    elif args.require_t4:
        raise ValueError("--require-t4 requires --device cuda")
    selected = select_val(args.tsv, args.prefix, args.max_queries)
    val_rows = selected["val_groups"]
    val_keys = {short_key(r["inchikey"]) for r in val_rows.values() if r["inchikey"]}
    contents = set()
    for row in val_rows.values():
        peaks, error = parse_spectrum(row["mzs"], row["intensities"])
        if error is None:
            contents.add(content_fp(peaks))
    _, fit, calibration, _, _, audit = collect_train_molecules(args.tsv, args.max_train, set(val_rows), val_keys, contents)
    # Missing/empty spectrum evidence cannot be a training target.
    def usable(m):
        return spectrum_features(m["peaks"], float(m["row"]["precursor_mz"])).any()
    n_before = [len(fit), len(calibration)]
    fit, calibration = [m for m in fit if usable(m)], [m for m in calibration if usable(m)]
    ids = {adduct: i + 1 for i, adduct in enumerate(sorted({m["row"]["adduct"] for m in fit}))}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, histories = {}, {}
    for arm, auxiliary in (("contrastive", 0.0), ("contrastive_forward", 0.2)):
        print(f"Training {arm} on {device}", flush=True)
        model, histories[arm] = train(fit, calibration, ids, device, args, auxiliary)
        torch.save({"model": model.cpu().state_dict(), "adduct_ids": ids, "args": vars(args),
                    "scope": "restricted-domain neural retrieval pilot"}, output / f"{arm}.pt")
        model.to(device)
        all_rows[arm] = evaluate(model, selected["pools"], ids, device, args.batch_size)
    metrics = {}
    for arm, rows in all_rows.items():
        metrics[arm] = {f"top{k}": sum(r["rank"] is not None and r["rank"] <= k for r in rows) / len(rows) for k in (1, 10, 25)}
        metrics[arm]["unrankable"] = sum(r["n_scored"] == 0 for r in rows)
    rows = all_rows["contrastive"]
    metrics["uniform"] = {f"top{k}": sum(r["uniform_rank"] is not None and r["uniform_rank"] <= k for r in rows) / len(rows) for k in (1, 10, 25)}
    hits = lambda arm: [r["rank"] is not None and r["rank"] <= 25 for r in all_rows[arm]]
    summary = {"scope": "neural retrieval pilot; no stability validation or completion training",
               "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
               "torch": torch.__version__, "config": vars(args), "fit": len(fit), "calibration": len(calibration),
               "empty_exclusions": [n_before[0] - len(fit), n_before[1] - len(calibration)],
               "audit": dict(audit), "metrics": metrics,
               "forward_minus_contrastive_top25": bootstrap_difference(hits("contrastive_forward"), hits("contrastive"), args.seed),
               "contrastive_minus_uniform_top25": bootstrap_difference(hits("contrastive"),
                   [r["uniform_rank"] is not None and r["uniform_rank"] <= 25 for r in rows], args.seed),
               "data_sha256": {"tsv": sha256_file(args.tsv), "prefix": sha256_file(args.prefix)},
               "script_sha256": sha256_file(__file__), "histories": histories,
               "limitations": ["no predicted-substructure inputs or learned molecular prior",
                               "no Mamba decoder or de novo generation",
                               "connectivity-group disjointness; not scaffold disjointness",
                               "restricted typed parser excludes unsupported candidates into seeded tails",
                               "graph encoder does not represent stereochemistry",
                               "fixed 1-Da binned FPNet-style preprocessing; not pretrained FPNet",
                               "energy_count unavailable in benchmark, represented as unknown=0",
                               "previously inspected validation slice; not an untouched test result",
                               "in-batch training negatives; same-formula negatives not implemented",
                               "no stability labels, calibration, or physical stability claims"]}
    (output / "predictions.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="data/pinned/MassSpecGym1.5.tsv")
    parser.add_argument("--prefix", default="data/pinned/msgym_candidates_formula_prefix64.json")
    parser.add_argument("--out", default="experiments/molecular_completion/neural_t4")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-t4", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.max_train, args.max_queries) < 1 or args.batch_size < 2:
        parser.error("epochs/train/query caps must be positive; batch-size must be >=2")
    run(args)
