"""Train and evaluate every neural component of the declared CHNO proposal."""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ms2_full_model import *
from tools.ms2_neural_ranker import metadata, bootstrap_difference
from tools.ms2_spectral_rank import parse_spectrum, content_fp, sha256_file, seeded_positions
from tools.ms2_msgym_corpus import iter_json_entries


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def make_record(row):
    peaks, reason = parse_spectrum(row["mzs"], row["intensities"])
    if reason:
        return None, "peaks:" + reason
    try:
        precursor = float(row["precursor_mz"])
        features, mask, bins = prep_peaks(peaks, precursor)
        meta, _ = metadata(row, ADDUCT_IDS)
        neutral_mass(row)
    except (ValueError, ArithmeticError):
        return None, "metadata"
    if not mask.any():
        return None, "empty_peaks"
    graph, reason = parse_graph(row["smiles"])
    record = {"row": row, "peaks": features, "mask": mask, "bins": bins,
              "meta": meta, "adduct": ADDUCT_IDS[row["adduct"]], "graph": graph,
              "identity": identity(row["smiles"]), "predicted": [],
              "content": content_fp(peaks)}
    return record, reason


def load_data(args, table):
    train, val, test = {}, {}, {}
    with open(args.tsv) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            bucket = {"train": train, "val": val, "test": test}.get(row["fold"])
            if bucket is not None and row["smiles"] not in bucket:
                if bucket is not train or len(train) < args.max_train * 2:
                    bucket[row["smiles"]] = row
    queries = []
    for smi, candidates in iter_json_entries(args.prefix, 1 << 30):
        if smi in test and len(queries) < args.queries:
            queries.append({"row": test[smi], "pool": list(dict.fromkeys(candidates)), "qid": f"TEST-{len(queries):04d}"})
    if not queries:
        raise ValueError("no test queries join the pinned prefix")
    # Use the benchmark val fold only for checkpoint selection and calibration.
    calibration_rows = list(val.values())[:args.calibration]
    reserved_ids = {identity(q["row"]["smiles"]) for q in queries}
    reserved_ids.update(identity(r["smiles"]) for r in calibration_rows)
    reserved_scaffolds = {scaffold(r["smiles"]) for r in calibration_rows}
    reserved_scaffolds.update(scaffold(q["row"]["smiles"]) for q in queries)
    reserved_content = set()
    for r in calibration_rows + [q["row"] for q in queries]:
        peaks, error = parse_spectrum(r["mzs"], r["intensities"])
        if error is None:
            reserved_content.add(content_fp(peaks))
    audit, fit, calibration, seen_ids, seen_content = Counter(), [], [], set(), set()
    for row in train.values():
        if len(fit) >= args.max_train:
            break
        key = identity(row["smiles"])
        group = scaffold(row["smiles"])
        if key in reserved_ids or group in reserved_scaffolds:
            audit["train_heldout_identity_or_scaffold"] += 1
            continue
        r, reason = make_record(row)
        if r is None or reason:
            audit["train:" + (reason or "unknown")] += 1
            continue
        if r["content"] in reserved_content or r["content"] in seen_content or key in seen_ids:
            audit["train_duplicate"] += 1
            continue
        hypotheses = table.hypotheses(row)
        if r["graph"].counts not in hypotheses:
            audit["train_formula_outside_mass_or_table"] += 1
            continue
        r.update(scaffold=group, formula_hypotheses=hypotheses)
        teacher_batch([r["graph"]], "cpu")
        fit.append(r)
        seen_ids.add(key)
        seen_content.add(r["content"])
    for row in calibration_rows:
        r, reason = make_record(row)
        if r is None or reason:
            audit["calibration:" + (reason or "unknown")] += 1
            continue
        hypotheses = table.hypotheses(row)
        if r["graph"].counts not in hypotheses:
            audit["calibration_formula_outside_mass_or_table"] += 1
            continue
        r.update(scaffold=scaffold(row["smiles"]), formula_hypotheses=hypotheses)
        teacher_batch([r["graph"]], "cpu")
        calibration.append(r)
    fit_scaffolds = {r["scaffold"] for r in fit}
    assert not fit_scaffolds.intersection(reserved_scaffolds)
    assert not {r["identity"] for r in fit}.intersection(reserved_ids)
    assert not {r["content"] for r in fit}.intersection(reserved_content)
    return fit, calibration, queries, dict(audit), reserved_ids, reserved_scaffolds


def fit_loop(model, fit, calibration, objective, epochs, batch_size, out, name, seed):
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    history, best, minimum = [], None, math.inf
    started = time.monotonic()
    emit({"stage": name, "event": "start", "fit": len(fit), "calibration": len(calibration), "epochs": epochs})
    for epoch in range(epochs):
        model.train()
        order = list(fit)
        random.shuffle(order)
        train_values = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            value = objective(model, batch)
            if not torch.isfinite(value):
                raise RuntimeError(f"{name}: nonfinite loss")
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_values.append(value.item())
            if len(train_values) % 64 == 0:
                emit({"stage": name, "epoch": epoch + 1, "batch": len(train_values),
                      "train_running": float(np.mean(train_values))})
        model.eval()
        with torch.no_grad():
            valid = [objective(model, calibration[i:i + batch_size]).item() for i in range(0, len(calibration), batch_size)]
        if not train_values or not valid or not all(math.isfinite(v) for v in valid):
            raise ValueError(f"{name}: empty split or nonfinite calibration loss")
        metric = float(np.mean(valid))
        item = {"stage": name, "epoch": epoch + 1, "train": float(np.mean(train_values)), "calibration": metric}
        history.append(item)
        emit(item)
        if metric < minimum:
            minimum, best = metric, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    torch.save({"state": model.cpu().state_dict(), "stage": name, "domain": DOMAIN}, out / f"{name}.pt")
    model.to(DEVICE)
    (out / f"{name}_history.json").write_text(json.dumps({"history": history, "seconds": time.monotonic() - started}, indent=2))
    return model


def upstream_objective(vocab):
    def objective(model, records):
        present = [motifs(r["graph"]) for r in records]
        targets = torch.tensor([[key in keys for key in vocab] for keys in present], device=DEVICE, dtype=torch.float32)
        return F.binary_cross_entropy_with_logits(model(batch_inputs(records, DEVICE)), targets)
    return objective


@torch.no_grad()
def predict_substructures(model, records, vocab, batch_size):
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        probabilities = model(batch_inputs(batch, DEVICE)).sigmoid().cpu().numpy()
        for record, probability in zip(batch, probabilities):
            selected = np.argsort(-probability)[:6]
            record["predicted"] = [(vocab[i], float(probability[i])) for i in selected if probability[i] >= 0.2]


def train_upstream(fit, calibration, args, out):
    folds = {r["scaffold"]: int(hashlib.sha256(r["scaffold"].encode()).hexdigest()[:8], 16) % 3 for r in fit}
    audits = []
    for fold in range(3):
        training = [r for r in fit if folds[r["scaffold"]] != fold]
        validation = [r for r in fit if folds[r["scaffold"]] == fold]
        # Vocabulary is fold-train-only too, not mined from the predicted targets.
        vocab = vocabulary(training)
        model = SubstructurePredictor(len(vocab)).to(DEVICE)
        fit_loop(model, training, calibration, upstream_objective(vocab), args.upstream_epochs, args.batch_size,
                 out, f"upstream_fold{fold}", args.seed + fold)
        predict_substructures(model, validation, vocab, args.batch_size)
        audits.append({"fold": fold, "train": len(training), "predicted": len(validation),
                       "scaffold_overlap": len({r["scaffold"] for r in training} & {r["scaffold"] for r in validation})})
        torch.save({"patterns": vocab, "domain": DOMAIN}, out / f"upstream_fold{fold}_vocab.pt")
    vocab = vocabulary(fit)
    model = SubstructurePredictor(len(vocab)).to(DEVICE)
    fit_loop(model, fit, calibration, upstream_objective(vocab), args.upstream_epochs,
             args.batch_size, out, "upstream_final", args.seed)
    predict_substructures(model, calibration, vocab, args.batch_size)
    torch.save({"patterns": vocab, "domain": DOMAIN}, out / "upstream_final_vocab.pt")
    (out / "upstream_crossfit_audit.json").write_text(json.dumps(audits, indent=2))
    return model, vocab


def completion_objective(conditional, auxiliary):
    def objective(model, records):
        graphs = [r["graph"] for r in records]
        likelihood = model.likelihood(graphs, records, DEVICE, conditional)
        result = -likelihood.mean()
        if conditional:
            context, _, _ = model.condition(records, DEVICE)
            query = model.formula_query(context)
            logits, targets = [], []
            maximum = max(len(r["formula_hypotheses"]) for r in records)
            for i, record in enumerate(records):
                counts = torch.tensor(record["formula_hypotheses"], device=DEVICE, dtype=torch.float32)
                values = (model.formula_embedding(counts) * query[i]).sum(-1) / math.sqrt(96)
                logits.append(F.pad(values, (0, maximum - len(values)), value=-1e4))
                targets.append(record["formula_hypotheses"].index(record["graph"].counts))
            result = result + F.cross_entropy(torch.stack(logits), torch.tensor(targets, device=DEVICE))
            encoded = model.graph_encoder(*graph_tensors(graphs, DEVICE))
            inputs = batch_inputs(records, DEVICE)
            prediction = model.predict_spectrum(encoded, inputs)
            result += auxiliary * (1 - (prediction * inputs[4]).sum(-1)).mean()
        return result
    return objective


def decoys(record, index, rng, forbidden, forbidden_scaffolds):
    graph = record["graph"]
    result = {g.identity: g for g in index.get(graph.counts, []) if g.identity != graph.identity}
    if len(result) < 4:
        bonds = list(graph.edges)
        for _ in range(60):
            if len(bonds) < 2 or len(result) >= 4:
                break
            first, second = rng.sample(bonds, 2)
            a, b, order = first
            c, d, other = second
            if order != other or len({a, b, c, d}) != 4:
                continue
            # Build from our graph ordering, not RDKit's post-kekulization order.
            editable = Chem.RWMol()
            for t in graph.types:
                atom = Chem.Atom(ATOM_TYPES[t][0])
                editable.AddAtom(atom)
            for x, y, value in graph.edges:
                editable.AddBond(x, y, (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[value - 1])
            editable.RemoveBond(a, b)
            editable.RemoveBond(c, d)
            new_edges = [(a, d), (b, c)] if rng.random() < 0.5 else [(a, c), (a, d)]
            if any(editable.GetBondBetweenAtoms(x, y) for x, y in new_edges):
                continue
            for x, y in new_edges:
                editable.AddBond(x, y, (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[order - 1])
            try:
                Chem.SanitizeMol(editable)
            except (Chem.AtomValenceException, Chem.KekulizeException):
                continue
            key = Chem.MolToSmiles(editable, canonical=True, isomericSmiles=False)
            if key in forbidden or key == graph.identity or scaffold(key) in forbidden_scaffolds:
                continue
            candidate, _ = parse_graph(key)
            if candidate is not None and candidate.counts == graph.counts:
                result[key] = candidate
    keys = sorted(result)
    rng.shuffle(keys)
    return [graph] + [result[k] for k in keys[:4]]


@torch.no_grad()
def ranking_features(model, prior, graphs, record):
    records = [record] * len(graphs)
    context, _, _ = model.condition(records, DEVICE)
    encoded = model.graph_encoder(*graph_tensors(graphs, DEVICE))
    dot = (F.normalize(context, dim=-1) * F.normalize(encoded, dim=-1)).sum(-1)
    conditional = model.likelihood(graphs, records, DEVICE)
    prior_score = prior.likelihood(graphs, records, DEVICE, False)
    inputs = batch_inputs(records, DEVICE)
    predicted_spectrum = model.predict_spectrum(encoded, inputs)
    similarity = (predicted_spectrum * inputs[4]).sum(-1)
    support = []
    for graph in graphs:
        present = motifs(graph)
        total = sum(confidence for _, confidence in record.get("predicted", []))
        support.append(sum(confidence for key, confidence in record.get("predicted", []) if key in present) / max(total, 1e-6))
    return torch.stack([dot, conditional, prior_score, similarity, torch.tensor(support, device=DEVICE)], -1).cpu().numpy()


class FeatureRanker(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.network = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.network((features - self.mean) / self.scale).squeeze(-1)


def train_ranker(model, prior, fit, calibration, forbidden, reserved_scaffolds, args, out, name):
    rng = random.Random(args.seed)
    index = defaultdict(list)
    for r in fit:
        index[r["graph"].counts].append(r["graph"])
    datasets = []
    audit = {}
    for split, records in (("fit", fit), ("calibration", calibration)):
        dataset = []
        for r in records:
            graphs = decoys(r, index, rng, forbidden, reserved_scaffolds)
            if len(graphs) < 2:
                continue
            if split == "fit" and any(g.identity in forbidden for g in graphs):
                raise RuntimeError("held-out identity in training ranker candidates")
            features = ranking_features(model, prior, graphs, r)
            dataset.append(features)
            if len(dataset) >= args.rank_cap:
                break
        audit[split] = {"rankable": len(dataset), "examined_cap": args.rank_cap}
        datasets.append(dataset)
        emit({"stage": name + "_features", "split": split, "rankable": len(dataset)})
    if not all(datasets):
        raise ValueError("no same-formula negatives: ranker cannot be verified")
    flattened = np.concatenate(datasets[0])
    ranker = FeatureRanker(flattened.mean(0), flattened.std(0).clip(1e-3)).to(DEVICE)
    # Independent fixed feature ablations test the reranking contributions.
    def objective(ranker, batch):
        maximum = max(len(x) for x in batch)
        logits = []
        for feature in batch:
            scores = ranker(torch.tensor(feature, device=DEVICE))
            logits.append(F.pad(scores, (0, maximum - len(scores)), value=-1e4))
        return F.cross_entropy(torch.stack(logits), torch.zeros(len(batch), dtype=torch.long, device=DEVICE))
    fit_loop(ranker, datasets[0], datasets[1], objective, args.rank_epochs, 32, out, name, args.seed)
    (out / f"{name}_negative_audit.json").write_text(json.dumps(audit, indent=2))
    return ranker


@torch.no_grad()
def formula_predictions(model, record, table):
    counts = table.hypotheses(record["row"])
    if not counts:
        return []
    context, _, _ = model.condition([record], DEVICE)
    query = model.formula_query(context)
    values = torch.tensor(counts, device=DEVICE, dtype=torch.float32)
    scores = (model.formula_embedding(values) * query).sum(-1)
    order = scores.argsort(descending=True).cpu().tolist()
    return [counts[i] for i in order[:4]]


@torch.no_grad()
def retrieval_prediction(model, prior, ranker, query, record):
    canonical_pool = canonical_candidates(query["pool"])
    pool = list(canonical_pool)
    scores = [None] * len(pool)
    if record is not None:
        supported = [(i, canonical_pool[s]) for i, s in enumerate(pool) if canonical_pool[s] is not None]
        for start in range(0, len(supported), 32):
            chunk = supported[start:start + 32]
            features = ranking_features(model, prior, [g for _, g in chunk], record)
            values = ranker(torch.tensor(features, device=DEVICE)).cpu().tolist()
            for (index, _), value in zip(chunk, values):
                scores[index] = value
    positions = seeded_positions(query["qid"], len(pool))
    order = sorted(range(len(pool)), key=lambda i: (scores[i] is None, -(scores[i] or 0), positions[i]))
    target_identity = identity(query["row"]["smiles"])
    target = pool.index(target_identity) if target_identity in pool else None
    uniform = sorted(range(len(pool)), key=lambda i: positions[i])
    return {"qid": query["qid"], "pool": len(pool), "scored": sum(v is not None for v in scores),
            "rank": order.index(target) + 1 if target is not None else None,
            "uniform_rank": uniform.index(target) + 1 if target is not None else None}


def evaluate_full(model, prior, ranker, upstream, vocab, queries, table, args, out):
    predictions, generation = [], []
    for query_index, query in enumerate(queries):
        record, reason = make_record(query["row"])
        row = {"qid": query["qid"], "status": "ok" if record is not None else reason, "generated": [], "rank": None,
               "formula_recovered": False, "valid_candidates": 0, "query_identity": identity(query["row"]["smiles"])}
        row["target_domain"] = reason or "supported"
        if record is not None:
            predict_substructures(upstream, [record], vocab, 1)
            formulas = formula_predictions(model, record, table)
            # Reference graph is used only for evaluation below, never generation.
            row["formula_recovered"] = record["graph"] is not None and record["graph"].counts in formulas
            generated, counters = sample_graphs(model, record, formulas, DEVICE, args.trajectories, args.seed + query_index)
            row["work"] = counters
            row["valid_candidates"] = len(generated)
            if generated:
                scored = []
                for start in range(0, len(generated), 32):
                    feature = ranking_features(model, prior, generated[start:start + 32], record)
                    with torch.no_grad():
                        scores = ranker(torch.tensor(feature, device=DEVICE)).cpu().tolist()
                    scored.extend(scores)
                ordered = sorted(zip(generated, scored), key=lambda x: (-x[1], x[0].identity))
                row["generated"] = [g.identity for g, _ in ordered[:25]]
                row["rank"] = next((i + 1 for i, (g, _) in enumerate(ordered) if g.identity == row["query_identity"]), None)
            else:
                row["status"] = "no_valid_generation"
            actual = motifs(record["graph"]) if record["graph"] is not None else set()
            row["predicted_patterns"] = [{"types": key[0], "edges": key[1], "confidence": confidence,
                                           "correct": key in actual} for key, confidence in record["predicted"]]
        generation.append(row)
        # Retrieval diagnostic evaluates the same full conditional model on supplied
        # pools; it cannot replace the primary de novo generation result.
        predictions.append(retrieval_prediction(model, prior, ranker, query, record))
        emit({"stage": "evaluation", "query": query_index + 1, "total": len(queries),
              "generation_status": row["status"], "valid": row["valid_candidates"],
              "generated_rank": row["rank"], "retrieval_rank": predictions[-1]["rank"]})
        # Persist progress so Colab interruptions never erase completed queries.
        (out / "generation_predictions.json").write_text(json.dumps(generation, indent=2))
        (out / "retrieval_predictions.json").write_text(json.dumps(predictions, indent=2))
    return generation, predictions


def main(args):
    global DEVICE
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and (not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name()):
        raise RuntimeError("Colab Tesla T4 required")
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(vars(args), indent=2))
    table = FormulaTable()
    fit, calibration, queries, audit, forbidden, reserved_scaffolds = load_data(args, table)
    emit({"stage": "data", "fit": len(fit), "calibration": len(calibration), "test": len(queries), "audit": audit})
    (out / "data_audit.json").write_text(json.dumps({"fit": len(fit), "calibration": len(calibration),
        "test": len(queries), "exclusions": audit, "source_sha256": {"tsv": sha256_file(args.tsv), "prefix": sha256_file(args.prefix)},
        "fit_identities": [r["identity"] for r in fit], "calibration_identities": [r["identity"] for r in calibration],
        "test_identities": [identity(q["row"]["smiles"]) for q in queries]}, indent=2))
    if len(fit) < 50 or len(calibration) < 10:
        raise ValueError("insufficient domain examples after disjointness exclusions")
    upstream, vocab = train_upstream(fit, calibration, args, out)
    prior = CompletionModel().to(DEVICE)
    fit_loop(prior, fit, calibration, completion_objective(False, 0), args.prior_epochs,
             args.batch_size, out, "molecular_prior", args.seed)
    model = CompletionModel().to(DEVICE)
    model.load_state_dict(prior.state_dict())
    model.spectrum_encoder.load_state_dict(upstream.encoder.state_dict())
    fit_loop(model, fit, calibration, completion_objective(True, 0.2), args.completion_epochs,
             args.batch_size, out, "conditional_completion", args.seed)
    ranker = train_ranker(model, prior, fit, calibration, forbidden, reserved_scaffolds, args, out, "same_formula_ranker")
    generation, retrieval = evaluate_full(model, prior, ranker, upstream, vocab, queries, table, args, out)
    hit = lambda row, k: row["rank"] is not None and row["rank"] <= k
    summary = {"scope": "full declared CHNO neural pipeline; physical stability not evaluated",
               "domain": DOMAIN, "config": vars(args), "fit": len(fit), "calibration": len(calibration),
               "test": len(queries), "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None,
               "parameters": sum(p.numel() for p in model.parameters()),
               "generation": {f"top{k}": sum(hit(r, k) for r in generation) / len(queries) for k in (1, 10, 25)},
               "retrieval_diagnostic": {f"top{k}": sum(hit(r, k) for r in retrieval) / len(queries) for k in (1, 10, 25)},
               "formula_top4_recovery": sum(r["formula_recovered"] for r in generation) / len(queries),
               "generation_status": dict(Counter(r["status"] for r in generation)),
               "mean_valid_candidates": float(np.mean([r["valid_candidates"] for r in generation])),
               "limitations": ["physical stability labels absent", "single seed", "neutral CHNO domain",
                               "unknown energy-count metadata", "connectivity identity excludes stereochemistry",
                               "PyTorch Mamba-3 reference, not native Rust CUDA training"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="data/pinned/MassSpecGym1.5.tsv")
    parser.add_argument("--prefix", default="data/pinned/msgym_candidates_formula_prefix64.json")
    parser.add_argument("--out", default="experiments/molecular_completion/full_t4")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--calibration", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--upstream-epochs", type=int, default=8)
    parser.add_argument("--prior-epochs", type=int, default=10)
    parser.add_argument("--completion-epochs", type=int, default=15)
    parser.add_argument("--rank-epochs", type=int, default=30)
    parser.add_argument("--rank-cap", type=int, default=1500)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
