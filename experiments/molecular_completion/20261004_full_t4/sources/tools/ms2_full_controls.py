"""Matched decoder and substructure controls for the complete T4 pipeline.

The Transformer control replaces only the graph decoder. The spectrum encoder,
training split, formula inference, molecular-prior pretraining, losses, ranking,
and generation budget remain the same. Neither control evaluates physical
stability. Run only after the full pipeline has completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.ms2_full_experiment as experiment
from tools.ms2_full_model import CompletionModel, FormulaTable, SubstructurePredictor
from tools.ms2_neural_ranker import bootstrap_difference


class CausalTransformerBlock(nn.Module):
    """One causal layer matched to two width-96 Mamba-3 decoder blocks."""
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(96, 4, 166, dropout=0,
            activation="gelu", batch_first=True, norm_first=True)

    def forward(self, x):
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        return self.layer(x, src_mask=mask)


class EmptyPredictor(nn.Module):
    def forward(self, batch):
        return batch[0].new_zeros(batch[0].shape[0], 0)


def completion_model(transformer=False):
    model = CompletionModel()
    if transformer:
        model.layers = nn.ModuleList([CausalTransformerBlock()])
    return model.to(experiment.DEVICE)


def restore(model, path):
    model.load_state_dict(torch.load(path, map_location=experiment.DEVICE, weights_only=True)["state"])
    return model.eval()


def load_upstream(base, fit, calibration, batch_size):
    # Reconstruct exactly the out-of-fold graph predictions used in completion
    # training; the final predictor is reserved for the external validation/test.
    for fold in range(3):
        vocab = torch.load(base / f"upstream_fold{fold}_vocab.pt", weights_only=True)["patterns"]
        model = restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE), base / f"upstream_fold{fold}.pt")
        records = [r for r in fit if int(hashlib.sha256(r["scaffold"].encode()).hexdigest()[:8], 16) % 3 == fold]
        experiment.predict_substructures(model, records, vocab, batch_size)
    vocab = torch.load(base / "upstream_final_vocab.pt", weights_only=True)["patterns"]
    model = restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE), base / "upstream_final.pt")
    experiment.predict_substructures(model, calibration, vocab, batch_size)
    return model, vocab


def metrics(generation, retrieval):
    def hit(row, k):
        return row["rank"] is not None and row["rank"] <= k
    return {"generation": {f"top{k}": sum(hit(r, k) for r in generation) / len(generation) for k in (1, 10, 25)},
            "retrieval": {f"top{k}": sum(hit(r, k) for r in retrieval) / len(retrieval) for k in (1, 10, 25)}}


def main(base):
    args = argparse.Namespace(**json.loads((base / "plan.json").read_text()))
    experiment.DEVICE = torch.device(args.device)
    if args.device != "cuda" or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("T4 required for matched controls")
    original = json.loads((base / "summary.json").read_text())
    table = FormulaTable()
    fit, calibration, queries, audit, forbidden, reserved = experiment.load_data(args, table)
    recorded = json.loads((base / "data_audit.json").read_text())
    if [r["identity"] for r in fit] != recorded["fit_identities"] or len(queries) != original["test"]:
        raise RuntimeError("control split differs from the full pipeline")
    upstream, vocab = load_upstream(base, fit, calibration, args.batch_size)
    results = {"full": {"generation": original["generation"], "retrieval": original["retrieval_diagnostic"],
                        "parameters": original["parameters"]}}
    full_generation = json.loads((base / "generation_predictions.json").read_text())
    full_retrieval = json.loads((base / "retrieval_predictions.json").read_text())
    controls = base / "controls"
    controls.mkdir(exist_ok=True)
    (controls / "plan.json").write_text(json.dumps({"arms": ["transformer_decoder", "no_substructures"],
        "seed": args.seed, "epochs": {"prior": args.prior_epochs, "conditional": args.completion_epochs,
        "ranker": args.rank_epochs}, "queries": len(queries), "trajectories_per_formula": args.trajectories,
        "transformer": {"layers": 1, "width": 96, "heads": 4, "feedforward": 166},
        "scope": "decoder and substructure ablations; physical stability not assessed"}, indent=2))
    for arm in ("transformer_decoder", "no_substructures"):
        experiment.seed_all(args.seed)
        out = controls / arm
        out.mkdir(exist_ok=True)
        arm_fit = [{**r, "predicted": [] if arm == "no_substructures" else r["predicted"]} for r in fit]
        arm_cal = [{**r, "predicted": [] if arm == "no_substructures" else r["predicted"]} for r in calibration]
        if arm == "transformer_decoder":
            prior = completion_model(True)
            experiment.fit_loop(prior, arm_fit, arm_cal, experiment.completion_objective(False, 0),
                args.prior_epochs, args.batch_size, out, "molecular_prior", args.seed)
        else:
            prior = restore(completion_model(), base / "molecular_prior.pt")
        model = completion_model(arm == "transformer_decoder")
        model.load_state_dict(prior.state_dict())
        model.spectrum_encoder.load_state_dict(upstream.encoder.state_dict())
        experiment.fit_loop(model, arm_fit, arm_cal, experiment.completion_objective(True, 0.2),
            args.completion_epochs, args.batch_size, out, "conditional_completion", args.seed)
        ranker = experiment.train_ranker(model, prior, arm_fit, arm_cal, forbidden, reserved, args, out, "same_formula_ranker")
        predictor, patterns = (EmptyPredictor().to(experiment.DEVICE), []) if arm == "no_substructures" else (upstream, vocab)
        generation, retrieval = experiment.evaluate_full(model, prior, ranker, predictor, patterns, queries, table, args, out)
        result = metrics(generation, retrieval)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        if arm == "transformer_decoder" and abs(result["parameters"] / original["parameters"] - 1) > 0.01:
            raise RuntimeError("Transformer control parameter mismatch exceeds 1%")
        hits = lambda rows: [r["rank"] is not None and r["rank"] <= 25 for r in rows]
        result["generation_full_minus_control"] = bootstrap_difference(hits(full_generation), hits(generation), args.seed)
        result["retrieval_full_minus_control"] = bootstrap_difference(hits(full_retrieval), hits(retrieval), args.seed)
        results[arm] = result
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        (controls / "summary.json").write_text(json.dumps(results, indent=2))
        experiment.emit({"stage": "control_complete", "arm": arm, **result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("experiments/molecular_completion/full_t4"))
    main(parser.parse_args().base)
