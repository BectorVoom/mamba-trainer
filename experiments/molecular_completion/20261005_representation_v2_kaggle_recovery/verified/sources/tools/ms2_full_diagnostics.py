"""Independent component and generated-candidate checks for the full T4 run."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.ms2_full_experiment as experiment
from tools.ms2_full_model import (CompletionModel, FormulaTable, SubstructurePredictor,
    batch_inputs, graph_tensors, motifs, parse_graph)
from tools.ms2_full_controls import restore
from tools.ms2_neural_ranker import bootstrap_difference


def main(base):
    args = argparse.Namespace(**json.loads((base / 'plan.json').read_text()))
    experiment.DEVICE = torch.device(args.device)
    table = FormulaTable()
    fit, calibration, queries, _, _, _ = experiment.load_data(args, table)
    model = restore(CompletionModel().to(experiment.DEVICE), base / 'conditional_completion.pt')
    prior = restore(CompletionModel().to(experiment.DEVICE), base / 'molecular_prior.pt')
    vocab = torch.load(base / 'upstream_final_vocab.pt', weights_only=True)['patterns']
    upstream = restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE), base / 'upstream_final.pt')
    template = np.mean(np.stack([r['bins'] for r in fit]), axis=0)
    template /= max(np.linalg.norm(template), 1e-12)
    nll_prior, nll_full, nll_without_sub, forward, baseline, motif_scores = [], [], [], [], [], []
    reasons = Counter()
    for query in queries:
        record, reason = experiment.make_record(query['row'])
        reasons[reason or 'supported'] += 1
        if record is None or record['graph'] is None:
            continue
        experiment.predict_substructures(upstream, [record], vocab, 1)
        graph = record['graph']
        present = motifs(graph)
        motif_scores.extend((confidence, float(key in present)) for key, confidence in record['predicted'])
        with torch.no_grad():
            nll_prior.append(float(-prior.likelihood([graph], [record], experiment.DEVICE, False)[0]))
            nll_full.append(float(-model.likelihood([graph], [record], experiment.DEVICE)[0]))
            nll_without_sub.append(float(-model.likelihood([graph], [{**record, 'predicted':[]}], experiment.DEVICE)[0]))
            encoded = model.graph_encoder(*graph_tensors([graph], experiment.DEVICE))
            prediction = model.predict_spectrum(encoded, batch_inputs([record], experiment.DEVICE))[0].cpu().numpy()
        forward.append(float(prediction @ record['bins']))
        baseline.append(float(template @ record['bins']))
    generation = json.loads((base / 'generation_predictions.json').read_text())
    retrieval = json.loads((base / 'retrieval_predictions.json').read_text())
    counts = Counter()
    for row, query in zip(generation, queries):
        counts.update(row.get('work', {}))
        accepted = set(table.hypotheses(query['row'])) if experiment.make_record(query['row'])[0] is not None else set()
        for smiles in row['generated']:
            graph, reason = parse_graph(smiles)
            if reason or graph is None or graph.counts not in accepted:
                raise RuntimeError(f"invalid or mass-inconsistent generated candidate: {row['qid']}")
            counts['checked_returned_candidates'] += 1
        if len(row['generated']) != len(set(row['generated'])):
            raise RuntimeError('duplicate generated connectivity')
    confidence = np.asarray([v[0] for v in motif_scores])
    correct = np.asarray([v[1] for v in motif_scores])
    ece = 0.0
    calibration_bins = []
    for low in np.arange(0, 1, 0.1):
        mask = (confidence >= low) & (confidence < low + 0.1 + (1e-8 if low > 0.85 else 0))
        if mask.any():
            mean, accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += mask.mean() * abs(mean - accuracy)
            calibration_bins.append({'lower':float(low), 'n':int(mask.sum()), 'confidence':mean,'precision':accuracy})
    def mean(values):
        return float(np.mean(values)) if values else None
    hits = [r['rank'] is not None and r['rank'] <= 25 for r in retrieval]
    uniform = [r['uniform_rank'] is not None and r['uniform_rank'] <= 25 for r in retrieval]
    result = {'test_queries':len(queries), 'test_domain':dict(reasons),
        'teacher_forced_supported_test':{'n':len(nll_full), 'prior_nll_per_token':mean(nll_prior),
            'conditional_nll_per_token':mean(nll_full), 'conditional_without_subgraph_nll':mean(nll_without_sub),
            'note':'Input perturbation without retraining; not a trained ablation or stability measure'},
        'forward_spectrum':{'true_graph_cosine':mean(forward),'training_mean_spectrum_cosine':mean(baseline)},
        'predicted_motifs':{'selected_predictions':len(motif_scores),'precision':mean(correct.tolist()),
            'mean_confidence':mean(confidence.tolist()),'ece':ece if len(motif_scores) else None,'bins':calibration_bins,
            'note':'Raw sigmoid scores; confidence has not been calibrated as physical stability'},
        'generation_validity':dict(counts),
        'retrieval_uniform':{f'top{k}':sum(r['uniform_rank'] is not None and r['uniform_rank'] <= k for r in retrieval)/len(retrieval) for k in (1,10,25)},
        'retrieval_full_minus_uniform_top25':bootstrap_difference(hits,uniform,args.seed),
        'scope':'Component/chemical-validity diagnostics; no physical-stability labels'}
    (base / 'component_diagnostics.json').write_text(json.dumps(result,indent=2))
    experiment.emit(result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('experiments/molecular_completion/full_t4'))
    main(parser.parse_args().base)
