"""Held-out correct-spectrum versus substituted-spectrum intervention.

Same-formula donors are preferred. Unmatched-formula substitutions are reported
separately and are not evidence of same-formula discrimination. Weights are frozen.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ms2_full_experiment as experiment
from tools.ms2_full_model import CompletionModel, FormulaTable, SubstructurePredictor, prep_peaks
from tools.ms2_full_controls import restore
from tools.ms2_spectral_rank import parse_spectrum


def substitute_peaks(target, donor):
    peaks, reason = parse_spectrum(donor['row']['mzs'], donor['row']['intensities'])
    if reason:
        raise ValueError(f'invalid donor peaks: {reason}')
    features, mask, bins = prep_peaks(peaks, float(target['row']['precursor_mz']))
    return {**target, 'peaks': features, 'mask': mask, 'bins': bins, 'predicted': []}


def main(base):
    args = argparse.Namespace(**json.loads((base/'plan.json').read_text()))
    experiment.DEVICE = torch.device(args.device)
    _, _, queries, *_ = experiment.load_data(args, FormulaTable())
    model = restore(CompletionModel().to(experiment.DEVICE), base/'conditional_completion.pt')
    prior = restore(CompletionModel().to(experiment.DEVICE), base/'molecular_prior.pt')
    vocab = torch.load(base/'upstream_final_vocab.pt', weights_only=True)['patterns']
    upstream = restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE), base/'upstream_final.pt')
    ranker = restore(experiment.FeatureRanker(np.zeros(5), np.ones(5)).to(experiment.DEVICE),
                     base/'same_formula_ranker.pt')
    records = []
    for query in queries:
        record, reason = experiment.make_record(query['row'])
        if record is not None and reason is None:
            records.append((query, record))
    rows = []
    for query, record in records:
        donors = [(q,r) for q,r in records if r['identity'] != record['identity']]
        matched = [(q,r) for q,r in donors if r['graph'].counts == record['graph'].counts]
        if not donors:
            continue
        donor_query, donor = min(matched or donors,
            key=lambda pair: (abs(float(pair[1]['row']['precursor_mz'])-float(record['row']['precursor_mz'])), pair[0]['qid']))
        wrong = substitute_peaks(record, donor)
        if not wrong['mask'].any():
            rows.append({'qid':query['qid'], 'donor':donor_query['qid'], 'status':'empty_after_precursor_filter'})
            continue
        experiment.predict_substructures(upstream, [record, wrong], vocab, 2)
        with torch.no_grad():
            correct_nll = float(-model.likelihood([record['graph']], [record], experiment.DEVICE)[0])
            wrong_nll = float(-model.likelihood([record['graph']], [wrong], experiment.DEVICE)[0])
            correct_rank = experiment.retrieval_prediction(model, prior, ranker, query, record)['rank']
            wrong_rank = experiment.retrieval_prediction(model, prior, ranker, query, wrong)['rank']
        rows.append({'qid':query['qid'], 'donor':donor_query['qid'], 'status':'ok',
            'query_identity':record['identity'], 'same_formula_donor':bool(matched),
            'correct_nll':correct_nll, 'substituted_nll':wrong_nll,
            'nll_increase':wrong_nll-correct_nll, 'correct_rank':correct_rank, 'substituted_rank':wrong_rank})
    summary = {}
    for matched, name in ((True,'same_formula'), (False,'unmatched_formula')):
        group = [r for r in rows if r['status']=='ok' and r['same_formula_donor']==matched]
        summary[name] = {'queries':len(group),
            'mean_nll_increase':float(np.mean([r['nll_increase'] for r in group])) if group else None,
            'fraction_correct_spectrum_lower_nll':float(np.mean([r['nll_increase']>0 for r in group])) if group else None}
    result = {'scope':'frozen held-out spectral substitution; target metadata retained; motifs recomputed',
              'rows':rows, 'summary':summary}
    (base/'paired_spectrum_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    main(parser.parse_args().base)
