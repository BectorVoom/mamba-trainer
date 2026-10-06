"""Aggregate archived full-pipeline results with connectivity-cluster intervals."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def clustered_difference(a, b, identities, seed=42, repeats=5000):
    if len(a) != len(b) or len(a) != len(identities) or not a:
        raise ValueError('paired nonempty query lists and identities required')
    groups = defaultdict(list)
    for left, right, key in zip(a, b, identities):
        groups[key].append(float(left) - float(right))
    values = list(groups.values())
    sums = np.asarray([sum(v) for v in values])
    counts = np.asarray([len(v) for v in values])
    samples = np.random.default_rng(seed).integers(len(groups), size=(repeats, len(groups)))
    means = sums[samples].sum(1) / counts[samples].sum(1)
    return {'difference':float((np.asarray(a,dtype=float)-np.asarray(b,dtype=float)).mean()),
            'ci95':np.quantile(means,[0.025,0.975]).tolist(),'connectivity_groups':len(groups),
            'unit':'query-weighted difference; bootstrap resamples connectivity groups'}


def main(base):
    audit=json.loads((base/'data_audit.json').read_text())
    identities=audit['test_identities']
    paths={'full':base,'transformer_decoder':base/'controls/transformer_decoder',
           'no_substructures':base/'controls/no_substructures'}
    predictions={}
    result={'queries':len(identities),'distinct_connectivities':len(set(identities)),
            'calibration_queries':audit['calibration'],
            'calibration_distinct_connectivities':len(set(audit['calibration_identities'])),
            'physical_stability_evaluated':False,'single_seed':42,'arms':{},'clustered_comparisons':{}}
    hit=lambda rows,k:[r['rank'] is not None and r['rank']<=k for r in rows]
    for arm,path in paths.items():
        generation=json.loads((path/'generation_predictions.json').read_text())
        retrieval=json.loads((path/'retrieval_predictions.json').read_text())
        if len(generation)!=len(identities) or len(retrieval)!=len(identities):
            raise ValueError('incomplete arm predictions')
        if [r['query_identity'] for r in generation]!=identities:
            raise ValueError('query ordering/identity mismatch')
        predictions[arm]=(generation,retrieval)
        result['arms'][arm]={
            'generation':{f'top{k}':float(np.mean(hit(generation,k))) for k in (1,10,25)},
            'retrieval':{f'top{k}':float(np.mean(hit(retrieval,k))) for k in (1,10,25)},
            'queries_with_completed_graphs':sum(r['valid_candidates']>0 for r in generation),
            'distinct_candidates_per_query_total':sum(r['valid_candidates'] for r in generation)}
    uniform=[r['uniform_rank'] is not None and r['uniform_rank']<=25 for r in predictions['full'][1]]
    result['retrieval_uniform_top25']=float(np.mean(uniform))
    result['clustered_comparisons']['retrieval_full_minus_uniform']=clustered_difference(hit(predictions['full'][1],25),uniform,identities)
    for arm in ('transformer_decoder','no_substructures'):
        for index,name in enumerate(('generation','retrieval')):
            result['clustered_comparisons'][name+'_full_minus_'+arm]=clustered_difference(
                hit(predictions['full'][index],25),hit(predictions[arm][index],25),identities)
    (base/'verification_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('experiments/molecular_completion/full_t4'))
    main(parser.parse_args().base)
