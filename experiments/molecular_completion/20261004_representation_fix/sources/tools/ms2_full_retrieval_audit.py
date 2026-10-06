"""Recheck retrieval eligibility with fixed isotope validation and frozen weights.

Preserves the original diagnostic predictions. Generation and trained weights
are never modified. Run after both controls complete.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tools.ms2_full_experiment as experiment
from tools.ms2_full_model import FormulaTable, SubstructurePredictor
from tools.ms2_full_controls import completion_model, restore, EmptyPredictor


def main(base):
    args=argparse.Namespace(**json.loads((base/'plan.json').read_text()))
    experiment.DEVICE=torch.device(args.device)
    fit,calibration,queries,*_=experiment.load_data(args,FormulaTable())
    vocab=torch.load(base/'upstream_final_vocab.pt',weights_only=True)['patterns']
    upstream=restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE),base/'upstream_final.pt')
    paths={'full':base,'transformer_decoder':base/'controls/transformer_decoder',
           'no_substructures':base/'controls/no_substructures'}
    audit={}
    for arm,out in paths.items():
        model=restore(completion_model(arm=='transformer_decoder'),out/'conditional_completion.pt')
        prior_out=base if arm=='no_substructures' else out
        prior=restore(completion_model(arm=='transformer_decoder'),prior_out/'molecular_prior.pt')
        ranker=restore(experiment.FeatureRanker(np.zeros(5),np.ones(5)).to(experiment.DEVICE),out/'same_formula_ranker.pt')
        predictor,patterns=(EmptyPredictor().to(experiment.DEVICE),[]) if arm=='no_substructures' else (upstream,vocab)
        original_path=out/'retrieval_predictions_before_isotope_audit.json'
        if not original_path.exists():
            shutil.copyfile(out/'retrieval_predictions.json',original_path)
        original=json.loads(original_path.read_text())
        rows=[]
        for query in queries:
            record,_=experiment.make_record(query['row'])
            if record is not None:
                experiment.predict_substructures(predictor,[record],patterns,1)
            rows.append(experiment.retrieval_prediction(model,prior,ranker,query,record))
        (out/'retrieval_predictions.json').write_text(json.dumps(rows,indent=2))
        changed=[b['qid'] for a,b in zip(original,rows) if a['rank']!=b['rank'] or a['scored']!=b['scored']]
        metric={f'top{k}':sum(r['rank'] is not None and r['rank']<=k for r in rows)/len(rows) for k in (1,10,25)}
        audit[arm]={'changed_queries':changed,'retrieval':metric,'generation_unchanged':True,'weights_unchanged':True}
        experiment.emit({'stage':'retrieval_isotope_audit','arm':arm,**audit[arm]})
    (base/'retrieval_eligibility_audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('experiments/molecular_completion/full_t4'))
    main(parser.parse_args().base)
