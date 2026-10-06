"""Evaluate completed primary checkpoints without altering training provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ms2_full_experiment as experiment
from tools.ms2_full_controls import restore
from tools.ms2_full_model import CompletionModel, FormulaTable, SubstructurePredictor
from tools.ms2_spectral_rank import sha256_file


def verify_training_sources(base):
    sources = base/'training_sources'
    manifest = json.loads((base/'training_source_sha256.json').read_text())
    for name, expected in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe training source path: '+name)
        if sha256_file(sources/path) != expected:
            raise ValueError('Training source hash mismatch: '+name)
    return hashlib.sha256((sources/'tools/ms2_full_experiment.py').read_bytes() +
                          (sources/'tools/ms2_full_model.py').read_bytes()).hexdigest()


def main(base):
    args = argparse.Namespace(**json.loads((base/'plan.json').read_text()))
    args.out = str(base)
    experiment.DEVICE = torch.device(args.device)
    if args.device != 'cuda' or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Recovery requires an actual CUDA T4')
    source = verify_training_sources(base)
    audit = json.loads((base/'data_audit.json').read_text())
    for name, path in [('tsv',args.tsv), ('prefix',args.prefix)]:
        if sha256_file(path) != audit['source_sha256'][name]:
            raise ValueError('Recovery data hash mismatch: '+name)
    checkpoint_hashes = {}
    for name in ['upstream_fold0','upstream_fold1','upstream_fold2','upstream_final',
                 'molecular_prior','conditional_completion','same_formula_ranker']:
        path = base/(name+'.pt')
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['protocol']['source'] != source:
            raise ValueError('Original checkpoint source mismatch: '+name)
        epochs = args.upstream_epochs if name.startswith('upstream') else (
            args.prior_epochs if name == 'molecular_prior' else (
            args.completion_epochs if name == 'conditional_completion' else args.rank_epochs))
        history = json.loads((base/(name+'_history.json')).read_text())['history']
        if len(history) != epochs or checkpoint['protocol']['epochs'] != epochs:
            raise ValueError('Incomplete primary training schedule: '+name)
        checkpoint_hashes[name] = sha256_file(path)
    table = FormulaTable()
    fit, calibration, queries, *_ = experiment.load_data(args, table)
    if ([r['identity'] for r in fit] != audit['fit_identities'] or
        [r['identity'] for r in calibration] != audit['calibration_identities'] or
        len(queries) != audit['test']):
        raise ValueError('Recovery split differs from the original training run')
    model = restore(CompletionModel().to(experiment.DEVICE),base/'conditional_completion.pt')
    prior = restore(CompletionModel().to(experiment.DEVICE),base/'molecular_prior.pt')
    vocab = torch.load(base/'upstream_final_vocab.pt',weights_only=True)['patterns']
    upstream = restore(SubstructurePredictor(len(vocab)).to(experiment.DEVICE),base/'upstream_final.pt')
    ranker = restore(experiment.FeatureRanker(np.zeros(5),np.ones(5)).to(experiment.DEVICE),
                     base/'same_formula_ranker.pt')
    experiment.seed_all(args.seed)
    # Re-evaluate all queries with the fix; do not combine old partial results.
    generation, retrieval = experiment.evaluate_full(model,prior,ranker,upstream,vocab,queries,table,args,base)
    experiment.write_summary(args,base,model,fit,calibration,queries,generation,retrieval)
    if any(sha256_file(base/(name+'.pt')) != digest for name,digest in checkpoint_hashes.items()):
        raise ValueError('Primary checkpoint modified during evaluation')
    (base/'primary_recovery_verification.json').write_text(json.dumps({
        'scope':'evaluation-only; original primary training checkpoints retained',
        'training_source':source,'checkpoint_sha256':checkpoint_hashes,
        'queries':len(queries),'fix':'remove explicit stereo hydrogen before heavy-atom features'
    },indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args().base.resolve())
