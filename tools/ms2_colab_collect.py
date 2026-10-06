"""Collect a supervised Colab run and verify its archived artifacts locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def download(cli, session, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name+'.transfer')
    try:
        result = subprocess.run([cli,'download','-s',session,remote,str(temporary)],
                                capture_output=True,text=True,timeout=55)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:] or result.stdout[-1000:])
        temporary.replace(local)
    finally:
        temporary.unlink(missing_ok=True)


def remote_names(args, remote):
    result = subprocess.run([args.cli,'ls','-s',args.session,remote],
                            capture_output=True,text=True,timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:] or result.stdout[-1000:])
    names = []
    for line in result.stdout.splitlines():
        name = line.strip().rstrip('/')
        if not name or name.startswith('[colab]'):
            continue
        if Path(name).name != name or name in ('.','..'):
            raise ValueError('Unsafe remote member: '+name)
        names.append(name)
    return names


def backup_checkpoints(args, epochs):
    """Copy completed models and each newly completed epoch off the volatile VM."""
    backup = args.out.with_name(args.out.name+'_checkpoints')
    roots = [('',remote_names(args,args.remote))]
    if 'controls' in roots[0][1]:
        controls = remote_names(args,args.remote+'/controls')
        for arm in ('transformer_decoder','no_substructures'):
            if arm in controls:
                roots.append(('controls/'+arm,remote_names(args,args.remote+'/controls/'+arm)))
    for relative,names in roots:
        remote = args.remote+('/'+relative if relative else '')
        local = backup/relative
        for name in names:
            if name in ('plan.json','data_audit.json') or (name.endswith('.pt') and not name.endswith('_progress.pt')):
                destination = local/name
                if not destination.exists():
                    download(args.cli,args.session,remote+'/'+name,destination)
                    if name.endswith('.pt') and not name.endswith('_vocab.pt'):
                        history_name = name.removesuffix('.pt')+'_history.json'
                        if history_name in names:
                            download(args.cli,args.session,remote+'/'+history_name,local/history_name)
                        (local/(name.removesuffix('.pt')+'_progress.pt')).unlink(missing_ok=True)
            elif name.endswith('_progress.pt'):
                completed_name = name.removesuffix('_progress.pt')+'.pt'
                if completed_name in names:
                    (local/name).unlink(missing_ok=True)
                    continue
                history_name = name.removesuffix('_progress.pt')+'_history.json'
                if history_name not in names:
                    continue  # The first epoch commit is still being written.
                download(args.cli,args.session,remote+'/'+history_name,local/history_name)
                history = json.loads((local/history_name).read_text())['history']
                epoch = len(history)
                key = relative+'/'+name
                if epoch > epochs.get(key,0):
                    download(args.cli,args.session,remote+'/'+name,local/name)
                    epochs[key] = epoch
                    print('Backed up checkpoint:',key,'epoch',epoch,flush=True)
    if backup.exists():
        manifest = {str(p.relative_to(backup)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in backup.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (backup/'manifest.json').write_text(json.dumps(manifest,indent=2))


def verify_models(base):
    import numpy as np
    import torch
    from tools import ms2_full_experiment as experiment
    from tools.ms2_full_model import SubstructurePredictor
    from tools.ms2_full_controls import completion_model
    experiment.DEVICE = torch.device('cpu')
    models, vocabularies = 0, 0
    for path in sorted(base.rglob('*.pt')):
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if 'state' not in checkpoint:
            if 'patterns' not in checkpoint:
                raise ValueError(f'Unexpected checkpoint format: {path}')
            vocabularies += 1
            continue
        if checkpoint.get('representation') != experiment.REPRESENTATION_VERSION:
            raise ValueError(f'Unexpected representation: {path}')
        state = checkpoint['state']
        if path.stem.startswith('upstream_'):
            model = SubstructurePredictor(state['head.weight'].shape[0])
        elif path.stem == 'same_formula_ranker':
            model = experiment.FeatureRanker(np.zeros(5),np.ones(5))
        else:
            model = completion_model('transformer_decoder' in path.parts)
        model.load_state_dict(state,strict=True)
        if any(not torch.isfinite(t).all() for t in state.values() if t.is_floating_point()):
            raise ValueError(f'Nonfinite checkpoint: {path}')
        models += 1
    if models != 12 or vocabularies != 4:
        raise ValueError(f'Incomplete checkpoints: {models} models, {vocabularies} vocabularies')
    return {'strict_models':models,'vocabularies':vocabularies}


def write_results(base, checked):
    report=json.loads((base/'verification_report.json').read_text())
    audit=json.loads((base/'data_audit.json').read_text())
    lines=['# Representation v2: full Colab T4 verification','',
           f"Training used {audit['fit']:,} measured spectra across "
           f"{len(set(audit['fit_identities'])):,} molecular connectivities. "
           f"Evaluation used {report['queries']} test queries.",'',
           '| Arm | Generation top-1 / 10 / 25 | Supplied-pool retrieval top-1 / 10 / 25 |',
           '|---|---|---|']
    for name,arm in report['arms'].items():
        format_scores=lambda kind: ' / '.join(f"{arm[kind]['top'+str(k)]*100:.1f}%" for k in (1,10,25))
        lines.append(f"| {name} | {format_scores('generation')} | {format_scores('retrieval')} |")
    lines.extend(['',f"Shuffled-pool retrieval top-25: {report['retrieval_uniform_top25']*100:.1f}%.",'',
                  'Measured spectrum substitution results are in `paired_spectrum_audit.json`; '
                  'same-formula and unmatched-formula donor results are reported separately.', '',
                  'This experiment measures identification and graph validity. Physical stability '
                  'was not evaluated. The decoder remains connectivity only, and the new neural '
                  'features are implemented in the Python experiment.', '',
                  f"Archive SHA-256: `{checked['archive_sha256']}`. "
                  f"Verified {checked['manifest_files']} manifest files, "
                  f"{checked['strict_models']} model checkpoints and {checked['vocabularies']} vocabularies."])
    (base/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def collect(args):
    args.out.mkdir(parents=True,exist_ok=True)
    deadline = time.monotonic()+24*3600
    previous, failures, backup_epochs = None, 0, {}
    while time.monotonic() < deadline:
        try:
            download(args.cli,args.session,args.remote+'/run_status.json',args.out/'run_status.json')
            state = json.loads((args.out/'run_status.json').read_text())
            key = (state['state'],state.get('stage'))
            if key != previous:
                print(json.dumps(state),flush=True);previous=key
            stage = state.get('stage')
            if stage:
                download(args.cli,args.session,args.remote+'/'+stage+'.log',args.out/(stage+'.log'))
            if state['state']!='complete':
                backup_checkpoints(args,backup_epochs)
            if state['state']=='failed':
                raise RuntimeError('Remote stage failed: '+json.dumps(state))
            if state['state']=='complete':
                download(args.cli,args.session,str(Path(args.remote).parent/'representation_v2_archive.json'),
                         args.out/'download_expected.json')
                expected=json.loads((args.out/'download_expected.json').read_text())
                temporary=args.out/'results.download'
                download(args.cli,args.session,expected['path'],temporary)
                actual=hashlib.sha256(temporary.read_bytes()).hexdigest()
                if actual!=expected['sha256'] or temporary.stat().st_size!=expected['bytes']:
                    raise ValueError('Archive size/hash mismatch')
                temporary.replace(args.out/'results.zip')
                with zipfile.ZipFile(args.out/'results.zip') as archive:
                    if archive.testzip() is not None:
                        raise ValueError('Archive CRC failure')
                    for name in archive.namelist():
                        member=Path(name)
                        if member.is_absolute() or '..' in member.parts:
                            raise ValueError('Unsafe archive member: '+name)
                    archive.extractall(args.out)
                manifest=json.loads((args.out/'manifest.json').read_text())
                for name,digest in manifest.items():
                    if hashlib.sha256((args.out/name).read_bytes()).hexdigest()!=digest:
                        raise ValueError('Manifest hash mismatch: '+name)
                checked={'archive_sha256':actual,'manifest_files':len(manifest),**verify_models(args.out)}
                (args.out/'local_artifact_verification.json').write_text(json.dumps(checked,indent=2))
                write_results(args.out, checked)
                print('Verified full experiment:',json.dumps(checked),flush=True)
                subprocess.run([args.cli,'stop','-s',args.session],check=True,timeout=55)
                return
            failures=0
        except (OSError,RuntimeError,ValueError,zipfile.BadZipFile,subprocess.TimeoutExpired) as error:
            failures+=1
            print('Collection attempt failed:',str(error),flush=True)
            if failures>=10 or ('state' in locals() and state.get('state')=='failed'):
                (args.out/'collector_failure.json').write_text(json.dumps({'error':str(error)},indent=2))
                raise
        time.sleep(45)
    raise TimeoutError('Collection exceeded 24 hours')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',default='/Users/ods/.local/bin/colab')
    parser.add_argument('--session',required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--out',type=Path,required=True)
    collect(parser.parse_args())
