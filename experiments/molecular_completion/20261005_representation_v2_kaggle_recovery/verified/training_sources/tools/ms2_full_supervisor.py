"""Run the full T4 verification with logs, stage markers and atomic status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(base):
    base.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Actual CUDA Tesla T4 required')
    status_path = base/'run_status.json'
    start = time.monotonic()
    # Preflight is a separate process; wait without occupying the notebook kernel.
    while not (base/'preflight_complete.json').exists():
        write_json(status_path, {'state':'waiting_preflight','supervisor_pid':os.getpid(),
                                'elapsed_seconds':time.monotonic()-start})
        if time.monotonic()-start > 1800:
            raise TimeoutError('Preflight did not complete within 30 minutes; inspect preflight.log')
        time.sleep(5)
    stages = [
        ('gpu_tests', ['-m','unittest','tools.test_ms2_full_model','tools.test_ms2_full_controls',
                       'tools.test_ms2_full_report','tools.test_ms2_representation','-v']),
        ('training', ['-u','tools/ms2_full_experiment.py','--out',str(base)]),
        ('diagnostics', ['-u','tools/ms2_full_diagnostics.py','--base',str(base)]),
        ('paired_spectrum_audit', ['-u','tools/ms2_full_pairing_audit.py','--base',str(base)]),
        ('controls', ['-u','tools/ms2_full_controls.py','--base',str(base)]),
        ('retrieval_audit', ['-u','tools/ms2_full_retrieval_audit.py','--base',str(base)]),
        ('report', ['-u','tools/ms2_full_report.py','--base',str(base)]),
    ]
    for name, arguments in stages:
        marker = base/(name+'_stage_complete.json')
        if marker.exists():
            print('Restored completed stage',name,flush=True)
            continue
        print('Starting stage',name,flush=True)
        stage_start = time.monotonic()
        # Append preserves previous failure evidence; model epoch checkpoints
        # restore training progress when a stage needs to run again.
        with (base/(name+'.log')).open('a') as log:
            log.write('\nSUPERVISOR_STAGE_START '+name+'\n');log.flush()
            process = subprocess.Popen([sys.executable,*arguments],stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                write_json(status_path, {'state':'running','stage':name,'pid':process.pid,
                    'supervisor_pid':os.getpid(),'stage_seconds':time.monotonic()-stage_start,
                    'elapsed_seconds':time.monotonic()-start})
                if time.monotonic()-stage_start > 24*3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait()
                    raise TimeoutError(f'{name}: 24-hour stage limit reached; epoch checkpoints retained')
                time.sleep(5)
        if process.returncode:
            write_json(status_path, {'state':'failed','stage':name,'exit_code':process.returncode,
                                    'elapsed_seconds':time.monotonic()-start})
            raise RuntimeError(f'{name} exited {process.returncode}; see {base/(name+".log")}')
        write_json(marker, {'stage':name,'seconds':time.monotonic()-stage_start})
    import numpy, scipy, sklearn, rdkit
    write_json(base/'environment.json', {'python':platform.python_version(),'torch':torch.__version__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'rdkit':rdkit.__version__,'gpu':torch.cuda.get_device_name(),
        'revision':'d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a'})
    write_json(status_path, {'state':'complete','elapsed_seconds':time.monotonic()-start})
    manifest = {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in base.rglob('*') if p.is_file() and p.name not in ('manifest.json','supervisor.log')}
    write_json(base/'manifest.json',manifest)
    archive = Path(shutil.make_archive(str(base.parent/'representation_v2_full_t4_results'),'zip',base))
    write_json(base.parent/'representation_v2_archive.json',
        {'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    print('Full verification complete:',archive,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args.base.resolve())
    except Exception as error:
        status=args.base/'run_status.json'
        if not status.exists() or json.loads(status.read_text()).get('state')!='failed':
            write_json(status, {'state':'failed','error':str(error)})
        raise
