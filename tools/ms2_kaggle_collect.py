"""Collect Kaggle outputs, resume timed segments, and verify the final benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def command(arguments, timeout=600):
    result = subprocess.run(['kaggle', *arguments], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout


def verify(downloaded, out):
    from tools.ms2_colab_collect import verify_models, write_results
    metadata = list(downloaded.rglob('representation_v2_archive.json'))
    archives = list(downloaded.rglob('representation_v2_full_t4_results.zip'))
    if len(metadata) != 1 or len(archives) != 1:
        raise ValueError('Expected one final archive and its hash metadata')
    archive = archives[0]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != json.loads(metadata[0].read_text())['sha256']:
        raise ValueError('Archive SHA-256 mismatch')
    extracted = out/'verified'
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC failure')
        for name in bundle.namelist():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member: '+name)
        bundle.extractall(extracted)
    manifest = json.loads((extracted/'manifest.json').read_text())
    for name, expected in manifest.items():
        member = Path(name)
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('Unsafe manifest member: '+name)
        if hashlib.sha256((extracted/member).read_bytes()).hexdigest() != expected:
            raise ValueError('Manifest hash mismatch: '+name)
    checked = dict(archive_sha256=digest, manifest_files=len(manifest), **verify_models(extracted))
    (extracted/'local_artifact_verification.json').write_text(json.dumps(checked,indent=2))
    write_results(extracted,checked)
    results = (extracted/'RESULTS.md').read_text().replace('full Colab T4 verification','full Kaggle T4 verification')
    (extracted/'RESULTS.md').write_text(results)
    shutil.copyfile(extracted/'RESULTS.md',out/'RESULTS.md')
    return checked


def collect(out, max_segments=8):
    kernel = out/'kernel'
    first_segment = 1
    submitted = [p for p in out.glob('kernel_s*') if (p/'submission_complete.json').exists()]
    if submitted:
        kernel = max(submitted, key=lambda p: int(p.name.removeprefix('kernel_s')))
        first_segment = int(kernel.name.removeprefix('kernel_s'))
    metadata = json.loads((kernel/'kernel-metadata.json').read_text())
    ref = metadata['id']
    for segment in range(first_segment,max_segments+1):
        downloaded = out/f'segment_{segment}'
        failures = 0
        deadline = time.monotonic() + 13*3600
        # One-shot logs expose only persisted output; follow streams live events.
        with (out/'kaggle_live.log').open('a') as stream:
            subprocess.Popen(['kaggle','kernels','logs',ref,'--follow'],
                             stdout=stream,stderr=subprocess.STDOUT)
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('Kaggle segment exceeded 13 hours; inspect remote status')
            try:
                status = command(['kernels','status',ref],timeout=55).strip()
                (out/'collector_status.json').write_text(json.dumps({'kernel':ref,'segment':segment,'status':status,'updated_unix':time.time()},indent=2))
                print(status,flush=True)
                if 'COMPLETE' in status or 'ERROR' in status or 'CANCEL' in status:
                    command(['kernels','output',ref,'-p',str(downloaded)])
                    states = [p for p in downloaded.rglob('run_status.json') if p.parent.name == 'representation_v2_full_t4']
                    if len(states) != 1:
                        raise ValueError('Missing remote run status; inspect downloaded notebook log')
                    state = json.loads(states[0].read_text())
                    (out/'run_status.json').write_text(json.dumps(state,indent=2))
                    if state['state'] == 'complete':
                        print(json.dumps(verify(downloaded,out)),flush=True)
                        return
                    if state['state'] != 'segment_complete':
                        raise ValueError('Remote experiment failed: '+json.dumps(state))
                    break
                failures = 0
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                failures += 1
                print('Transient collection failure:',error,flush=True)
                if failures >= 10:
                    raise
            time.sleep(45)
        if segment == max_segments:
            raise TimeoutError('Segment limit reached; saved checkpoints remain available')
        next_kernel = out/f'kernel_s{segment+1}'
        next_kernel.mkdir(exist_ok=True)
        shutil.copyfile(kernel/metadata['code_file'],next_kernel/metadata['code_file'])
        metadata.update(id=ref.rsplit('-s',1)[0]+f'-s{segment+1}',
                        title=metadata['title'].rsplit(' s',1)[0]+f' s{segment+1}',kernel_sources=[ref])
        (next_kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        print(command(['kernels','push','-p',str(next_kernel),'--accelerator','NvidiaTeslaT4']),flush=True)
        (next_kernel/'submission_complete.json').write_text(json.dumps({'kernel':metadata['id']}))
        ref = metadata['id']
        kernel = next_kernel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    try:
        collect(args.out.resolve())
    except Exception as error:
        (args.out/'collector_failure.json').write_text(json.dumps({'error':str(error)},indent=2))
        raise
