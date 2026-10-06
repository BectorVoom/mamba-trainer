"""Build a private Kaggle T4 notebook from the current representation-v2 sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(destination, recover_from=None):
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'notebooks/ms2_full_t4_colab.ipynb').read_text())
    tree = ast.parse(''.join(original['cells'][2]['source']))
    names = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
    sources = {name: (root/name).read_text() for name in names}
    for name in ('tools/ms2_full_recover.py', 'tools/test_ms2_full_recover.py'):
        sources[name] = (root/name).read_text()
    cells = []
    def cell(source):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(True)))
    cell("import subprocess, sys\nimport torch\nassert torch.cuda.is_available(), 'CUDA required'\nassert 'T4' in torch.cuda.get_device_name(), 'T4 required'\nprint(torch.__version__, torch.cuda.get_device_name(), flush=True)\nsubprocess.run(['nvidia-smi'], check=True)\n")
    cell("""from pathlib import Path
import os, json, shutil, sys, hashlib
WORK = Path('/kaggle/working/ms2')
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
RUN_LOG_DIR = WORK / 'experiments/molecular_completion/representation_v2_full_t4'
# A continuation attaches the previous notebook's persisted output.
previous = list(Path('/kaggle/input').rglob('run_status.json'))
previous = [p for p in previous if p.parent.name == 'representation_v2_full_t4']
if len(previous) > 1:
    raise ValueError('Multiple continuation runs attached')
for status in previous:
    if status.parent.name == 'representation_v2_full_t4':
        shutil.copytree(status.parent, RUN_LOG_DIR, dirs_exist_ok=True)
        print('Restored previous segment:', status.parent)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
""" + 'RECOVER = ' + repr(bool(recover_from)) + '\nSOURCES = ' + repr(sources) + """
first_recovery = RECOVER and not (RUN_LOG_DIR/'recovery_request.json').exists()
if first_recovery:
    if not previous:
        raise ValueError('Recovery requires previous Kaggle outputs')
    old_hashes = json.loads((RUN_LOG_DIR/'source_sha256.json').read_text())
    for name, expected in old_hashes.items():
        if hashlib.sha256((RUN_LOG_DIR/'sources'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original source hash mismatch: '+name)
    shutil.copytree(RUN_LOG_DIR/'sources', RUN_LOG_DIR/'training_sources')
    (RUN_LOG_DIR/'training_source_sha256.json').write_text(json.dumps(old_hashes,indent=2))
    (RUN_LOG_DIR/'recovery_request.json').write_text(json.dumps({'mode':'evaluation-only primary; train matched controls normally'}))
    (RUN_LOG_DIR/'gpu_tests_stage_complete.json').unlink(missing_ok=True)
for name, source in SOURCES.items():
    path = WORK/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    snapshot = RUN_LOG_DIR/'sources'/name
    if snapshot.exists() and snapshot.read_text() != source and not first_recovery:
        raise ValueError('Continuation source changed: '+name)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(source)
(RUN_LOG_DIR/'source_sha256.json').write_text(json.dumps({n: hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()}, indent=2))
(RUN_LOG_DIR/'controls').mkdir(exist_ok=True)
(RUN_LOG_DIR/'controls/preregistered_plan.json').write_text(json.dumps({'arms':['transformer_decoder','no_substructures'], 'same_split_and_epochs':True,'seed':42,'queries':100,'trajectories_per_formula':32},indent=2))
""")
    cell(''.join(original['cells'][3]['source']))
    cell("""import platform, numpy, scipy, sklearn, rdkit, time, signal, subprocess
(RUN_LOG_DIR/'environment.json').write_text(json.dumps({'platform':'Kaggle','python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'revision':REVISION},indent=2))
(RUN_LOG_DIR/'preflight_complete.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'pinned_data_verified':True}))
# End before the platform's session limit so Kaggle can persist epoch checkpoints.
started = time.monotonic()
with (RUN_LOG_DIR/'supervisor.log').open('a') as log:
    process = subprocess.Popen([sys.executable,'-u','tools/ms2_full_supervisor.py','--base',str(RUN_LOG_DIR)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    offsets = {}
    while process.poll() is None:
        for path in sorted(RUN_LOG_DIR.glob('*.log')):
            with path.open() as stream:
                stream.seek(offsets.get(str(path),0))
                chunk = stream.read()
                offsets[str(path)] = stream.tell()
            if chunk:
                print(path.name+': '+chunk, flush=True)
        if time.monotonic()-started > 10.5*3600:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status_path = RUN_LOG_DIR/'run_status.json'
            status = json.loads(status_path.read_text())
            status.update(state='segment_complete', reason='Kaggle time budget; resume saved epoch checkpoints')
            status_path.write_text(json.dumps(status,indent=2))
            break
        time.sleep(15)
print((RUN_LOG_DIR/'run_status.json').read_text(),flush=True)
# The notebook finishes normally even on a failed stage, preserving diagnostic outputs.
# Delete bulky input downloads; all experiment outputs and source evidence remain.
shutil.rmtree(WORK/'data', ignore_errors=False)
""")
    # Kaggle preloads NumPy in its notebook process; run pinned dependencies in
    # a fresh interpreter to avoid mixing already-loaded extension versions.
    runner = '\n'.join(''.join(c['source']) for c in cells)
    cells.clear()
    cell("import subprocess, sys, shutil, os\nfrom pathlib import Path\n"
         "environment = Path('/kaggle/working/.ms2-deps')\n"
         "python = sys.executable\n"
         "subprocess.run([python, '-m', 'pip', 'install', '-q', '--target', str(environment), 'numpy==2.2.6', 'scipy==1.15.3', 'scikit-learn==1.6.1', 'requests==2.32.5', 'rdkit==2025.9.6'], check=True)\n"
         + "runner = " + repr(runner) + "\n"
         "script = Path('/kaggle/working/ms2_runner.py')\n"
         "script.write_text(runner)\n"
         "env = os.environ.copy()\n"
         "env['PYTHONPATH'] = str(environment) + os.pathsep + env.get('PYTHONPATH', '')\n"
         "result = subprocess.run([python, '-u', str(script)], env=env)\n"
         "shutil.rmtree(environment)\n"
         "if result.returncode: raise RuntimeError('MS2 runner failed; inspect preserved logs')\n")
    destination.mkdir(parents=True, exist_ok=True)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, cells=cells)
    for index, c in enumerate(cells):
        c['id'] = f'ms2-kaggle-{index}'
    (destination/'ms2_representation_v2_t4.ipynb').write_text(json.dumps(notebook,indent=1))
    metadata = dict(id='boomvector/ms2-representation-v2-t4-s1', title='MS2 representation v2 T4 s1', code_file='ms2_representation_v2_t4.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    if recover_from:
        metadata.update(id='boomvector/ms2-representation-v2-t4-recovery-s1',
                        title='MS2 representation v2 T4 recovery s1',
                        kernel_sources=[recover_from])
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover-from')
    args = parser.parse_args()
    prepare(args.out, args.recover_from)
