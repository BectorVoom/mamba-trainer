"""Kaggle launch and collection behavior without network access."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.ms2_kaggle_prepare import prepare
from tools.ms2_kaggle_collect import collect, verify


class KaggleTests(unittest.TestCase):
    def test_recovery_preserves_original_source_snapshot_and_requires_unchanged_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = prepare(root/'kernel', recover_from='boomvector/original')
            metadata = json.loads((root/'kernel/kernel-metadata.json').read_text())
            self.assertEqual(metadata['kernel_sources'], ['boomvector/original'])
            tree = ast.parse(''.join(notebook['cells'][0]['source']))
            runner = next(ast.literal_eval(n.value) for n in tree.body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'runner' for t in n.targets))
            setup = runner[runner.index('from pathlib import Path\nimport os'):].split('import hashlib, time, requests', 1)[0]
            setup = setup.replace('/kaggle/working/ms2', str(root/'work')).replace('/kaggle/input', str(root/'input'))
            previous = root/'input/representation_v2_full_t4'
            (previous/'sources/tools').mkdir(parents=True)
            (previous/'run_status.json').write_text('{"state":"failed"}')
            original = b'original immutable training model'
            (previous/'sources/tools/ms2_full_model.py').write_bytes(original)
            (previous/'source_sha256.json').write_text(json.dumps({'tools/ms2_full_model.py':hashlib.sha256(original).hexdigest()}))
            (previous/'gpu_tests_stage_complete.json').write_text('{}')
            import os
            before = Path.cwd()
            try:
                exec(compile(setup, '<recovery-setup>', 'exec'), {})
            finally:
                os.chdir(before)
            restored = root/'work/experiments/molecular_completion/representation_v2_full_t4'
            self.assertEqual((restored/'training_sources/tools/ms2_full_model.py').read_bytes(), original)
            self.assertTrue((restored/'recovery_request.json').exists())
            self.assertFalse((restored/'gpu_tests_stage_complete.json').exists())
            self.assertNotEqual((restored/'sources/tools/ms2_full_model.py').read_bytes(), original)

    def test_notebook_requires_t4_and_compiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            notebook = prepare(Path(temporary))
            for cell in notebook['cells']:
                compile(''.join(cell['source']), '<notebook>', 'exec')
            metadata = json.loads((Path(temporary)/'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertEqual(metadata['machine_shape'], 'NvidiaTeslaT4')
            tree = ast.parse(''.join(notebook['cells'][0]['source']))
            runner = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'runner' for t in n.targets))
            compile(runner,'<runner>','exec')
            self.assertIn("assert 'T4'", runner)

    def test_timed_segment_attaches_previous_outputs_and_then_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prepare(out/'kernel')
            calls = []
            def command(arguments, timeout=600):
                calls.append(arguments)
                if arguments[:2] == ['kernels','logs']:
                    return 'log'
                if arguments[:2] == ['kernels','status']:
                    return 'KernelWorkerStatus.COMPLETE'
                if arguments[:2] == ['kernels','output']:
                    destination = Path(arguments[arguments.index('-p')+1])
                    state = destination/'representation_v2_full_t4'/'run_status.json'
                    state.parent.mkdir(parents=True)
                    state.write_text(json.dumps({'state':'segment_complete' if 'segment_1' in str(destination) else 'complete'}))
                return 'ok'
            with patch('tools.ms2_kaggle_collect.subprocess.Popen'), patch('tools.ms2_kaggle_collect.command',side_effect=command), patch('tools.ms2_kaggle_collect.verify', return_value={}) as checked:
                collect(out,max_segments=2)
                checked.assert_called_once_with(out/'segment_2',out)
            next_metadata = json.loads((out/'kernel_s2/kernel-metadata.json').read_text())
            self.assertEqual(next_metadata['kernel_sources'],['boomvector/ms2-representation-v2-t4-s1'])
            self.assertTrue(any(c[:2] == ['kernels','push'] for c in calls))

    def test_failed_remote_stage_never_launches_another_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prepare(out/'kernel')
            def command(arguments, timeout=600):
                if arguments[:2] == ['kernels','logs']:
                    return 'log'
                if arguments[:2] == ['kernels','status']:
                    return 'KernelWorkerStatus.COMPLETE'
                state = out/'segment_1/representation_v2_full_t4/run_status.json'
                state.parent.mkdir(parents=True)
                state.write_text(json.dumps({'state':'failed','stage':'gpu_tests'}))
                return 'ok'
            with patch('tools.ms2_kaggle_collect.subprocess.Popen'), patch('tools.ms2_kaggle_collect.command',side_effect=command):
                with self.assertRaisesRegex(ValueError,'Remote experiment failed'):
                    collect(out)
            self.assertFalse((out/'kernel_s2').exists())

    def test_collector_restart_uses_latest_submitted_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prepare(out/'kernel_s2')
            metadata = json.loads((out/'kernel_s2/kernel-metadata.json').read_text())
            metadata['id'] = 'boomvector/ms2-representation-v2-t4-recovery-s2'
            (out/'kernel_s2/kernel-metadata.json').write_text(json.dumps(metadata))
            (out/'kernel_s2/submission_complete.json').write_text('{}')
            def command(arguments, timeout=600):
                self.assertEqual(arguments[2], metadata['id'])
                if arguments[:2] == ['kernels','status']:
                    return 'KernelWorkerStatus.COMPLETE'
                state = out/'segment_2/representation_v2_full_t4/run_status.json'
                state.parent.mkdir(parents=True)
                state.write_text('{"state":"complete"}')
                return 'ok'
            with patch('tools.ms2_kaggle_collect.subprocess.Popen'), patch('tools.ms2_kaggle_collect.command',side_effect=command), patch('tools.ms2_kaggle_collect.verify',return_value={}):
                collect(out, max_segments=2)

    def test_unsafe_archive_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            archive = out/'representation_v2_full_t4_results.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('../escape.txt','unsafe')
            (out/'representation_v2_archive.json').write_text(json.dumps({'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError,'Unsafe archive member'):
                verify(out,out)
            self.assertFalse((out/'escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
