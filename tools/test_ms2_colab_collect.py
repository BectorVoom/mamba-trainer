import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import ms2_colab_collect as collector


class CollectorTests(unittest.TestCase):
    def test_interrupted_transfer_preserves_last_good_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model_progress.pt'
            path.write_bytes(b'last good epoch')
            def interrupted(command,**_):
                Path(command[-1]).write_bytes(b'partial')
                return SimpleNamespace(returncode=1,stderr='interrupted transfer',stdout='')
            with patch.object(collector.subprocess,'run',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted transfer'):
                    collector.download('colab','test','/remote/model.pt',path)
            self.assertEqual(path.read_bytes(),b'last good epoch')
            self.assertFalse(path.with_name(path.name+'.transfer').exists())

    def test_each_completed_epoch_and_final_model_are_backed_up(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            args=SimpleNamespace(out=output,cli='colab',session='test',remote='/remote/run')
            epoch=1
            copies=[]
            def copy(_cli,_session,remote,local):
                local.parent.mkdir(parents=True,exist_ok=True)
                copies.append(remote)
                if remote.endswith('_history.json'):
                    local.write_text(json.dumps({'history':[{}]*epoch}))
                else:
                    local.write_bytes(('checkpoint '+str(epoch)).encode())
            names=['upstream_fold0.pt','conditional_completion_progress.pt','conditional_completion_history.json']
            seen={}
            with patch.object(collector,'remote_names',return_value=names), \
                 patch.object(collector,'download',side_effect=copy),patch('builtins.print'):
                collector.backup_checkpoints(args,seen)
                collector.backup_checkpoints(args,seen)
                epoch=2
                collector.backup_checkpoints(args,seen)
            self.assertEqual(copies.count('/remote/run/upstream_fold0.pt'),1)
            self.assertEqual(copies.count('/remote/run/conditional_completion_progress.pt'),2)
            backup=output.with_name('run_checkpoints')
            self.assertEqual((backup/'conditional_completion_progress.pt').read_bytes(),b'checkpoint 2')
            self.assertIn('conditional_completion_progress.pt',json.loads((backup/'manifest.json').read_text()))

    def fixture(self, unsafe=False):
        contents = b'{"fixture":true}'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            name = '../escaped.json' if unsafe else 'summary.json'
            archive.writestr(name, contents)
            archive.writestr('manifest.json',json.dumps({name:hashlib.sha256(contents).hexdigest()}))
        return buffer.getvalue()

    def run_collector(self, directory, data, wrong_hash=False):
        def download(_cli,_session,remote,local):
            if remote.endswith('run_status.json'):
                local.write_text(json.dumps({'state':'complete'}))
            elif remote.endswith('representation_v2_archive.json'):
                local.write_text(json.dumps({'path':'/remote/results.zip','bytes':len(data),
                    'sha256':'bad' if wrong_hash else hashlib.sha256(data).hexdigest()}))
            else:
                local.write_bytes(data)
        args = SimpleNamespace(out=directory,cli='colab',session='test',remote='/remote/run')
        with patch.object(collector,'download',side_effect=download), \
             patch.object(collector,'verify_models',return_value={'strict_models':12,'vocabularies':4}), \
             patch.object(collector,'write_results'), \
             patch.object(collector.time,'sleep'), patch('builtins.print'), \
             patch.object(collector.subprocess,'run') as stop:
            try:
                collector.collect(args)
            except ValueError:
                stop.assert_not_called()
                raise
        return stop

    def test_verified_archive_is_extracted_before_runtime_is_stopped(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            stop=self.run_collector(directory,self.fixture())
            self.assertTrue((directory/'summary.json').exists())
            self.assertEqual(json.loads((directory/'local_artifact_verification.json').read_text())['strict_models'],12)
            stop.assert_called_once_with(['colab','stop','-s','test'],check=True,timeout=55)

    def test_wrong_archive_hash_never_extracts_or_stops_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with self.assertRaisesRegex(ValueError,'size/hash mismatch'):
                self.run_collector(directory,self.fixture(),wrong_hash=True)
            self.assertFalse((directory/'summary.json').exists())

    def test_archive_path_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'output'
            with self.assertRaisesRegex(ValueError,'Unsafe archive member'):
                self.run_collector(directory,self.fixture(unsafe=True))
            self.assertFalse((Path(temp)/'escaped.json').exists())


if __name__=='__main__':
    unittest.main()
