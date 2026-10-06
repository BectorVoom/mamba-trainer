"""Recovery must verify immutable original training sources."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.ms2_full_recover import verify_training_sources


class RecoveryTests(unittest.TestCase):
    def test_verified_training_hash_retained_and_modified_sources_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            source=base/'training_sources/tools'
            source.mkdir(parents=True)
            manifest={}
            for name, text in [('ms2_full_experiment.py','training code'),('ms2_full_model.py','model code')]:
                (source/name).write_text(text)
                manifest['tools/'+name]=hashlib.sha256(text.encode()).hexdigest()
            (base/'training_source_sha256.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_training_sources(base),hashlib.sha256(b'training codemodel code').hexdigest())
            (source/'ms2_full_model.py').write_text('modified')
            with self.assertRaisesRegex(ValueError,'Training source hash mismatch'):
                verify_training_sources(base)

    def test_unsafe_manifest_path_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            (base/'training_source_sha256.json').write_text(json.dumps({'../escape':'invalid'}))
            with self.assertRaisesRegex(ValueError,'Unsafe training source path'):
                verify_training_sources(base)


if __name__ == '__main__':
    unittest.main()
