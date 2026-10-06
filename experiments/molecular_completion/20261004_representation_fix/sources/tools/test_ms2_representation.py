"""Regression checks for distinctions lost by the original MS2 representation."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools import ms2_full_experiment as experiment
from tools.ms2_full_model import *


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_subdalton_forward_targets_remain_distinct(self):
        a = prep_peaks([(20.101, 1)], 50)[2]
        b = prep_peaks([(20.109, 1)], 50)[2]
        np.testing.assert_array_equal(a[:2000], b[:2000])
        self.assertLess(float(a @ b), 0.8)
        np.testing.assert_array_equal(a, prep_peaks([(20.101, 10)], 50)[2])
        self.assertEqual(len(a), SPECTRUM_DIM)
        with self.assertRaisesRegex(ValueError, 'finite'):
            prep_peaks([(20, float('nan'))], 50)

    def test_regular_graphs_with_identical_local_messages_remain_distinct(self):
        # Triangular prism and K3,3: same six C-H atoms, degree and bond counts.
        carbon = TYPE_ID[('C', 1, 4)]
        prism = Graph((carbon,) * 6, ((0,1,1),(1,2,1),(0,2,1),
                      (3,4,1),(4,5,1),(3,5,1),(0,3,1),(1,4,1),(2,5,1)))
        bipartite = Graph((carbon,) * 6, tuple((a,b,1) for a in range(3) for b in range(3,6)))
        tensors = graph_tensors([prism, bipartite], 'cpu')
        encoder = GraphEncoder(96).eval()
        with torch.no_grad():
            encoded = encoder(*tensors)
            # Original pooled branch gives exactly the same representation.
            pooled = encoded - encoder.canonical(tensors[2])
            torch.testing.assert_close(pooled[0], pooled[1], atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(encoded[0], encoded[1]))
        permutation = [5,2,4,0,3,1]
        index = {old:new for new,old in enumerate(permutation)}
        moved = Graph(tuple(prism.types[i] for i in permutation),
                      tuple((index[a],index[b],o) for a,b,o in prism.edges))
        np.testing.assert_array_equal(canonical_graph_features(prism), canonical_graph_features(moved))

    def test_stereo_annotations_survive_connectivity_grouping(self):
        for left, right in [('C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O'),
                            ('C/C=C/C', 'C/C=C\\C')]:
            a, reason = parse_graph(left)
            b, other = parse_graph(right)
            self.assertIsNone(reason)
            self.assertIsNone(other)
            self.assertEqual(a.identity, b.identity)
            self.assertNotEqual(a.stereo_smiles, b.stereo_smiles)
            self.assertFalse(np.array_equal(canonical_graph_features(a), canonical_graph_features(b)))
            encoder = GraphEncoder(96).eval()
            with torch.no_grad():
                encoded = encoder(*graph_tensors([a,b], 'cpu'))
            self.assertFalse(torch.allclose(encoded[0], encoded[1]))

    def test_distinct_measurement_conditions_retained_without_identity_leakage(self):
        fields = ['fold','smiles','mzs','intensities','precursor_mz','adduct','collision_energy']
        def row(fold, smiles, peak, energy):
            return dict(zip(fields, [fold,smiles,str(peak),'1','47.049141','[M+H]+',energy]))
        rows = [row('train','CCO',20,'10'), row('train','CCO',20,'20'),
                row('train','CCO',20,'20'), row('train','COC',21,'10'),
                row('val','CCN',22,'10'), row('test','CCC',23,'10')]
        table = SimpleNamespace(hypotheses=lambda r: [parse_graph(r['smiles'])[0].counts])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.tsv'
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fields, delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            args = SimpleNamespace(tsv=str(path), prefix='unused', max_train=2, calibration=1, queries=1)
            with patch.object(experiment, 'iter_json_entries', return_value=iter([('CCC',['CCC'])])):
                fit, calibration, queries, audit, forbidden, _ = experiment.load_data(args, table)
        self.assertEqual(len(fit), 3)
        self.assertEqual(sum(r['identity']=='CCO' for r in fit), 2)
        self.assertEqual(audit['train_duplicate'], 1)
        self.assertFalse({r['identity'] for r in fit} & forbidden)

    def paired_training_check(self, device):
        previous_device = experiment.DEVICE
        self.addCleanup(setattr, experiment, 'DEVICE', previous_device)
        experiment.DEVICE = torch.device(device)
        a, _ = parse_graph('CCO')
        b, _ = parse_graph('COC')
        peaks, mask, target = prep_peaks([(20.101,1),(31.02,.5)], 47.049141)
        record = dict(graph=a, pair_graphs=[a,b], peaks=peaks, mask=mask, bins=target,
                      meta=np.zeros(5,dtype=np.float32), adduct=1, predicted=[])
        model = CompletionModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        initial = experiment.paired_candidate_loss(model, [record])
        initial.backward()
        for module in (model.spectrum_encoder, model.graph_encoder, model.layers):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                   if p.grad is not None), 0)
        before = initial.item()
        optimizer.step()
        for _ in range(11):
            optimizer.zero_grad()
            loss = experiment.paired_candidate_loss(model, [record])
            loss.backward()
            optimizer.step()
        self.assertLess(experiment.paired_candidate_loss(model, [record]).item(), before * .5)

    def test_cpu_same_formula_training_updates_both_encoders_and_decoder(self):
        self.paired_training_check('cpu')

    def test_old_checkpoints_require_explicit_retraining(self):
        from tools.ms2_full_controls import restore
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'old.pt'
            torch.save({'state': {}}, path)
            with self.assertRaisesRegex(ValueError, 'incompatible representation; retrain'):
                restore(GraphEncoder(96), path)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_same_formula_training_updates_both_encoders_and_decoder(self):
        self.paired_training_check('cuda')


if __name__ == '__main__':
    unittest.main()
