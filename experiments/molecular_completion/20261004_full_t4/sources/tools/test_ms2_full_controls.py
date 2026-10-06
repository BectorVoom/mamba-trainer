"""Observable behavior and capacity checks for the full-pipeline controls."""
import unittest
import torch

from tools.ms2_full_model import CompletionModel, batch_inputs, prep_peaks
from tools.ms2_full_controls import CausalTransformerBlock, EmptyPredictor
import tools.ms2_full_experiment as experiment
import numpy as np


class FullControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_transformer_capacity_and_causal_prefix(self):
        model = CompletionModel()
        layer = CausalTransformerBlock().eval()
        difference = abs(sum(p.numel() for p in model.layers.parameters()) - sum(p.numel() for p in layer.parameters()))
        self.assertLess(difference / sum(p.numel() for p in model.parameters()), 0.01)
        x = torch.randn(2, 20, 96)
        changed = x.clone()
        changed[:, 12:] += 100
        with torch.no_grad():
            torch.testing.assert_close(layer(x)[:, :12], layer(changed)[:, :12], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(layer(x)[:, :12], layer(x[:, :12]), atol=1e-5, rtol=1e-5)

    def test_empty_predictor_returns_no_patterns(self):
        features, mask, bins = prep_peaks([(20, 1)], 47.049141)
        record = {"peaks":features, "mask":mask, "bins":bins, "meta":[0]*5, "adduct":1}
        result = EmptyPredictor()(batch_inputs([record], "cpu"))
        self.assertEqual(tuple(result.shape), (1, 0))

    def test_retrieval_leaves_isotope_only_identity_unscored(self):
        experiment.DEVICE = torch.device("cpu")
        model, prior = CompletionModel().eval(), CompletionModel().eval()
        ranker = experiment.FeatureRanker(np.zeros(5), np.ones(5)).eval()
        features, mask, bins = prep_peaks([(20, 1)], 47.049141)
        record = {"peaks":features, "mask":mask, "bins":bins,
                  "meta":np.zeros(5, dtype=np.float32), "adduct":1, "predicted":[]}
        query = {"qid":"isotope-regression", "pool":["[13CH3]CO", "COC"], "row":{"smiles":"CCO"}}
        row = experiment.retrieval_prediction(model, prior, ranker, query, record)
        self.assertEqual(row["pool"], 2)
        self.assertEqual(row["scored"], 1)
        self.assertEqual(row["rank"], 2)


if __name__ == "__main__":
    unittest.main()
