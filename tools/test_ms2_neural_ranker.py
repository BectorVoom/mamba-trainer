"""Behavior tests for the standalone neural ranking pilot."""
import unittest
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from tools.ms2_neural_ranker import (Ranker, bootstrap_difference, evaluate,
                                     graph, loss, metadata, order_scores,
                                     pack_graphs, spectrum_features)
from tools.ms2_neural_ranker import run
from tools.ms2_spectral_rank import WANTED_COLUMNS


class RankerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_peak_contract_and_errors(self):
        vec = spectrum_features([(10, 4), (20, 1), (30, 0.001), (103, 4)], 100)
        self.assertAlmostEqual(float(vec[10] / vec[20]), 2)
        self.assertEqual(vec[30], 0)
        self.assertEqual(vec[103], 0)
        self.assertEqual(float(np.linalg.norm(spectrum_features([], 100))), 0)
        with self.assertRaises(ValueError):
            spectrum_features([(1, -1)], 100)

    def test_energy_missing_is_distinct_from_zero(self):
        base = {"precursor_mz": "100", "adduct": "[M+H]+"}
        missing, _ = metadata(base, {})
        zero, _ = metadata(dict(base, collision_energy="0"), {})
        normalized, _ = metadata(dict(base, collision_energy="20%"), {})
        self.assertEqual(missing[3], 0)
        self.assertEqual(zero[3], 1)
        self.assertEqual(normalized[3], 0)
        self.assertEqual(missing[4], 0)
        with self.assertRaises(ValueError):
            metadata(dict(base, energy_count="9"), {})

    def test_graph_permutation_and_padding(self):
        model = Ranker(2).eval()
        x, a = graph("CCO")
        p = [2, 0, 1]
        permuted = (x[p], a[:, p][:, :, p])
        with torch.no_grad():
            single = model.encode_graph(pack_graphs([(x, a)], "cpu"))[0]
            together = model.encode_graph(pack_graphs([permuted, graph("CCCCCC")], "cpu"))[0]
        torch.testing.assert_close(single, together, rtol=1e-5, atol=1e-5)

    def test_aromatic_graphs_are_encodable(self):
        model = Ranker(2).eval()
        aromatic = graph("c1ccccc1")
        self.assertIsNotNone(aromatic)
        self.assertEqual(float(aromatic[1][3].sum()), 12)
        with torch.no_grad():
            encoded = model.encode_graph(pack_graphs([aromatic, graph("CCO")], "cpu"))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_tail_preserves_candidates(self):
        self.assertEqual(order_scores([None, -2, 0, None], {0: 3, 1: 2, 2: 1, 3: 0}), [2, 1, 3, 0])
        model = Ranker(2).eval()
        pools = {"q": {"distinct": ["CCO", "[Na+]"], "smiles": "[Na+]",
                       "row": {"mzs": "10,20", "intensities": "0,0",
                               "precursor_mz": "100", "adduct": "[M+H]+"}}}
        result = evaluate(model, pools, {}, "cpu", 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["n_pool"], 2)
        self.assertEqual(result[0]["n_scored"], 0)
        self.assertIsNotNone(result[0]["rank"])

    def training_check(self, device):
        molecules = []
        for i, smiles in enumerate(("CCO", "COC", "CCC", "CCN")):
            molecules.append({"smiles": smiles, "group": str(i),
                              "peaks": [(10 + 20 * i, 1), (15 + 20 * i, 0.5)],
                              "row": {"precursor_mz": "100", "adduct": "[M+H]+"}})
        model = Ranker(2, width=32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        initial = loss(model, molecules, {"[M+H]+": 1}, device, 0.2).item()
        for _ in range(40):
            optimizer.zero_grad()
            value = loss(model, molecules, {"[M+H]+": 1}, device, 0.2)
            value.backward()
            optimizer.step()
        final = loss(model, molecules, {"[M+H]+": 1}, device, 0.2).item()
        self.assertLess(final, initial * 0.5)

    def test_cpu_training_learns(self):
        self.training_check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_training_and_cpu_parity(self):
        self.training_check("cuda")
        cpu = Ranker(2).eval()
        gpu = Ranker(2).cuda().eval()
        gpu.load_state_dict(cpu.state_dict())
        graphs = [graph("CCO"), graph("COC")]
        with torch.no_grad():
            torch.testing.assert_close(cpu.encode_graph(pack_graphs(graphs, "cpu")),
                                       gpu.encode_graph(pack_graphs(graphs, "cuda")).cpu(), rtol=1e-4, atol=1e-4)

    def test_paired_bootstrap(self):
        result = bootstrap_difference([1, 0], [0, 0], 42)
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result, bootstrap_difference([1, 0], [0, 0], 42))

    def test_file_backed_training_and_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tsv, prefix = root / "data.tsv", root / "pools.json"
            smiles = ("CCO", "COC", "CCC", "CCN", "CCCO", "CCOC", "CCCC", "CCCN", "CCNC", "CC(C)O", "CN")
            with tsv.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=WANTED_COLUMNS, delimiter="\t")
                writer.writeheader()
                for i, smi in enumerate(smiles):
                    row = dict.fromkeys(WANTED_COLUMNS, "")
                    row.update(identifier=str(i), mzs=f"{10 + i},{30 + i}", intensities="1,0.5",
                               smiles=smi, inchikey=f"{i:014d}-TEST", precursor_mz="100",
                               adduct="[M+H]+", fold="val" if i == 10 else "train")
                    writer.writerow(row)
            prefix.write_text(json.dumps({"CN": ["CCN", "CN", "[Na+]"]}))
            args = argparse.Namespace(tsv=str(tsv), prefix=str(prefix), out=str(root / "output"),
                                      device="cpu", require_t4=False, epochs=1, batch_size=4,
                                      max_train=10, max_queries=1, seed=42)
            run(args)
            summary = json.loads((root / "output/summary.json").read_text())
            predictions = json.loads((root / "output/predictions.json").read_text())
            self.assertEqual(summary["fit"], 8)
            self.assertEqual(summary["calibration"], 2)
            self.assertEqual(predictions["contrastive"][0]["n_pool"], 3)
            self.assertEqual(predictions["contrastive"][0]["n_scored"], 2)
            self.assertTrue((root / "output/contrastive.pt").exists())


if __name__ == "__main__":
    unittest.main()
