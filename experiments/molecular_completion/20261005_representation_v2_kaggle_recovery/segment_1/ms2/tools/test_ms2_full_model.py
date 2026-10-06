import json
import os
import subprocess
import unittest

import numpy as np
import torch

from tools.ms2_full_model import *


class FullModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_roundtrip_closed_graph_and_early_stop(self):
        for smiles in ("CC", "CCO", "c1ccccc1", "CC(=O)N", "C1CC1", "CCCCCCCCCCCCCCCCCC"):
            graph, reason = parse_graph(smiles)
            self.assertIsNone(reason)
            state = CompleteState(graph.counts)
            for index, token in enumerate(graph.trace()):
                if index == 2:
                    self.assertFalse(state.masks((STOP, 0, 0, 0))[1])
                state.apply(token)
            self.assertEqual(state.graph().identity, graph.identity)
            self.assertFalse(any(state.residual))
            teacher_batch([graph], "cpu")

    def test_original_isotopes_charges_and_radicals_are_rejected(self):
        for smiles in ("[13CH3]CO", "[NH3+]CC", "[CH2]C"):
            graph, reason = parse_graph(smiles)
            self.assertIsNone(graph)
            self.assertEqual(reason, "charged_isotopic_or_radical")

    def test_candidate_domain_checks_precede_identity_deduplication(self):
        isotope_only = canonical_candidates(["[13CH3]CO"])
        self.assertEqual(isotope_only, {"CCO": None})
        both = canonical_candidates(["[13CH3]CO", "OCC"])
        self.assertEqual(list(both), ["CCO"])
        self.assertIsNotNone(both["CCO"])

    def test_graph_encoder_invariant_and_parent_hydrogens(self):
        graph, _ = parse_graph("CCO")
        motifs_before = motifs(graph)
        perm = [2, 0, 1]
        position = {old: new for new, old in enumerate(perm)}
        permuted = Graph(tuple(graph.types[i] for i in perm), tuple((position[a], position[b], o) for a,b,o in graph.edges))
        self.assertEqual(motifs_before, motifs(permuted))
        encoder = GraphEncoder(96).eval()
        with torch.no_grad():
            torch.testing.assert_close(encoder(*graph_tensors([graph], "cpu")), encoder(*graph_tensors([permuted], "cpu")))
        self.assertIn(((9,), ()), motifs_before)  # Parent OH, not isolated water.

    def test_mass_hypotheses_without_target_labels(self):
        table = FormulaTable()
        row = {"precursor_mz": "47.049141", "adduct": "[M+H]+"}
        self.assertIn((2,6,0,1), table.hypotheses(row))
        with self.assertRaises(ValueError):
            table.hypotheses({"precursor_mz":"47", "adduct":"unknown"})

    def test_mamba_causality_and_prefix_equivalence(self):
        layer = block(96).eval()
        x = torch.randn(2, 20, 96)
        altered = x.clone()
        altered[:, 12:] += 100
        with torch.no_grad():
            full = layer(x)
            torch.testing.assert_close(full[:, :12], layer(altered)[:, :12], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(full[:, :12], layer(x[:, :12]), atol=1e-5, rtol=1e-5)

    def test_substructure_order_invariance(self):
        model = CompletionModel().eval()
        graph, _ = parse_graph("CCO")
        patterns = [(key, 0.7) for key in sorted(motifs(graph))[:5]]
        with torch.no_grad():
            a = substructure_context(model, [{"predicted":patterns}], "cpu")
            b = substructure_context(model, [{"predicted":patterns[::-1]}], "cpu")
        torch.testing.assert_close(a,b)

    def test_inference_condition_does_not_read_reference_graph(self):
        features, mask, bins = prep_peaks([(20, 1)], 47.049141)
        record = {"peaks": features, "mask": mask, "bins": bins,
                  "meta": np.zeros(5, dtype=np.float32), "adduct": 1, "predicted": []}
        model = CompletionModel().eval()
        with torch.no_grad():
            a = model.condition([{**record, "graph": parse_graph("CCO")[0]}], "cpu")
            b = model.condition([{**record, "graph": None}], "cpu")
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)

    def training_check(self, device):
        graph, _ = parse_graph("CC")
        model = CompletionModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        initial = -model.likelihood([graph], [{}], device, False).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = -model.likelihood([graph], [{}], device, False).mean()
            loss.backward()
            optimizer.step()
        final = -model.likelihood([graph], [{}], device, False).item()
        self.assertLess(final, initial * 0.25)
        features, mask, bins = prep_peaks([(20,1)], 31.05)
        record = {"peaks":features, "mask":mask, "bins":bins, "meta":np.zeros(5,dtype=np.float32), "adduct":1, "predicted":[]}
        # Only the prior is trained in this behavior test; condition is zeroed.
        original = model.condition
        model.condition = lambda records, dev, conditional=True: (torch.zeros(len(records),96,device=dev), torch.zeros(len(records),1,96,device=dev), torch.zeros(len(records),1,dtype=torch.bool,device=dev))
        model.eval()
        candidates, work = sample_graphs(model, record, [graph.counts], device, trajectories=2)
        model.condition = original
        self.assertTrue(candidates)
        self.assertTrue(all(c.counts == graph.counts and c.identity == "CC" for c in candidates))

    def test_cpu_training_and_generation(self):
        self.training_check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable locally")
    def test_cuda_training_and_cpu_gpu_parity(self):
        self.training_check("cuda")
        cpu = CompletionModel().eval()
        gpu = CompletionModel().cuda().eval()
        gpu.load_state_dict(cpu.state_dict())
        graphs = [parse_graph("CCO")[0], parse_graph("c1ccccc1")[0]]
        with torch.no_grad():
            a = cpu.likelihood(graphs, [{},{}], "cpu", False)
            b = gpu.likelihood(graphs, [{},{}], "cuda", False).cpu()
        torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
        features, mask, bins = prep_peaks([(20, 1), (31, 0.5)], 47.049141)
        record = {"peaks": features, "mask": mask, "bins": bins,
                  "meta": np.zeros(5, dtype=np.float32), "adduct": 1,
                  "predicted": [(key, 0.7) for key in sorted(motifs(graphs[0]))[:3]]}
        with torch.no_grad():
            a = cpu.likelihood(graphs, [record, record], "cpu")
            b = gpu.likelihood(graphs, [record, record], "cuda").cpu()
            torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
            encoded_cpu = cpu.graph_encoder(*graph_tensors(graphs, "cpu"))
            encoded_gpu = gpu.graph_encoder(*graph_tensors(graphs, "cuda"))
            a = cpu.predict_spectrum(encoded_cpu, batch_inputs([record, record], "cpu"))
            b = gpu.predict_spectrum(encoded_gpu, batch_inputs([record, record], "cuda")).cpu()
            torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)

    @unittest.skipUnless(os.environ.get("MS2_RUST_PARITY_BIN"), "Rust parity executable not configured")
    def test_rust_python_grammar_parity(self):
        cases, expected = [], []
        for smiles in ("CC", "CCO", "c1ccccc1", "CC(=O)N", "C1CC1", "CCCCCCCCCCCCCCCCCC"):
            graph, _ = parse_graph(smiles)
            state = CompleteState(graph.counts)
            masks=[]
            for token in graph.trace():
                masks.append(state.masks(token)[0])
                state.apply(token)
            cases.append({"counts":graph.counts,"trace":graph.trace()})
            expected.append({"masks":masks,"residual":state.residual})
        process = subprocess.run([os.environ["MS2_RUST_PARITY_BIN"]], input=json.dumps(cases), capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(process.stdout), expected)


if __name__ == "__main__":
    unittest.main()
