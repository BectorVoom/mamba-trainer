"""Supervisor regressions using cached source data; no network or tuning."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools import ms2_external_validation as ext
from tools.ms2_chebi_corpus import formula_key
from tools.ms2_database_retrieval import DatabaseIndex
from tools.ms2_msgym_corpus import standardize_smiles


class IndependentRegressions(unittest.TestCase):
    def test_actual_pubchem_response_schema(self):
        cache = ROOT / 'experiments/molecular_completion/20261003_remaining/pubchem_cache'
        response = json.loads((cache / 'smiles_C13H16O4_0.json').read_text())
        expected = response['PropertyTable']['Properties'][0]['ConnectivitySMILES']
        with patch.object(ext, '_pug_get_json', side_effect=AssertionError('network forbidden')) as remote:
            results, stats = ext.pubchem_expand(['C13H16O4'], str(cache))
        self.assertEqual(stats['api_calls'], 0)
        self.assertTrue(stats['cache_only'])
        self.assertEqual(len(results['C13H16O4']['smiles']), 100)
        self.assertTrue(all(results['C13H16O4']['smiles']))
        self.assertIn(expected, results['C13H16O4']['smiles'])
        remote.assert_not_called()

    def test_missing_pubchem_cache_never_requests(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(ext, '_pug_get_json', side_effect=AssertionError('network forbidden')) as remote:
                results, stats = ext.pubchem_expand(['C2H6O'], cache)
        self.assertIn('cached CID response unavailable', results['C2H6O']['error'])
        self.assertEqual(stats['api_calls'], 0)
        remote.assert_not_called()

    def stage(self, candidate, measured_shift=0, precision=True):
        q, reason = standardize_smiles('CCO', 'supervisor', 'Q')
        self.assertIsNone(reason)
        r, reason = standardize_smiles(candidate, 'supervisor', 'DB')
        self.assertIsNone(reason)
        r['canon'] = ext.rdkit_canon(candidate)[0]
        return ext.chebi_stage_query('Q', q, 'CCO', q['mass'] + measured_shift,
                                     precision, DatabaseIndex([r]),
                                     {formula_key(r['formula']): [r]},
                                     uncertainty_mu=100 if precision else None)

    def test_absent_identity_misses_every_stage(self):
        s = self.stage('COC')
        self.assertTrue(s['nested_ok'])
        self.assertFalse(s['stage1']['target_present_s1'])
        self.assertFalse(s['stage2']['target_present_s2'])
        self.assertFalse(s['stage3_oracle']['target_present_s3'])

    def test_mass_rejection_persists_through_formula_filter(self):
        s = self.stage('CCO', measured_shift=1_000_000)
        self.assertEqual(s['stage1']['n_s1'], 0)
        self.assertEqual(s['stage2']['n_s2'], 0)
        self.assertEqual(s['stage3_oracle']['n_s3'], 0)
        self.assertFalse(s['stage2']['target_present_s2'])

    def test_unknown_precision_preserved(self):
        s = self.stage('CCO', measured_shift=1_000_000, precision=False)
        self.assertIsNone(s['uncertainty_mu'])
        self.assertEqual(s['stage1']['n_unavailable'], 1)
        self.assertTrue(s['stage2']['target_present_s2'])
        self.assertTrue(s['stage3_oracle']['target_present_s3'])

    def test_charge_sign_conflict_rejected(self):
        adduct, reason = ext.gnps_adduct_of({'params': {
            'NAME': 'X M+H', 'IONMODE': 'positive', 'CHARGE': '1-'}})
        self.assertIsNone(adduct)

    def test_massbank_peak_count_enforced(self):
        record, reason = ext.parse_massbank_record(ext._MB_MINIMAL.replace(
            'PK$NUM_PEAK: 3', 'PK$NUM_PEAK: 999'))
        self.assertIsNone(record)
        self.assertIsNotNone(reason)


if __name__ == '__main__':
    unittest.main()
