import unittest
from tools.ms2_full_report import clustered_difference


class ClusteredReportTests(unittest.TestCase):
    def test_duplicate_queries_preserve_point_estimate_and_clusters(self):
        report=clustered_difference([1,1,0],[0,0,0],['same','same','other'],repeats=100)
        self.assertEqual(report['connectivity_groups'],2)
        self.assertAlmostEqual(report['difference'],2/3)
        self.assertTrue(report['ci95'][0]<=2/3<=report['ci95'][1])

    def test_mismatched_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            clustered_difference([1],[],['a'])


if __name__=='__main__':
    unittest.main()
