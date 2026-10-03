"""Out-of-pool generation demo: bounded enumeration recovers a pool miss.

The database plan evaluates generation separately on cases where a
reference pool misses the target. This demo uses the tiny synthetic
domain only: a pool containing just dimethyl ether is queried for
ethanol (C2H6O + hydroxyl evidence). Retrieval correctly reports the
target absent; the independent bounded enumerator
(tools/ms2_completion_ambiguity, C/H/O, <=4 heavy atoms) then recovers
ethanol exactly. No claim beyond this restricted domain.

Run:   python3 tools/ms2_oop_enumerate.py
Test:  python3 -m unittest tools.ms2_oop_enumerate
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_completion_ambiguity import (
    CH2_OH,
    mass,
    query as enumerate_query,
)
from tools.ms2_database_retrieval import (
    DatabaseIndex,
    WorkCounters,
    run_query,
    standardize_record,
)


def _record(mid, atom_types, edges):
    rec, reason = standardize_record(
        {"id": mid, "source": "demo-pool", "family": "demo",
         "atom_types": tuple(atom_types), "edges": tuple(edges), "charge": 0})
    assert rec is not None, reason
    return rec


C3 = ("C", 3, 4)
C2 = ("C", 2, 4)
O1 = ("O", 1, 2)
O0 = ("O", 0, 2)


def demo():
    # Pool holds only dimethyl ether; the query target is ethanol.
    pool = [_record("POOL-DME", (C3, O0, C3), ((0, 1, 1), (1, 2, 1)))]
    index = DatabaseIndex(pool)
    ethanol = _record("TARGET-ETHANOL", (C3, C2, O1), ((0, 1, 1), (1, 2, 1)))
    observed = mass({"C": 2, "H": 6, "O": 1})
    res = run_query(index, ethanol, (CH2_OH,), "unknown", None,
                    counters=WorkCounters())
    retrieval = {
        "n_s3": len(res["s3"]),
        "in_pool": res["in_pool"],
        "recall_s3": res["recall"]["s3"],
        "zero_status": res["zero_status"],
    }
    # Generation arm: bounded enumeration over mass + hydroxyl evidence.
    status, n_formulas, n_graphs, _ = enumerate_query(observed, (CH2_OH,))
    return {"retrieval": retrieval,
            "enumeration": {"status": status, "formulas": n_formulas,
                            "graphs": n_graphs},
            "target_recovered_by_enumeration": status == "complete" and n_graphs == 1}


def main():
    import json

    sys.stdout.write(json.dumps(demo(), indent=2) + "\n")


class OopDemoTests(unittest.TestCase):
    def test_miss_then_recover(self):
        out = demo()
        # Retrieval is honest about the miss...
        self.assertFalse(out["retrieval"]["in_pool"])
        self.assertIsNone(out["retrieval"]["recall_s3"])
        self.assertEqual(out["retrieval"]["zero_status"], "target_absent")
        self.assertEqual(out["retrieval"]["n_s3"], 0)
        # ...and bounded enumeration recovers the target in-domain.
        self.assertTrue(out["target_recovered_by_enumeration"])
        self.assertEqual(out["enumeration"]["graphs"], 1)


if __name__ == "__main__":
    main()
