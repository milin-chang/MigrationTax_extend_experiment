"""Operation semantics independent of analytical completion-time formulas."""

import unittest
from algorithm.native import build_plan
from algorithm.model import Turn, CacheState


class PlanTests(unittest.TestCase):
    def test_no_duration_fields_and_all_methods_states(self):
        for state in CacheState:
            for method in ["Serial", "PPD", "MigrationTax"]:
                plan = build_plan(method, Turn(80, 16, state, 5))
                for op in plan.operations:
                    self.assertFalse(
                        any(
                            "second" in f or "duration" in f
                            for f in op.__dataclass_fields__
                        )
                    )
                self.assertEqual(plan.operations[-2].tokens, 84)

    def test_hf_output_boundaries_and_barrier(self):
        for outputs in [1, 2, 7]:
            p = build_plan(
                "MigrationTax", Turn(80, 16, CacheState.HF, outputs), handoff_outputs=2
            )
            ops = {o.id: o for o in p.operations}
            k = min(2, outputs)
            self.assertEqual(ops["incremental_sync"].tokens, k - 1)
            self.assertEqual(ops["remaining"].tokens, outputs - k)
            self.assertEqual(ops["remaining"].context_tokens, 80 + k - 1)
            self.assertIn("base_handoff", ops["incremental_sync"].after)
            self.assertIn("p_continue", ops["incremental_sync"].after)
            self.assertEqual(ops["remaining"].after, ("ownership",))

    def test_layer_dependency_graph(self):
        p = build_plan("MigrationTax", Turn(80, 16, CacheState.FF, 5), layers=80)
        ops = {o.id: o for o in p.operations}
        self.assertEqual(ops["fetch_1"].after, ("fetch_0",))
        self.assertEqual(ops["prefill_1"].after, ("fetch_1", "prefill_0"))
        self.assertEqual(ops["send_1"].after, ("prefill_1", "send_0"))
        self.assertEqual(ops["first_output"].after, ("send_79",))
