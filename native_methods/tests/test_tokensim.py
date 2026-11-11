"""Native D-local PPD prefill and one-token response/writeback boundaries."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "native_methods/profiles/llama3-8b-h100-tp1.json"
sys.path.insert(0, str(ROOT / "TokenSim"))
spec = importlib.util.spec_from_file_location(
    "token_native_test_backend", ROOT / "TokenSim/TokenSim/research/native_runtime.py"
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class TokenSimTests(unittest.TestCase):
    def test_d_hit_bypasses_full_p_admission_budget(self):
        def pressure(engine, admission, env):
            def hold_p_budgets():
                yield env.timeout(9.0)
                held = []
                for worker in admission.router.prefill_workers:
                    budget = admission.budgets[worker.id]
                    amount = budget.level
                    yield budget.get(amount)
                    held.append((budget, amount))
                yield env.timeout(6.0)
                for budget, amount in held:
                    budget.put(amount)

            env.process(hold_p_budgets())

        rows = [
            dict(
                source_id="0",
                previous_source_id=None,
                conversation=0,
                turn=0,
                context_tokens=64,
                new_tokens=64,
                output_tokens=3,
                arrival_seconds=0.0,
            ),
            dict(
                source_id="1",
                previous_source_id="0",
                conversation=0,
                turn=1,
                context_tokens=80,
                new_tokens=16,
                output_tokens=2,
                arrival_seconds=10.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = backend.execute(
                rows, "PPD", 0, output, probe=pressure, hardware_profile=LEGACY
            )
            self.assertEqual(summary["status"], "completed")
            records = [
                json.loads(s)
                for s in (output / "requests.jsonl").read_text().splitlines()
            ]
            child = next(r for r in records if r["source_id"] == "1")
            self.assertEqual(child["plan_branch"], "decode_local_prefill")
            self.assertEqual(child["native_budget_wait_s"], 0)
            self.assertLess(child["completed_at"], 15)

    def test_burst_and_returning_pins_drain_with_native_capacity(self):
        rows = []
        count = 320
        for index in range(count):
            rows.append(
                dict(
                    source_id=str(index),
                    previous_source_id=None,
                    conversation=index,
                    turn=0,
                    context_tokens=16384,
                    new_tokens=16384,
                    output_tokens=2,
                    arrival_seconds=0.0,
                )
            )
        for index in range(count):
            rows.append(
                dict(
                    source_id=str(count + index),
                    previous_source_id=str(index),
                    conversation=index,
                    turn=1,
                    context_tokens=16400,
                    new_tokens=16,
                    output_tokens=2,
                    arrival_seconds=0.001,
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = backend.execute(rows, "PPD", 0, output, hardware_profile=LEGACY)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed"], len(rows))
            records = [
                json.loads(s)
                for s in (output / "requests.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(row["native_budget_wait_s"] > 0 for row in records))
            self.assertEqual(sum(row["generated"] for row in records), len(rows) * 2)
            self.assertTrue(all(row["arrival_s"] in (0.0, 0.001) for row in records))
            budget = json.loads((output / "native-admission-budget.json").read_text())
            for device in budget["devices"].values():
                self.assertEqual(device["occupied_at_end"], 0)
                self.assertLessEqual(
                    device["peak_budget_blocks"], device["capacity_blocks"]
                )
            final = json.loads((output / "native-end-state.json").read_text())
            self.assertTrue(
                all(
                    not d["running"]
                    and not d["waiting"]
                    and not d["incoming"]
                    and not d["pinned"]
                    for d in final
                )
            )

    def test_response_boundaries_and_direct_d_local_execution(self):
        for method in ("Serial", "PPD"):
            for outputs in (1, 2, 5):
                with self.subTest(
                    method=method, outputs=outputs
                ), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    rows = [
                        dict(
                            source_id="0",
                            previous_source_id=None,
                            conversation=0,
                            turn=0,
                            context_tokens=64,
                            new_tokens=64,
                            output_tokens=3,
                            arrival_seconds=0.0,
                        ),
                        dict(
                            source_id="1",
                            previous_source_id="0",
                            conversation=0,
                            turn=1,
                            context_tokens=80,
                            new_tokens=16,
                            output_tokens=outputs,
                            arrival_seconds=10.0,
                        ),
                    ]
                    summary = backend.execute(
                        rows, method, 0, output, hardware_profile=LEGACY
                    )
                    self.assertEqual(summary["status"], "completed")
                    self.assertEqual(summary["completed"], 2)
                    self.assertGreater(
                        summary["native_counters"]["LLMEngine.run_complete_compute"], 0
                    )
                    records = [
                        json.loads(s) for s in (output / "requests.jsonl").open()
                    ]
                    self.assertEqual([r["generated"] for r in records], [3, outputs])
                    routes = [
                        json.loads(s)
                        for s in (output / "native-route-decisions.jsonl").open()
                    ]
                    child = routes[1]
                    if method == "PPD":
                        self.assertEqual(
                            child["native_method_branch"], "decode_local_prefill"
                        )
                        self.assertEqual(
                            child["native_compute_worker"], child["decode_worker"]
                        )
                        self.assertEqual(
                            records[1]["plan_branch"], "decode_local_prefill"
                        )
                    else:
                        self.assertEqual(
                            child["native_compute_worker"], child["prefill_worker"]
                        )
                        self.assertEqual(records[1]["plan_branch"], "serial")
                    self.assertGreaterEqual(records[1]["completed_at"], 10.0)
                    final = json.loads((output / "native-end-state.json").read_text())
                    self.assertTrue(
                        all(
                            not d["running"]
                            and not d["waiting"]
                            and not d["loading"]
                            and not d["incoming"]
                            and not d["pinned"]
                            for d in final
                        )
                    )


if __name__ == "__main__":
    unittest.main()
