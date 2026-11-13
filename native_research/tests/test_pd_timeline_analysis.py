"""Keep gate attribution and shared-batch diagnostic denominators explicit."""

import csv
import tempfile
import unittest
from pathlib import Path

from native_research.pd_timeline_analysis import (
    DELTAS,
    STEP_PARTS,
    describe,
    request_intervals,
    stream_steps,
)


class TimelineAnalysisTests(unittest.TestCase):
    def request(self):
        row = {key: "" for fields in DELTAS.values() for key in fields}
        row.update(
            source_id="a",
            plan_branch="layer_pipeline",
            admission_started_s="2",
            p_budget_acquired_s="5",
            d_budget_acquired_s="7",
            submitted_s="7",
            first_client_output_at_s="8",
            parent_wait_s="2",
            admission_wait_s="5",
            ttft_s="8",
        )
        return row

    def test_p_then_d_gate_sum(self):
        result = request_intervals(self.request())
        self.assertEqual(result["gate_p_wait"], 3)
        self.assertEqual(result["gate_d_after_p_wait"], 2)
        self.assertEqual(result["after_admission"], 1)
        self.assertIsNone(result["d_queue_to_output"])

    def test_d_only_has_no_fabricated_p_observation(self):
        row = self.request()
        row.update(plan_branch="decode_local_prefill", p_budget_acquired_s="")
        result = request_intervals(row)
        self.assertEqual(result["gate_p_wait"], 0)
        self.assertEqual(result["gate_d_after_p_wait"], 5)

    def test_missing_or_reversed_budget_is_rejected(self):
        row = self.request()
        row["p_budget_acquired_s"] = ""
        with self.assertRaisesRegex(ValueError, "required P"):
            request_intervals(row)
        row = self.request()
        row["d_budget_acquired_s"] = "4"
        with self.assertRaises(ValueError):
            request_intervals(row)

    def test_first_output_differs_from_first_decode_end(self):
        row = self.request()
        row.update(
            d_queue_enter_s="7",
            d_scheduler_enter_s="7.1",
            d_first_compute_start_s="7.2",
            first_d_output_at_s="8",
            d_first_decode_step_end_s="9",
        )
        result = request_intervals(row)
        self.assertEqual(result["d_queue_to_output"], 1)
        self.assertEqual(result["d_queue_to_decode_end"], 2)

    def test_exact_quantiles_and_empty_population(self):
        result = describe([4, 1, 3, 2])
        self.assertEqual(result, dict(n=4, mean=2.5, p50=2.5, p95=4, maximum=4))
        self.assertIsNone(describe([])["mean"])

    def test_steps_are_weighted_once_per_batch(self):
        rows = []
        for role in ("prefill", "decode"):
            for size, service in ((1, 1), (3, 3)):
                row = {key: "0" for key in STEP_PARTS}
                row.update(
                    role=role,
                    status="completed",
                    batch_size=str(size),
                    batch_tokens=str(size * 10),
                    scheduled_at_s="0",
                    step_completed_at_s=str(service),
                    compute_service_s=str(service),
                    request_ids='["a"]' if size == 1 else '["a", "b", "c"]',
                )
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary, histogram, samples = stream_steps(path, {"tp": 1}, "a")
        for row in summary:
            self.assertEqual(row["steps"], 2)
            self.assertEqual(row["compute_service_s_mean"], 2)
            self.assertEqual(row["batch_size_mean"], 2)
        self.assertEqual(sum(r["steps"] for r in histogram), 4)
        self.assertEqual(len(samples), 4)


if __name__ == "__main__":
    unittest.main()
