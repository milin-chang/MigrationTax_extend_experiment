"""Protect exact pairing, strict thresholds and missing-D-sample accounting."""

from copy import deepcopy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from native_research.report_five_stage_links import paired_rows
from native_research.run_five_stage_links import sanity, controller_directory
from native_research.five_stage_order import batches


def row(source_turn, d, client=None):
    return dict(
        arrival_s=1.0,
        source_turn=source_turn,
        generated=2,
        d_output_sampled_latency_s=d,
        native_prefill_latency_s=client if client is not None else d,
    )


class PairingTests(unittest.TestCase):
    def test_each_batch_has_an_exclusive_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for batch in ("00-sanity", "01-mooncake-fast", "02-mooncake-pcie6"):
                controller = controller_directory(Path(directory), batch)
                controller.parent.mkdir(parents=True)
                with (controller.parent / "controller.log").open("x") as log:
                    log.write(batch)

    def test_all_mooncake_jobs_precede_every_servegen_job(self):
        jobs = []
        for phase, trace, count in [
            (0, "mooncake", 2),
            (1, "mooncake", 4),
            (2, "servegen", 4),
            (3, "mooncake", 3),
            (3, "servegen", 3),
            (4, "mooncake", 3),
            (4, "servegen", 3),
        ]:
            jobs.extend(
                dict(id=f"{phase}-{trace}-{n}", phase=phase, trace=trace)
                for n in range(count)
            )
        order = batches(jobs)
        traces = [batch["trace"] for batch in order for _ in batch["job_ids"]]
        self.assertEqual(traces, ["mooncake"] * 12 + ["servegen"] * 10)
        self.assertEqual([batch["count"] for batch in order], [2, 4, 3, 3, 4, 3, 3])
        with self.assertRaises(ValueError):
            batches(jobs[:-1])

    def test_initial_turn_excluded_and_exact_two_times_not_affected(self):
        reference = {"0": row(0, 1.0), "1": row(1, 1.0), "2": row(1, 1.0)}
        actual = {"0": row(0, 4.0), "1": row(1, 2.0), "2": row(1, 2.0001)}
        paired = paired_rows(actual, reference)
        self.assertFalse(paired[0]["returning"])
        self.assertIsNone(paired[0]["affected_d"])
        self.assertFalse(paired[1]["affected_d"])
        self.assertTrue(paired[2]["affected_d"])

    def test_missing_d_sample_is_retained_and_not_replaced_by_client_time(self):
        reference = {"1": row(1, 1.0)}
        actual = {"1": row(1, None, client=3.0)}
        paired = paired_rows(actual, reference)
        self.assertEqual(len(paired), 1)
        self.assertTrue(paired[0]["returning"])
        self.assertTrue(paired[0]["d_sample_missing"])
        self.assertIsNone(paired[0]["affected_d"])
        self.assertTrue(paired[0]["affected_client"])

    def test_workload_mismatch_rejected(self):
        reference = {"1": row(1, 1.0)}
        actual = deepcopy(reference)
        actual["1"]["arrival_s"] = 2.0
        with self.assertRaisesRegex(ValueError, "workload differs"):
            paired_rows(actual, reference)
        with self.assertRaisesRegex(ValueError, "Source IDs"):
            paired_rows({}, reference)

    def test_reference_requires_real_d_sample(self):
        with self.assertRaisesRegex(ValueError, "reference lacks"):
            paired_rows({"1": row(1, 1.0)}, {"1": row(1, None, client=1.0)})

    def test_sanity_stops_when_slow_link_does_not_delay_the_mean(self):
        def record(link, value):
            return dict(
                phase=0,
                link=link,
                input_sha256="same",
                mean_d_first_observed_s=value,
                rows={"0": row(0, value)},
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "native_research.run_five_stage_links.collect",
                return_value=[record("nvl4", 1.0), record("ib400", 2.0)],
            ):
                sanity(Path("unused"), Path(directory))
            with patch(
                "native_research.run_five_stage_links.collect",
                return_value=[record("nvl4", 1.0), record("ib400", 0.5)],
            ):
                with self.assertRaisesRegex(RuntimeError, "did not increase"):
                    sanity(Path("unused"), Path(directory))


if __name__ == "__main__":
    unittest.main()
