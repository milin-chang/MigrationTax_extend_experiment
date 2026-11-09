"""Check report populations, output-event timing and refusal of partial data."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from native_research.report_native_run_markdown import describe, splitwise_tpot


class OutputTimingTests(unittest.TestCase):
    def test_tpot_excludes_writeback_and_keeps_single_output_zero(self):
        requests = {
            "0": dict(generated=3, first_at_s=1.0, completed_at_s=99.0),
            "1": dict(generated=1, first_at_s=2.0, completed_at_s=88.0),
        }
        events = [
            dict(source_id="0", source_outputs_emitted=1, completed_at_s=1.0),
            dict(source_id="1", source_outputs_emitted=1, completed_at_s=2.0),
            dict(source_id="0", source_outputs_emitted=2, completed_at_s=3.0),
            dict(source_id="1", source_outputs_emitted=0, completed_at_s=88.0),
            dict(source_id="0", source_outputs_emitted=0, completed_at_s=99.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with gzip.open(output / "native-nodes.jsonl.gz", "wt") as stream:
                for event in events:
                    stream.write(json.dumps(event) + "\n")
            values, _ = splitwise_tpot(output, requests)
        self.assertEqual(values, [1.0, 0.0])
        self.assertEqual(describe(values)["mean"], 0.5)

    def test_missing_output_emission_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with gzip.open(output / "native-nodes.jsonl.gz", "wt") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            source_id="0", source_outputs_emitted=1, completed_at_s=1.0
                        )
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "emission count"):
                splitwise_tpot(
                    output,
                    {"0": dict(generated=3, first_at_s=1.0, completed_at_s=99.0)},
                )

    def test_p95_uses_nearest_rank(self):
        values = list(range(1, 21))
        self.assertEqual(describe(values)["p95_nearest_rank"], 19)
        with self.assertRaises(ValueError):
            describe([0.1, float("nan")])


if __name__ == "__main__":
    unittest.main()
