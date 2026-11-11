"""Actual profiled native batches, first-token placement and cache lifecycle."""

import gzip
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimAI/vidur-alibabacloud"))
from vidur.shared_kv.method_predictor import build_predictor
from vidur.shared_kv.method_workload import make_requests
from vidur.shared_kv.method_observations import Observations
from vidur.shared_kv.method_scheduler import MethodScheduler
from vidur.shared_kv.method_runtime import make_simulator
from vidur.shared_kv.native_runtime import NativeKVEvent


class VidurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.predictor, cls.replica, cls.config = build_predictor(Path(cls.temp.name), 0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_first_output_and_cache_pressure(self):
        for state in ("HH", "FH", "HF", "FF"):
            for outputs, new_tokens in itertools.product((1, 2, 5), (0, 16)):
                for method in ("Serial", "PPD"):
                    with self.subTest(
                        state=state,
                        outputs=outputs,
                        method=method,
                        new_tokens=new_tokens,
                    ), tempfile.TemporaryDirectory() as directory:
                        output = Path(directory)
                        rows = [
                            dict(
                                source_id="0",
                                previous_source_id=None,
                                conversation=0,
                                turn=0,
                                context_tokens=67,
                                new_tokens=67,
                                output_tokens=3,
                                arrival_seconds=0.0,
                            ),
                            dict(
                                source_id="1",
                                previous_source_id="0",
                                conversation=0,
                                turn=1,
                                context_tokens=67 + new_tokens,
                                new_tokens=new_tokens,
                                output_tokens=outputs,
                                arrival_seconds=10.0,
                            ),
                        ]
                        requests = make_requests(rows)
                        obs = Observations(output, requests)
                        scheduler = MethodScheduler(
                            method,
                            requests,
                            obs,
                            self.predictor,
                            self.replica,
                            self.config,
                        )
                        simulator = make_simulator(scheduler, obs)

                        def pressure(time, payload):
                            self.assertIn("0", scheduler.done)
                            for devices, miss in (
                                (scheduler.prefill, state[0] == "F"),
                                (scheduler.decode, state[1] == "F"),
                            ):
                                if miss:
                                    for device in devices:
                                        device.pool.reclaim_unpinned()
                            scheduler.admission_dirty = True

                        scheduler.on_pressure = pressure
                        simulator._add_event(NativeKVEvent(9.0, "pressure"))
                        simulator.run()
                        summary = json.loads((output / "summary.json").read_text())
                        self.assertEqual(summary["status"], "completed")
                        child = obs.rows["1"]
                        self.assertEqual(child["state_at_admission"], state)
                        self.assertEqual(child["completed_output_tokens"], outputs)
                        self.assertEqual(
                            child["native_branch"] == "decode_local_prefill",
                            method == "PPD" and state[1] == "H",
                        )
                        with gzip.open(
                            output / "native-events.jsonl.gz", "rt"
                        ) as stream:
                            samples = [
                                json.loads(line)
                                for line in stream
                                if json.loads(line).get("kind")
                                == "native_sampled_output"
                                and json.loads(line).get("source_id") == "1"
                            ]
                        self.assertEqual(sum(x["produced"] for x in samples), outputs)
                        self.assertTrue(all(x["role"] == "D" for x in samples))
                        self.assertEqual(
                            samples[0]["time_s"], child["client_first_token_s"]
                        )
                        for device in summary["devices"]:
                            self.assertEqual(device["active_leases"], 0)
                            self.assertEqual(device["snapshots"], 0)
                            self.assertEqual(device["references_at_end"], 0)
                        if new_tokens and state != "FF":
                            self.assertGreater(
                                sum(d["cow_count"] for d in summary["devices"]), 0
                            )


if __name__ == "__main__":
    unittest.main()
