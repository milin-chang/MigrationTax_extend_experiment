"""Execute real native nodes for each state and O=1/2/long HF handoffs."""

import contextlib
import gzip
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "splitwise-sim"))
from native_method_runtime import MethodSimulation
from kv_cache import Block
from native_methods.tests.profile_fixtures import write_timing_fixture
from native_methods.hardware_profile import load_profile

LEGACY = ROOT / "native_methods/profiles/llama3-8b-h100-tp1.json"


class SplitwiseTests(unittest.TestCase):
    def test_four_states_and_output_boundaries(self):
        for state in ["HH", "FH", "HF", "FF"]:
            for outputs, new_tokens in itertools.product([1, 2, 5], [0, 16]):
                for method in ["Serial", "PPD", "MigrationTax"]:
                    with self.subTest(
                        state=state,
                        outputs=outputs,
                        method=method,
                        new_tokens=new_tokens,
                    ), tempfile.TemporaryDirectory() as d:
                        output = Path(d)
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
                                context_tokens=64 + new_tokens,
                                new_tokens=new_tokens,
                                output_tokens=outputs,
                                arrival_seconds=10.0,
                            ),
                        ]
                        with contextlib.chdir(output):
                            sim = MethodSimulation(
                                rows,
                                method,
                                write_timing_fixture(output, load_profile(LEGACY)),
                                output,
                                deployment=load_profile(LEGACY),
                            )

                            def pressure():
                                if "0" not in sim.completed:
                                    raise AssertionError("Fixture parent not complete")
                                job = sim.jobs["0"]
                                D = job["D"]
                                for role, miss in [
                                    ("P", state[0] == "F"),
                                    ("D", state[1] == "F"),
                                ]:
                                    if not miss:
                                        continue
                                    devices = sim.devices[:4] if role == "P" else [D]
                                    for device in devices:
                                        dummy = Block(
                                            "pressure:" + device.instance_id,
                                            1,
                                            device.cache.capacity_bytes,
                                        )
                                        device.cache.reserve(
                                            [dummy], "background", sim.time
                                        )
                                        device.cache.publish(
                                            [dummy], "background", sim.time
                                        )
                                        device.cache.release("background", sim.time)

                            sim.schedule(9, pressure)
                            sim.run()
                            result = sim.finish()
                        observed = [
                            json.loads(s) for s in (output / "requests.jsonl").open()
                        ]
                        self.assertEqual(result["status"], "completed")
                        self.assertEqual(observed[1]["state"], state)
                        self.assertEqual(observed[1]["generated"], outputs)
                        events = [
                            json.loads(s)
                            for s in gzip.open(output / "native-nodes.jsonl.gz", "rt")
                            if json.loads(s)["source_id"] == "1"
                        ]
                        nodes = {e["operation"]["id"]: e for e in events}
                        fused = (
                            method != "Serial"
                            and state[1] == "H"
                            or (method == "MigrationTax" and state == "HF")
                        )
                        self.assertEqual(
                            sum(e["source_outputs_emitted"] for e in events), outputs
                        )
                        expected_device = (
                            "P" if method == "MigrationTax" and state == "HF" else "D"
                        )
                        self.assertEqual(
                            observed[1]["first_output_device"], expected_device
                        )
                        producer = nodes["prefill"] if fused else nodes["first_output"]
                        self.assertEqual(
                            producer["native_class"],
                            "MethodPrompt" if fused else "MethodDecode",
                        )
                        self.assertEqual(producer["source_outputs_emitted"], 1)
                        self.assertEqual(
                            producer["completed_at_s"], observed[1]["first_at_s"]
                        )
                        if fused:
                            self.assertTrue(nodes["first_output"]["fused_first_marker"])
                            self.assertEqual(
                                producer["native_logits_replay"], new_tokens == 0
                            )
                        self.assertGreater(
                            result["native_predictor_calls"][
                                "DatabasePerformanceModel.get_iteration_duration"
                            ],
                            0,
                        )
                        if method == "MigrationTax" and state == "HF":
                            self.assertGreaterEqual(
                                nodes["ownership"]["completed_at_s"],
                                nodes["base_handoff"]["completed_at_s"],
                            )
                            self.assertGreaterEqual(
                                nodes["ownership"]["completed_at_s"],
                                nodes["incremental_sync"]["completed_at_s"],
                            )
                            self.assertGreaterEqual(
                                nodes["remaining"]["completed_at_s"],
                                nodes["ownership"]["completed_at_s"],
                            )
                            self.assertLess(
                                producer["native_metrics"]["start_timestamp"],
                                nodes["base_handoff"]["completed_at_s"],
                            )
                        if method == "MigrationTax" and state == "FF":
                            transferred = sum(
                                e["bytes"]
                                for e in events
                                if e["operation"]["device"] == "P->D"
                            )
                            self.assertEqual(
                                transferred,
                                sum(b.bytes for b in sim.source_blocks(rows[1])),
                            )

    def test_cross_device_capacity_deferral_releases_no_future_kv(self):
        rows = []
        for i in range(12):
            rows.append(
                dict(
                    source_id=str(i),
                    previous_source_id=None,
                    conversation=i,
                    turn=0,
                    context_tokens=64,
                    new_tokens=64,
                    output_tokens=5,
                    arrival_seconds=0.0,
                )
            )
        for i in range(12):
            rows.append(
                dict(
                    source_id=str(12 + i),
                    previous_source_id=str(i),
                    conversation=i,
                    turn=1,
                    context_tokens=80,
                    new_tokens=16,
                    output_tokens=5,
                    arrival_seconds=0.001,
                )
            )
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            sim = MethodSimulation(
                rows,
                "Serial",
                write_timing_fixture(directory, load_profile(LEGACY)),
                Path(directory),
                deployment=load_profile(LEGACY),
            )
            for device in sim.devices:
                device.cache.capacity_bytes = 96 * sim.kv_bytes
            sim.run()
            result = sim.finish()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed"], len(rows))
            self.assertGreater(result["native_counters"]["capacity_deferrals"], 0)
            for device in sim.devices:
                self.assertEqual(device.cache.active_bytes, 0)
                self.assertFalse(device.cache._owners)
                self.assertTrue(
                    all(r.ready_at is not None for r in device.cache.replicas.values())
                )
