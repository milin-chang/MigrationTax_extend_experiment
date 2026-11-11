"""Validate block geometry, controlled links and original-arrival cache probes."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.mooncake import Segment
from TokenSim.research.native_runtime import execute

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/hardware-llama7b-h100-tp1-block64.json"


def request(source, parent, turn, context, new, outputs, arrival):
    return dict(
        source_id=str(source),
        previous_source_id=parent,
        conversation=0,
        turn=turn,
        context_tokens=context,
        new_tokens=new,
        output_tokens=outputs,
        arrival_seconds=arrival,
    )


class BandwidthStudyTests(unittest.TestCase):
    def run_native(self, rows, method, label, probe=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = execute(
                rows,
                method,
                0,
                output,
                hardware_profile=PROFILE,
                experiment_config=ROOT / f"configs/study_{label}.json",
                probe=probe,
            )
            self.assertEqual(result["status"], "completed")
            with (output / "requests.jsonl").open() as stream:
                records = [json.loads(line) for line in stream]
            for device in json.loads((output / "native-end-state.json").read_text()):
                self.assertEqual(device["gpu_status"][1], 0)
                self.assertFalse(
                    any(
                        device[key]
                        for key in (
                            "waiting",
                            "running",
                            "loading",
                            "incoming",
                            "pinned",
                        )
                    )
                )
            return sorted(records, key=lambda row: int(row["source_id"]))

    def test_64_token_blocks_and_original_arrival_miss_on_later_d_hit(self):
        rows = [
            request(0, None, 0, 128, 128, 65, 0),
            request(1, "0", 1, 256, 128, 3, 0.001),
        ]
        for method in ("Serial", "PPD", "MigrationTax"):
            observed = []

            def inspect(engine, admission, env):
                self.assertEqual(len(engine.workers), 8)
                for worker in engine.workers:
                    self.assertEqual(worker.cache_config.block_size, 64)
                    self.assertEqual(worker.cache_config.num_gpu_blocks, 2034)
                    self.assertEqual(
                        worker.device.memory_bandwidth_bytes_per_s.value, 3352e9
                    )
                observed.append(True)

            records = self.run_native(rows, method, "bw100", inspect)
            self.assertTrue(observed)
            self.assertTrue(records[1]["decode_miss_at_original_arrival"])
            self.assertEqual(records[1]["d_local_tokens_at_original_arrival"], 0)
            if method in ("PPD", "MigrationTax"):
                self.assertEqual(records[1]["plan_branch"], "decode_local_prefill")
            self.assertGreaterEqual(
                records[0]["first_d_output_visible_at_s"], records[0]["arrival_s"]
            )

    def test_native_transfer_service_tracks_bandwidth_and_preserves_hbm(self):
        rows = [request(0, None, 0, 8192, 8192, 3, 0)]
        first = {}
        for label, bandwidth in (("bw50", 50e9), ("bw200", 200e9)):

            def inspect(engine, admission, env):
                worker = engine.workers[0]
                connectors = {c.name: c for c in flatten_connectors(worker.connector)}
                direct = connectors["MooncakeConnector"]
                plan = direct.build_transfer_plan(
                    requests=[
                        SimpleNamespace(
                            id=123, num_physical_token_blocks=8, hash_ids=None
                        )
                    ],
                    request_block_counts={123: 8},
                    source_worker_id=0,
                    target_worker_id=4,
                    kind="save",
                )
                self.assertEqual(plan.bytes, 8 * 64 * 524288)
                self.assertAlmostEqual(plan.latency, plan.bytes / bandwidth, places=12)
                store = connectors["MooncakeStoreConnector"]
                transfer = store.transfer_engine.create_transfer(
                    source_segment=Segment("store", "dram"),
                    target_segment=Segment("P", "vram"),
                    blocks=8,
                    bytes_=plan.bytes,
                )
                self.assertAlmostEqual(
                    store._transfer_latency(transfer), plan.bytes / bandwidth, places=12
                )
                self.assertEqual(
                    worker.device.memory_bandwidth_bytes_per_s.value, 3352e9
                )

            first[label] = self.run_native(rows, "Serial", label, inspect)[0][
                "d_output_visible_latency_s"
            ]
        self.assertGreater(first["bw50"], first["bw200"])


if __name__ == "__main__":
    unittest.main()
