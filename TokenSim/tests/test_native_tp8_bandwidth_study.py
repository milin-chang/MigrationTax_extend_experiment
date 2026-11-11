"""Exercise TP8 aggregate KV transport while keeping collectives on NVLink."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.mooncake import Segment
from TokenSim.research.native_runtime import execute
from tests.test_native_bandwidth_study import request

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/servegen7b-tp8-200"


class TP8BandwidthStudyTests(unittest.TestCase):
    def run_native(self, rows, method, label, probe=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = execute(
                rows,
                method,
                0,
                output,
                hardware_profile=CONFIG / "hardware.json",
                experiment_config=CONFIG / f"study_{label}.json",
                probe=probe,
            )
            self.assertEqual(result["completed"], len(rows))
            self.assertFalse(result["unfinished"] or result["rejected"])
            for rank in json.loads((output / "native-end-state.json").read_text()):
                self.assertEqual(rank["gpu_status"][1], 0)
                self.assertFalse(
                    any(
                        rank[k]
                        for k in ("waiting", "running", "loading", "incoming", "pinned")
                    )
                )
            native = json.loads(
                next((output / "native").glob("result_*.json")).read_text()
            )
            self.assertGreater(native["parallel_tp_collective_latency"], 0)
            with (output / "requests.jsonl").open() as stream:
                return [json.loads(line) for line in stream]

    def test_all_methods_complete_with_64_physical_ranks(self):
        rows = [
            request(0, None, 0, 128, 128, 65, 0),
            request(1, "0", 1, 256, 128, 4, 0.001),
        ]

        def probe(engine, admission, env):
            self.assertEqual(len(engine.workers), 64)
            self.assertEqual(len(engine.handover_tp_replicas), 8)
            self.assertEqual(len(admission.budgets), 8)
            for worker in engine.workers:
                self.assertEqual(worker.cache_config.size_per_token, 65536)
                self.assertEqual(worker.device.scale_up_link, "nvlink4")

        for method in ("Serial", "PPD", "MigrationTax"):
            records = self.run_native(rows, method, "bw100", probe)
            self.assertTrue(
                all(row["first_d_output_sampled_at_s"] is not None for row in records)
            )

    def test_aggregate_payload_uses_swept_bandwidth_not_per_rank_bandwidth(self):
        first = {}
        for label, bandwidth in (("bw50", 50e9), ("bw200", 200e9)):

            def probe(engine, admission, env):
                worker = engine.workers[0]
                direct, store = [
                    next(
                        c
                        for c in flatten_connectors(worker.connector)
                        if c.name == name
                    )
                    for name in ("MooncakeConnector", "MooncakeStoreConnector")
                ]
                plan = direct.build_transfer_plan(
                    requests=[
                        SimpleNamespace(
                            id=123, num_physical_token_blocks=8, hash_ids=None
                        )
                    ],
                    request_block_counts={123: 8},
                    source_worker_id=0,
                    target_worker_id=32,
                    kind="save",
                )
                self.assertEqual(plan.bytes, 8 * 64 * 524288)
                self.assertAlmostEqual(plan.latency, plan.bytes / bandwidth, places=12)
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
                self.assertEqual(worker.device.scale_up_link, "nvlink4")

            records = self.run_native(
                [request(0, None, 0, 8192, 8192, 4, 0)], "Serial", label, probe
            )
            first[label] = records[0]["native_prefill_latency_s"]
        self.assertGreater(first["bw50"], first["bw200"])


if __name__ == "__main__":
    unittest.main()
