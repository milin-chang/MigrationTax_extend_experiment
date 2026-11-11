"""Symmetric TP handover: physical inventory, payloads and lifetime barriers."""

from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.handover_parallel import validate_handover_parallel
from TokenSim.kv_transfer.connectors.base import (
    flatten_connectors,
    transfer_block_bytes,
)
from TokenSim.research.native_hardware import configure_native_hardware
from TokenSim.research.native_runtime import execute
from TokenSim.research.native_migration_tests import one_case
from TokenSim.research.native_migration_overlap_tests import one_case as overlap_case
from TokenSim.research.native_migration_tests import requests as fixture_requests
from tests.test_handover_states import evict_idle_cache
from native_methods.hardware_profile import ROOT, load_profile


class TensorParallelHandoverTests(unittest.TestCase):
    def test_serial_and_ppd_returning_states_release_all_ranks(self):
        for method in ("Serial", "PPD"):
            for cell in ("HH", "FH", "HF", "FF"):
                for outputs in (1, 2, 5):
                    with self.subTest(
                        method=method, cell=cell, outputs=outputs
                    ), tempfile.TemporaryDirectory() as directory:

                        def probe(engine, admission, env):
                            original = engine.record_request_completion

                            def completed(kind, req):
                                original(kind, req)
                                if kind == "decode" and req.id == 1:

                                    def flush():
                                        # Let native completion listeners release P snapshots.
                                        yield env.timeout(0)
                                        for worker in engine.request_workers:
                                            missed = (
                                                cell[0] == "F"
                                                if worker.role == "prefill"
                                                else cell[1] == "F"
                                            )
                                            if (
                                                missed
                                                and engine._is_schedulable_worker(
                                                    worker
                                                )
                                            ):
                                                evict_idle_cache(worker)

                                    env.process(flush())

                            engine.record_request_completion = completed

                        result = execute(
                            fixture_requests(outputs, 17, 64),
                            method,
                            0,
                            directory,
                            probe=probe,
                        )
                        self.assertEqual(result["status"], "completed")
                        path = Path(directory)
                        rows = [
                            json.loads(s)
                            for s in (path / "requests.jsonl").read_text().splitlines()
                        ]
                        self.assertEqual(rows[-1]["generated"], outputs)
                        routes = [
                            json.loads(s)
                            for s in (path / "native-route-decisions.jsonl")
                            .read_text()
                            .splitlines()
                        ]
                        route = next(r for r in routes if r["request_id"] == 2)
                        state = ("H" if route["p_local_tokens"] >= 64 else "F") + (
                            "H" if route["d_local_tokens"] >= 64 else "F"
                        )
                        self.assertEqual(state, cell)
                        if method == "PPD" and cell[1] == "H":
                            self.assertEqual(
                                route["native_compute_worker"], route["decode_worker"]
                            )
                        end = json.loads((path / "native-end-state.json").read_text())
                        self.assertTrue(
                            all(
                                r["gpu_status"][1] == 0
                                and not r["waiting"]
                                and not r["running"]
                                for r in end
                            )
                        )
                        budgets = json.loads(
                            (path / "native-admission-budget.json").read_text()
                        )
                        self.assertTrue(
                            all(
                                r["occupied_at_end"] == 0
                                for r in budgets["devices"].values()
                            )
                        )

    def test_tp8_has_eight_complete_model_replicas_and_per_rank_capacity(self):
        deployment = load_profile()
        observed = {}

        def probe(engine, admission, env):
            observed["engine"] = engine
            self.assertEqual(len(engine.workers), 64)
            self.assertEqual(len(engine.router.prefill_workers), 4)
            self.assertEqual(len(engine.router.decode_workers), 4)
            self.assertEqual(len(admission.budgets), 8)
            managers = set()
            for ids in engine.handover_tp_replicas.values():
                self.assertEqual(len(ids), 8)
                workers = [engine.workers[i] for i in ids]
                leader = workers[0]
                managers.add(id(leader.scheduler.block_manager))
                self.assertEqual(leader.parallel_communicator.tp_group(), ids)
                self.assertEqual(engine.placement.lowest_common_level(ids), 0)
                for worker in workers:
                    self.assertIs(
                        worker.scheduler.block_manager, leader.scheduler.block_manager
                    )
                    self.assertEqual(worker.cache_config.model_param_size, 17243832320)
                    self.assertEqual(worker.cache_config.size_per_token, 40960)
                    self.assertEqual(worker.cache_config.num_gpu_blocks, 98398)
                    self.assertEqual(
                        transfer_block_bytes(worker.cache_config), 16 * 327680
                    )
                for connector in flatten_connectors(leader.connector):
                    if connector.name == "MooncakeStoreConnector":
                        self.assertEqual(connector.block_bytes, 16 * 327680)
                        self.assertEqual(
                            connector.service.store.memory_capacity_bytes, 2**39
                        )
                        self.assertTrue(
                            all(
                                ranks == 8
                                for _, ranks, _ in connector.service.contributors.values()
                            )
                        )
            self.assertEqual(len(managers), 8)

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
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = execute(rows, "Serial", 0, directory, probe=probe)
            self.assertEqual(result["status"], "completed")
            effective = json.loads(
                (Path(directory) / "effective-config.json").read_text()
            )
            self.assertEqual(effective["hardware_alignment"]["total_gpus"], 64)
            self.assertEqual(effective["hardware_alignment"]["model_dtype"], "fp16")
            self.assertTrue(effective["tensor_parallel_handover"]["enabled"])
        engine = observed["engine"]
        leader = engine.router.prefill_workers[0]
        self.assertGreater(leader.parallel_communicator.stats.tp_collective_latency, 0)
        self.assertGreater(leader.parallel_communicator.stats.tp_shard_event_count, 0)

    def test_native_migration_boundaries_and_multichunk_release(self):
        with tempfile.TemporaryDirectory() as directory:
            for new_tokens in (0, 17, 4097):
                for cell in ("HH", "FH", "HF", "FF"):
                    for outputs in (1, 2, 5):
                        with self.subTest(
                            cell=cell, outputs=outputs, new_tokens=new_tokens
                        ):
                            path = Path(directory) / f"{cell}-{new_tokens}-{outputs}"
                            one_case(path, cell, outputs, new_tokens, 64)
                            evidence = json.loads(
                                (path / "native-migration-evidence.json").read_text()
                            )
                            self.assertEqual(evidence["layers"], 80)
                            self.assertEqual(
                                evidence["native_kv_geometry"]["tp_shards"], 8
                            )
                            self.assertEqual(
                                evidence["native_kv_geometry"]["token_bytes"], 327680
                            )
                            with gzip.open(
                                path / "native-migration-events.jsonl.gz", "rt"
                            ) as stream:
                                for line in stream:
                                    event = json.loads(line)
                                    if "tp_shards" in event:
                                        self.assertEqual(event["tp_shards"], 8)
                                        self.assertEqual(
                                            event["bytes"], event["per_rank_bytes"] * 8
                                        )
                                        self.assertEqual(
                                            event["completion_scope"], "all_tp_shards"
                                        )

    def test_short_hf_writeback_does_not_corrupt_active_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            for outputs in (1, 2):
                with self.subTest(outputs=outputs):
                    proof = overlap_case(Path(directory) / f"O{outputs}", outputs)
                    self.assertTrue(proof["owners_and_budgets_released"])

    def test_tp8_is_opt_in_and_rejects_asymmetric_or_pipeline_execution(self):
        deployment = load_profile()
        hardware = HardwareContext.load(ROOT / "TokenSim/data")
        model, parallel, cluster = configure_native_hardware(deployment, hardware)
        config = SimulationConfig(
            batching="paged-attn",
            handover_flow=True,
            handover_tp_groups=True,
            handover_cost_model="native",
        )
        validate_handover_parallel(config, parallel, cluster, model)
        for changed in (
            replace(config, handover_tp_groups=False),
            replace(config, dp_lockstep=True),
        ):
            with self.assertRaises(ConfigurationError):
                validate_handover_parallel(changed, parallel, cluster, model)
        with self.assertRaises(ConfigurationError):
            validate_handover_parallel(
                config,
                ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=2),
                cluster,
                model,
            )
        with self.assertRaises(ConfigurationError):
            validate_handover_parallel(
                config, parallel, cluster, replace(model, num_key_value_heads=2)
            )

    def test_old_tp1_profile_keeps_single_gpu_geometry(self):
        deployment = load_profile(
            ROOT / "native_methods/profiles/llama3-8b-h100-tp1.json"
        )
        hardware = HardwareContext.load(ROOT / "TokenSim/data")
        model, parallel, cluster = configure_native_hardware(deployment, hardware)
        self.assertEqual(parallel.world_size, 1)
        self.assertEqual(cluster.num_workers, 8)
        self.assertEqual(deployment.kv_blocks, 31314)
        self.assertEqual(deployment.weights_bytes_per_gpu, deployment.weights_bytes)
