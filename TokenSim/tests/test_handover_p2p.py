"""Opt-in native handoff preserves copies and accepts incoming KV once."""

import unittest
from unittest.mock import patch
import simpy
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.config import ClusterConfig, WorkerGroupConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from tests.test_kv_transfer_integration import (
    _MODEL,
    _DEVICE,
    _pd_cluster,
    _workload_config,
)
from tests.support.hardware import make_hardware
from tests.test_handover_store_policy import FixedBackend


def engine_for(
    env,
    enabled=True,
    backend=None,
    routing=False,
    capacity=None,
    bandwidth=None,
    store_admission="always",
    workers_per_phase=1,
    flow=False,
    clock="completion",
    block_size=16,
    cost_model="native",
    overhead_s=0.0,
    max_num_seqs=None,
):
    cluster = _pd_cluster()
    if workers_per_phase != 1:
        cluster = ClusterConfig(
            num_workers=2 * workers_per_phase,
            networks=cluster.networks,
            worker_groups=[
                WorkerGroupConfig(
                    role=role,
                    hardware="TestGPU",
                    num_workers=workers_per_phase,
                    network="net1",
                )
                for role in ("prefill", "decode")
            ],
        )
    connector = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_connector_extra_config": (
                        {"bandwidth_bytes_per_s": bandwidth, "fixed_latency_us": 0}
                        if bandwidth
                        else {}
                    ),
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_connector_extra_config": {
                        "save_policy": "turn_end" if flow else "prefill_delta",
                        "global_segment_size": 65536,
                        "admission_policy": store_admission,
                        "local_buffer_size": 65536,
                        "host_available_bytes": 2**20,
                        "eviction_policy": "none",
                    },
                },
            ]
        },
    )
    if capacity is not None:
        cluster = ClusterConfig(
            num_workers=2 * workers_per_phase,
            networks=cluster.networks,
            worker_groups=cluster.worker_groups,
            kv_cache_capacity_tokens_per_dp_rank=capacity,
        )
    with patch(
        "TokenSim.llm.engine.worker.build_latency_backend",
        return_value=backend or FixedBackend(),
    ):
        return LLMEngine(
            env=env,
            hardware=make_hardware(_DEVICE, models=[_MODEL]),
            model_spec=_MODEL,
            cluster_config=cluster,
            kv_transfer_config=connector,
            workload_config=_workload_config(),
            config=SimulationConfig(
                batching="paged-attn",
                max_num_batched_tokens=4096,
                preserve_p2p_kv=enabled,
                handover_routing=routing,
                handover_flow=flow,
                handover_clock=clock,
                block_size=block_size,
                handover_cost_model=cost_model,
                handover_fixed_overhead_s=overhead_s,
                max_num_seqs=(
                    max_num_seqs
                    if max_num_seqs is not None
                    else (1 if clock == "arrival" else 64)
                ),
            ),
        )


def request(env, engine, rid, hashes, outputs=4, conv=None, turn=None, prefix=None):
    req = Request(
        id=rid,
        prefill_len=engine.config.block_size * len(hashes),
        decode_len=outputs,
        block_size=engine.config.block_size,
        hash_ids=hashes,
        chat_id=conv,
        turn=turn,
        prefix_tokens=prefix,
    )
    if engine.config.handover_clock == "arrival":
        req.handover_context_hash_ids = list(hashes) + [
            f"output:{conv}:{turn}:{i}" for i in range(outputs - 1)
        ]
    req.arrive(env)
    engine.add_requests([req])
    return req


def store_connector(worker):
    return next(
        c
        for c in flatten_connectors(worker.connector)
        if c.name == "MooncakeStoreConnector"
    )


def p2p_connector(worker):
    return next(
        c for c in flatten_connectors(worker.connector) if c.name == "MooncakeConnector"
    )


class HandoverP2PTest(unittest.TestCase):
    def test_received_kv_does_not_fetch_the_same_prompt_from_store(self):
        env = simpy.Environment()
        engine = engine_for(env)
        req = request(env, engine, 0, ["a", "b"])
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertEqual(store_connector(engine.workers[1]).stats.load_wait_time, 0)
        self.assertIsNotNone(req.handoff_ready_at)
        # Native fused-prefill TTFT is retained and documented separately.
        self.assertGreater(req.handoff_ready_at, req.arrival_at + req.prefill_latency)

    def test_d_prefix_hit_transfers_only_missing_blocks(self):
        env = simpy.Environment()
        engine = engine_for(env)
        first = request(env, engine, 0, ["a", "b"])
        env.run()
        p2p = p2p_connector(engine.workers[0])
        initial = p2p.stats.transfer_bytes
        plans = []
        expected = []
        source_connector = engine.workers[0].connector
        original = source_connector.build_transfer_plan

        def observe(**kwargs):
            plan = original(**kwargs)
            plans.append(plan)
            blocks = engine.workers[0].scheduler.block_manager.block_table.get_blocks(1)
            expected.extend(str(b.block_hash) for b in blocks[2:4])
            return plan

        with patch.object(source_connector, "build_transfer_plan", side_effect=observe):
            second = request(env, engine, 1, ["a", "b", "c", "d"])
            env.run()
        engine.raise_if_failed()
        self.assertTrue(first.is_done and second.is_done)
        self.assertEqual(initial, 2 * 16 * 4)
        self.assertEqual(p2p.stats.transfer_bytes - initial, 2 * 16 * 4)
        self.assertEqual(plans[0].extra["missing_block_ranges"][1], [2, 4])
        self.assertEqual(plans[0].keys, expected)
        self.assertEqual(store_connector(engine.workers[1]).stats.load_wait_time, 0)

    def test_p_copy_stays_pinned_until_d_completes(self):
        env = simpy.Environment()
        engine = engine_for(env)
        req = request(env, engine, 0, ["a", "b"], outputs=8)
        env.run(until=0.002)
        source = engine.workers[0].scheduler.block_manager
        self.assertFalse(req.is_done)
        self.assertTrue(source.block_table.get_blocks(req.id))
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertFalse(source.block_table.get_blocks(req.id))
        later = Request(
            id=3, prefill_len=32, decode_len=2, block_size=16, hash_ids=["a", "b"]
        )
        self.assertEqual(source.kv_cache_manager.plan_reuse(later).hit_tokens, 32)

    def test_cancel_releases_source_and_destination(self):
        env = simpy.Environment()
        engine = engine_for(env)
        req = request(env, engine, 0, ["a", "b"], outputs=16)

        def cancel():
            yield env.timeout(0.002)
            yield engine.cancel_requests([req])[0]

        env.process(cancel())
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.cancel_requested)
        for worker in engine.workers:
            self.assertFalse(
                worker.scheduler.block_manager.block_table.get_blocks(req.id)
            )

    def test_single_output_token_still_exposes_native_endpoint_limitation(self):
        env = simpy.Environment()
        engine = engine_for(env)
        req = request(env, engine, 0, ["a", "b"], outputs=1)
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertIsNone(getattr(req, "handoff_ready_at", None))
        self.assertEqual(p2p_connector(engine.workers[0]).stats.transfer_bytes, 0)

    def test_pinned_capacity_waits_for_d_release_without_zero_time_spin(self):
        env = simpy.Environment()
        engine = engine_for(env, capacity=64, bandwidth=10000)
        first = request(env, engine, 0, ["a", "b"], outputs=4)
        pending = []

        def later():
            yield env.timeout(0.002)
            pending.append(request(env, engine, 1, ["x", "y", "z"], outputs=2))

        env.process(later())
        for _ in range(1000):
            if env.peek() == float("inf"):
                break
            env.step()
        else:
            self.fail("pinned capacity caused a zero-time scheduling spin")
        engine.raise_if_failed()
        self.assertTrue(first.is_done and pending[0].is_done)
        self.assertFalse(engine.workers[0].pinned_handoffs)


if __name__ == "__main__":
    unittest.main()
