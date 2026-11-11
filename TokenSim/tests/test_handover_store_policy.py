"""Completed prompt deltas are saved once without changing default semantics."""

import unittest
from unittest.mock import patch
import simpy

from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.latency.base import LatencyBackend
from tests.support.hardware import make_device, make_hardware, make_model


class FixedBackend(LatencyBackend):
    def estimate_step_latency(self, requests):
        return 0.001


def make_engine(env, policy):
    model = make_model(
        "handover-test",
        hidden_size=1,
        intermediate_size=2,
        num_layers=1,
        num_attention_heads=1,
        vocab_size=4,
    )
    device = make_device("TestGPU", memory_gib=0.001)
    cluster = ClusterConfig(
        num_workers=1,
        networks={"n": "ethernet-test"},
        worker_groups=[WorkerGroupConfig("hybrid", "TestGPU", 1, "n")],
    )
    connector = KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "master_server_address": "handover-test",
            "global_segment_size": 65536,
            "local_buffer_size": 65536,
            "host_available_bytes": 2**20,
            "save_policy": policy,
            "eviction_policy": "none",
            "load_async": False,
        },
    )
    with patch(
        "TokenSim.llm.engine.worker.build_latency_backend", return_value=FixedBackend()
    ):
        engine = LLMEngine(
            env=env,
            hardware=make_hardware(device, models=[model]),
            model_spec=model,
            cluster_config=cluster,
            kv_transfer_config=connector,
            config=SimulationConfig(batching="paged-attn", max_num_batched_tokens=4096),
        )
    return engine


def run_request(env, engine, rid, hashes, outputs=4):
    req = Request(
        id=rid,
        prefill_len=len(hashes) * 16,
        decode_len=outputs,
        block_size=16,
        hash_ids=hashes,
    )
    req.arrive(env)
    engine.add_requests([req])
    env.run(until=req.completion_event) if req.completion_event else env.run()
    engine.raise_if_failed()
    assert req.is_done
    return req


class HandoverStorePolicyTest(unittest.TestCase):
    def test_prefill_delta_keeps_new_blocks_after_loading(self):
        env = simpy.Environment()
        engine = make_engine(env, "prefill_delta")
        run_request(env, engine, 0, ["a", "b"])
        store = engine.workers[0].connector.service.store
        self.assertEqual(len(store.objects), 2)
        # Another logical GPU must fetch the old prefix, not reuse local blocks.
        consumer = make_engine(env, "prefill_delta")
        self.assertIs(
            consumer.workers[0].connector.service, engine.workers[0].connector.service
        )
        grown = run_request(env, consumer, 1, ["a", "b", "c", "d"], outputs=8)
        self.assertEqual(grown.first_prefill_source.cpu_hit_tokens, 32)
        self.assertEqual(len(store.objects), 4)
        next_engine = make_engine(env, "prefill_delta")
        next_req = run_request(env, next_engine, 2, ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(next_req.first_prefill_source.cpu_hit_tokens, 64)
        self.assertEqual(len(store.objects), 6)

    def test_decode_length_does_not_repeat_prompt_writes(self):
        counts = []
        for outputs in (2, 16):
            env = simpy.Environment()
            engine = make_engine(env, "prefill_delta")
            run_request(env, engine, 0, ["a", "b"], outputs)
            connector = engine.workers[0].connector
            counts.append(
                (connector.stats.transfer_count, connector.stats.transfer_bytes)
            )
        self.assertEqual(counts[0], counts[1])

    def test_native_policy_is_unchanged(self):
        env = simpy.Environment()
        engine = make_engine(env, "mooncake")
        run_request(env, engine, 0, ["a", "b"])
        consumer = make_engine(env, "mooncake")
        run_request(env, consumer, 1, ["a", "b", "c", "d"])
        self.assertEqual(len(engine.workers[0].connector.service.store.objects), 2)
        self.assertEqual(consumer.workers[0].connector.options.save_policy, "mooncake")

    def test_full_hit_still_leaves_a_real_query(self):
        env = simpy.Environment()
        engine = make_engine(env, "prefill_delta")
        run_request(env, engine, 0, ["a", "b"])
        consumer = make_engine(env, "prefill_delta")
        req = run_request(env, consumer, 1, ["a", "b"])
        self.assertLess(req.cached_prefill_tokens, req.prefill_len)


if __name__ == "__main__":
    unittest.main()
