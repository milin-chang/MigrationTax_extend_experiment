"""Check completion visibility and explicit multi-worker placements."""

import unittest
import simpy
from TokenSim.llm.llm_request import Request
from TokenSim.errors import SimulationStateError
from unittest.mock import patch
from tests.test_handover_p2p import engine_for, request
from tests.test_handover_states import evict_idle_cache


def probe_request(rid=90):
    return Request(
        id=rid, prefill_len=32, decode_len=2, block_size=16, hash_ids=["a", "b"]
    )


def visibility_case():
    env = simpy.Environment()
    engine = engine_for(env, routing=True, bandwidth=10000)
    req = request(env, engine, 0, ["a", "b"], outputs=4, conv="a", turn=0, prefix=0)
    destination = engine.workers[1].scheduler.block_manager
    env.run(until=0.002)
    before = destination.kv_cache_manager.plan_reuse(probe_request()).hit_tokens
    pinned = [
        b.ref_count
        for b in engine.workers[0].scheduler.block_manager.block_table.get_blocks(
            req.id
        )
    ]
    started = getattr(req, "handoff_received_at", None)
    ready_before = getattr(req, "handoff_ready_at", None)
    env.run()
    engine.raise_if_failed()
    after = destination.kv_cache_manager.plan_reuse(probe_request(91)).hit_tokens
    return dict(
        before_tokens=before,
        after_tokens=after,
        p_refs_during_transfer=pinned,
        received_s=started,
        ready_s=req.handoff_ready_at,
        ready_before=ready_before,
        completed_s=req.completed_at,
    )


def multi_worker_case():
    env = simpy.Environment()
    engine = engine_for(env, routing=True, workers_per_phase=4)
    requests = []
    for i in range(4):
        requests.append(
            request(
                env,
                engine,
                i,
                [str(i) + "a", str(i) + "b"],
                outputs=2,
                conv=str(i),
                turn=0,
                prefix=0,
            )
        )
        env.run()
        engine.raise_if_failed()
    warm_routes = list(engine.router.observations)
    engine.reset_profile_stats()
    for i in range(4):
        requests.append(
            request(
                env,
                engine,
                i + 4,
                [str(i) + "a", str(i) + "b", str(i) + "c"],
                outputs=2,
                conv=str(i),
                turn=1,
                prefix=32,
            )
        )
        env.run()
        engine.raise_if_failed()
    return dict(
        roles=[w.role for w in engine.workers],
        warm_routes=warm_routes,
        returning_routes=engine.router.observations,
        completed=[r.is_done for r in requests],
        returning_p_hits=[r.first_prefill_source.gpu_hit_tokens for r in requests[4:]],
    )


class HandoverVisibilityTest(unittest.TestCase):
    def test_lost_destination_prefix_aborts_instead_of_fabricating_kv(self):
        env = simpy.Environment()
        engine = engine_for(env, routing=True)
        request(env, engine, 0, ["a", "b"], outputs=2, conv="a", turn=0, prefix=0)
        env.run()
        engine.raise_if_failed()
        original = engine.dispatch_prefill_to_decode

        def lose_queued_prefix(sender, requests):
            original(sender, requests)
            evict_idle_cache(engine.workers[1])

        with patch.object(
            engine, "dispatch_prefill_to_decode", side_effect=lose_queued_prefix
        ):
            request(
                env,
                engine,
                1,
                ["a", "b", "c", "d"],
                outputs=2,
                conv="a",
                turn=1,
                prefix=32,
            )
            env.run()
        with self.assertRaisesRegex(SimulationStateError, "destination prefix changed"):
            engine.raise_if_failed()

    def test_destination_cannot_reuse_inflight_blocks(self):
        result = visibility_case()
        self.assertEqual(result["before_tokens"], 0)
        self.assertEqual(result["after_tokens"], 32)
        self.assertIsNone(result["ready_before"])
        self.assertTrue(all(x > 0 for x in result["p_refs_during_transfer"]))
        self.assertGreater(result["ready_s"], result["received_s"])

    def test_eight_workers_preserve_holder_routing_and_d_affinity(self):
        result = multi_worker_case()
        self.assertEqual(result["roles"].count("prefill"), 4)
        self.assertEqual(result["roles"].count("decode"), 4)
        self.assertTrue(all(result["completed"]))
        self.assertEqual(result["returning_p_hits"], [32] * 4)
        self.assertEqual(len({r["prefill_worker"] for r in result["warm_routes"]}), 4)
        self.assertEqual(len({r["decode_worker"] for r in result["warm_routes"]}), 4)
        for warm, later in zip(result["warm_routes"], result["returning_routes"]):
            self.assertEqual(warm["prefill_worker"], later["prefill_worker"])
            self.assertEqual(warm["decode_worker"], later["decode_worker"])


if __name__ == "__main__":
    unittest.main()
