"""First output generation belongs to D in the optional complete flow."""

import unittest
import simpy
from tests.test_handover_p2p import engine_for, request, p2p_connector, store_connector
from tests.test_handover_states import evict_idle_cache


class HandoverFlowTest(unittest.TestCase):
    def test_generated_kv_writeback_is_fetchable_on_the_next_turn(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True)
        first = request(
            env, engine, 0, ["a", "b"], outputs=18, conv="a", turn=0, prefix=0
        )
        first.output_hash_ids = ["o"]
        env.run()
        engine.raise_if_failed()
        self.assertTrue(first.completion_event.triggered)
        store = store_connector(engine.workers[1]).service.store
        self.assertEqual(len(store.objects), 3)
        self.assertTrue(all(o.ready_at <= env.now for o in store.objects.values()))
        self.assertGreaterEqual(
            first.completed_at, first.handover_writeback_completed_at
        )
        evict_idle_cache(engine.workers[0])
        second = request(
            env, engine, 1, ["a", "b", "o", "c"], outputs=2, conv="a", turn=1, prefix=48
        )
        env.run()
        engine.raise_if_failed()
        self.assertEqual(second.first_prefill_source.cpu_hit_tokens, 48)
        self.assertTrue(second.is_done)

    def test_single_output_waits_for_transfer_and_one_d_step(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, bandwidth=10000)
        req = request(env, engine, 0, ["a", "b"], outputs=1, conv="a", turn=0, prefix=0)
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertEqual(req.generation_idx, 1)
        self.assertEqual(p2p_connector(engine.workers[0]).stats.transfer_bytes, 128)
        self.assertGreater(req.handoff_ready_at, req.handover_prefill_finished_at)
        self.assertAlmostEqual(
            req.arrival_at + req.prefill_latency,
            req.handoff_ready_at + req.handover_first_decode_service_s,
        )

    def test_d_prefix_is_reserved_from_arrival_until_admission(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True)
        first = request(
            env, engine, 0, ["a", "b"], outputs=2, conv="a", turn=0, prefix=0
        )
        env.run()
        engine.raise_if_failed()
        second = request(
            env, engine, 1, ["a", "b", "c", "d"], outputs=2, conv="a", turn=1, prefix=32
        )
        env.run(until=env.now + 0.0001)
        manager, blocks = second.handover_d_reservation
        self.assertTrue(blocks)
        self.assertTrue(all(b.ref_count > 0 for b in blocks))
        self.assertEqual(second.handover_d_cached_tokens, 32)
        env.run()
        engine.raise_if_failed()
        self.assertIsNone(second.handover_d_reservation)
        self.assertTrue(first.is_done and second.is_done)


if __name__ == "__main__":
    unittest.main()
