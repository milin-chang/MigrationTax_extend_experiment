"""Full and partial prefix states are produced by the arrival-clock runtime."""

import unittest
import simpy
from types import SimpleNamespace
from TokenSim.errors import ConfigurationError
from TokenSim.llm.handover_metrics import classify_prefix, native_timing_observation
from tests.test_handover_p2p import engine_for, request, p2p_connector, store_connector
from tests.test_handover_arrival import run_to_idle


def drop_logical(engine, worker, conversation):
    entry = engine.router.arrival_state.caches[worker.id].pop(conversation)
    entry.evict()


def run_arrival_cell(cell):
    env = simpy.Environment()
    engine = engine_for(
        env,
        flow=True,
        clock="arrival",
        block_size=1,
        capacity=128,
        store_admission="never" if cell[0] == "R" else "always",
    )
    prefix = [f"a{i}" for i in range(32)]
    first = request(env, engine, 0, prefix, outputs=2, conv="a", turn=0, prefix=0)
    run_to_idle(env, engine)
    if cell[0] != "H":
        drop_logical(engine, engine.workers[0], "a")
    if cell[1] == "F":
        drop_logical(engine, engine.workers[1], "a")
    before = p2p_connector(engine.workers[0]).stats.transfer_bytes
    second = request(
        env,
        engine,
        1,
        prefix + [f"b{i}" for i in range(32)],
        outputs=2,
        conv="a",
        turn=1,
        prefix=32,
    )
    run_to_idle(env, engine)
    observation = engine.router.observations[-1]
    src = second.first_prefill_source
    state = classify_prefix(
        32, src.gpu_hit_tokens, src.cpu_hit_tokens, observation["d_local_tokens"]
    )
    return dict(
        cell=state["cell"],
        p_local_tokens=src.gpu_hit_tokens,
        p_store_tokens=src.cpu_hit_tokens,
        d_local_tokens=observation["d_local_tokens"],
        computed_tokens=src.computed_tokens,
        p2p_bytes=p2p_connector(engine.workers[0]).stats.transfer_bytes - before,
        first_token=native_timing_observation(second),
        completed=second.completion_event.triggered,
    )


class ArrivalStateTest(unittest.TestCase):
    def test_turn_end_sends_full_context_once_including_existing_prefix(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, clock="arrival", block_size=1, capacity=128)
        first = request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(32)],
            outputs=2,
            conv="a",
            turn=0,
            prefix=0,
        )
        run_to_idle(env, engine)
        second = request(
            env,
            engine,
            1,
            [f"a{i}" for i in range(32)] + [f"b{i}" for i in range(16)],
            outputs=4,
            conv="a",
            turn=1,
            prefix=32,
        )
        run_to_idle(env, engine)
        self.assertEqual(first.handover_store_writeback_bytes, 33 * 4)
        self.assertEqual(second.handover_store_writeback_bytes, 51 * 4)
        stats = store_connector(engine.workers[1]).stats
        self.assertEqual(stats.transfer_count, 2)
        self.assertEqual(stats.transfer_bytes, (33 + 51) * 4)
        self.assertEqual(store_connector(engine.workers[0]).stats.transfer_bytes, 0)

    def test_older_divergent_copy_reports_only_its_common_prefix(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, clock="arrival", block_size=1, capacity=128)
        state = engine.router.arrival_state
        state.caches[0]["a"] = SimpleNamespace(
            tokens=4, request=SimpleNamespace(hash_ids=["a", "b", "c", "d"])
        )
        req = SimpleNamespace(
            chat_id="a", prefix_tokens=4, hash_ids=["a", "b", "x", "y"]
        )
        self.assertEqual(state._coverage(engine.workers[0], req)[1], 2)

    def test_invalid_preceding_turn_is_rejected(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, clock="arrival", block_size=1, capacity=128)
        request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(32)],
            outputs=2,
            conv="a",
            turn=0,
            prefix=0,
        )
        run_to_idle(env, engine)
        bad = request(
            env,
            engine,
            1,
            [f"wrong{i}" for i in range(40)],
            outputs=2,
            conv="a",
            turn=1,
            prefix=32,
        )
        with self.assertRaisesRegex(ConfigurationError, "preceding turn"):
            engine.router.select(bad)

    def test_decode_reserves_complete_turn_without_preemption(self):
        env = simpy.Environment()
        engine = engine_for(
            env, flow=True, clock="arrival", block_size=1, capacity=47, bandwidth=10000
        )
        req = request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(32)],
            outputs=16,
            conv="a",
            turn=0,
            prefix=0,
        )
        env.run(until=0.0015)
        manager = engine.workers[1].scheduler.block_manager
        self.assertEqual(len(manager.block_table.get_blocks(req.id)), 47)
        self.assertEqual(manager.get_num_free_blocks(), 0)
        run_to_idle(env, engine)
        self.assertTrue(req.completion_event.triggered)
        self.assertEqual(engine.workers[1].preempted_cnt, 0)

    def test_snapshot_pins_needed_prefix_without_unused_output_suffix(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, clock="arrival", block_size=1, capacity=128)
        request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(10)],
            outputs=100,
            conv="a",
            turn=0,
            prefix=0,
        )
        run_to_idle(env, engine)
        second = request(
            env,
            engine,
            1,
            [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(90)],
            outputs=2,
            conv="a",
            turn=1,
            prefix=10,
        )
        run_to_idle(env, engine)
        self.assertTrue(second.completion_event.triggered)
        self.assertEqual(second.first_prefill_source.gpu_hit_tokens, 10)

    def test_six_cells_and_exact_payloads(self):
        for cell in ("HH", "FH", "RH", "HF", "FF", "RF"):
            with self.subTest(cell=cell):
                result = run_arrival_cell(cell)
                self.assertEqual(result["cell"], cell)
                self.assertEqual(
                    result["computed_tokens"], 64 if cell[0] == "R" else 32
                )
                self.assertEqual(result["p2p_bytes"], 128 if cell[1] == "H" else 256)
                self.assertTrue(result["first_token"]["source_endpoint_supported"])
                self.assertTrue(result["completed"])

    def test_prefix_crosses_prompt_output_boundary_without_rounding(self):
        env = simpy.Environment()
        engine = engine_for(env, flow=True, clock="arrival", block_size=1, capacity=128)
        first = request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(32)],
            outputs=18,
            conv="a",
            turn=0,
            prefix=0,
        )
        run_to_idle(env, engine)
        inherited = first.handover_context_hash_ids[:37]
        second = request(
            env,
            engine,
            1,
            inherited + [f"b{i}" for i in range(7)],
            outputs=2,
            conv="a",
            turn=1,
            prefix=37,
        )
        run_to_idle(env, engine)
        src = second.first_prefill_source
        d = engine.router.observations[-1]["d_local_tokens"]
        self.assertEqual(
            (src.gpu_hit_tokens, src.cpu_hit_tokens, src.computed_tokens, d),
            (32, 5, 7, 37),
        )
        self.assertIsNone(classify_prefix(37, 32, 5, 37)["cell"])
        store = store_connector(engine.workers[0]).service.store
        self.assertTrue(any(o.ready_at <= env.now for o in store.objects.values()))

    def test_cancel_releases_pending_snapshots(self):
        env = simpy.Environment()
        engine = engine_for(
            env, flow=True, clock="arrival", block_size=1, capacity=128, bandwidth=10000
        )
        first = request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(32)],
            outputs=2,
            conv="a",
            turn=0,
            prefix=0,
        )
        run_to_idle(env, engine)
        second = request(
            env,
            engine,
            1,
            [f"a{i}" for i in range(32)] + [f"b{i}" for i in range(16)],
            outputs=2,
            conv="a",
            turn=1,
            prefix=32,
        )

        def cancel():
            yield env.timeout(0.00001)
            yield engine.cancel_requests([second])[0]

        env.process(cancel())
        run_to_idle(env, engine)
        self.assertTrue(second.cancel_requested)
        self.assertFalse(second.handover_snapshots)
        self.assertIsNotNone(second.cancelled_at)


if __name__ == "__main__":
    unittest.main()
