"""Six-state observations come from live caches, never from trace cell labels."""

import unittest
import simpy
from TokenSim.llm.handover_metrics import classify_prefix, native_timing_observation
from tests.test_handover_p2p import engine_for, request, store_connector, p2p_connector


def evict_idle_cache(worker):
    manager = worker.scheduler.block_manager.kv_cache_manager
    for block in list(manager._cache.values()):
        if block.ref_count:
            raise AssertionError("cannot evict a pinned block")
        manager.evict(block)


def run_cell(cell):
    env = simpy.Environment()
    engine = engine_for(
        env, routing=True, store_admission="never" if cell[0] == "R" else "always"
    )
    first = request(env, engine, 0, ["a", "b"], outputs=2, conv="a", turn=0, prefix=0)
    env.run()
    engine.raise_if_failed()
    if cell[0] != "H":
        evict_idle_cache(engine.workers[0])
    if cell[1] == "F":
        evict_idle_cache(engine.workers[1])
    transferred = p2p_connector(engine.workers[0]).stats.transfer_bytes
    returning = request(
        env, engine, 1, ["a", "b", "c", "d"], outputs=2, conv="a", turn=1, prefix=32
    )
    env.run()
    engine.raise_if_failed()
    obs = engine.router.observations[-1]
    src = returning.first_prefill_source
    result = classify_prefix(
        32, src.gpu_hit_tokens, src.cpu_hit_tokens, obs["d_local_tokens"]
    )
    return dict(
        cell=result["cell"],
        reason=result["reason"],
        p_local_tokens=src.gpu_hit_tokens,
        p_store_tokens=src.cpu_hit_tokens,
        d_local_tokens=obs["d_local_tokens"],
        computed_prompt_tokens=src.computed_tokens,
        p2p_bytes=p2p_connector(engine.workers[0]).stats.transfer_bytes - transferred,
        d_store_load_wait_s=store_connector(engine.workers[1]).stats.load_wait_time,
        timing=native_timing_observation(returning),
        completed=returning.is_done,
        route=obs,
    )


class HandoverStateTest(unittest.TestCase):
    def test_six_diagnostic_cells_use_actual_state_and_missing_bytes(self):
        for cell in ("HH", "FH", "RH", "HF", "FF", "RF"):
            with self.subTest(cell=cell):
                result = run_cell(cell)
                self.assertEqual(result["cell"], cell)
                self.assertTrue(result["completed"])
                self.assertEqual(result["p2p_bytes"], 128 if cell[1] == "H" else 256)
                self.assertEqual(
                    result["computed_prompt_tokens"], 64 if cell[0] == "R" else 32
                )
                self.assertEqual(result["d_store_load_wait_s"], 0)

    def test_mixed_prefix_is_reported_instead_of_forced_into_six_cells(self):
        for values in ((32, 16, 16, 32), (32, 0, 16, 32), (32, 32, 0, 16)):
            self.assertIsNone(classify_prefix(*values)["cell"])

    def test_native_endpoint_is_not_presented_as_the_source_endpoint(self):
        result = run_cell("FF")["timing"]
        self.assertFalse(result["source_endpoint_supported"])
        self.assertIsNone(result["source_first_token_s"])
        self.assertGreater(
            result["handoff_ready_after_arrival_s"], result["native_ttft_s"]
        )


if __name__ == "__main__":
    unittest.main()
