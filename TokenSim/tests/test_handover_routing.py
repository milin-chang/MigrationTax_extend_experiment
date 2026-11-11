"""Routing decisions use explicit prefix lengths and remain fixed per request."""

import unittest
from types import SimpleNamespace
from TokenSim.block.block_manager import BlockManager
from TokenSim.errors import ConfigurationError, WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.placement.handover import HandoverRouter
from TokenSim.workload.user_request import UserRequest


def worker(wid):
    manager = BlockManager(
        block_size=16, num_gpu_blocks=8, num_cpu_blocks=0, model="m", watermark=0
    )
    return SimpleNamespace(
        id=wid,
        cache_config=SimpleNamespace(model="m"),
        scheduler=SimpleNamespace(block_manager=manager),
    )


def req(rid, conv, turn, prefix, hashes):
    return Request(
        id=rid,
        prefill_len=16 * len(hashes),
        decode_len=2,
        block_size=16,
        chat_id=conv,
        turn=turn,
        prefix_tokens=prefix,
        hash_ids=hashes,
    )


def cache(w, r):
    manager = w.scheduler.block_manager
    manager.allocate(r)
    manager.commit_input_cache(r)
    manager.free(r)


class HandoverRoutingTest(unittest.TestCase):
    def setUp(self):
        self.p = [worker(0), worker(1)]
        self.d = [worker(2), worker(3)]
        self.clock = [0.0]
        self.router = HandoverRouter(self.p, self.d, lambda: self.clock[0])

    def test_prefix_holder_and_previous_d_are_selected_once(self):
        first = req(0, "a", 0, 0, ["a", "b"])
        self.router.select(first)
        cache(self.p[1], first)
        cache(self.d[0], first)
        second = req(1, "a", 1, 32, ["a", "b", "c"])
        self.clock[0] = 7
        self.assertIs(self.router.select(second), self.p[1])
        self.assertEqual(self.router.batches([second])[0][0].id, 2)
        self.assertEqual(self.router.observations[-1]["arrival_decision_s"], 7)
        cache(self.p[0], second)
        self.assertIs(self.router.select(second), self.p[1])
        self.assertEqual(len(self.router.observations), 2)

    def test_new_conversation_uses_fewer_resident_d_caches(self):
        first = req(0, "a", 0, 0, ["a", "b"])
        self.router.select(first)
        cache(self.d[0], first)
        other = req(1, "b", 0, 0, ["x", "y"])
        self.router.select(other)
        self.assertEqual(self.router.batches([other])[0][0].id, 3)

    def test_d_placement_is_per_request_not_only_per_batch(self):
        first = req(0, "a", 0, 0, ["a", "b"])
        self.router.select(first)
        cache(self.d[0], first)
        other = req(1, "b", 0, 0, ["x", "y"])
        self.router.select(other)
        groups = self.router.batches([first, other])
        self.assertEqual(
            [(w.id, [r.id for r in rs]) for w, rs in groups], [(2, [0]), (3, [1])]
        )

    def test_partial_prefix_is_not_silently_a_complete_hit(self):
        first = req(0, "a", 0, 0, ["a", "b"])
        self.router.select(first)
        fragment = req(10, "a", 0, 0, ["a"])
        cache(self.p[1], fragment)
        second = req(1, "a", 1, 24, ["a", "b", "c"])
        self.router.select(second)
        self.assertTrue(self.router.observations[-1]["partial_prefix"])
        self.assertLess(self.router.observations[-1]["p_local_tokens"], 24)

    def test_missing_or_duplicate_ancestry_fails(self):
        with self.assertRaises(ConfigurationError):
            self.router.select(req(0, "a", 1, 16, ["a", "b"]))
        first = req(0, "a", 0, 0, ["a", "b"])
        self.router.select(first)
        with self.assertRaises(ConfigurationError):
            self.router.select(req(1, "a", 0, 0, ["a", "b"]))
        with self.assertRaises(ConfigurationError):
            self.router.select(req(0, "b", 0, 0, ["x", "y"]))

    def test_prefix_length_survives_workload_conversion(self):
        source = UserRequest(
            request_id=0,
            prefill_len=48,
            decode_len=2,
            chat_id="a",
            turn=1,
            prefix_tokens=32,
            hash_ids=["a", "b", "c"],
        )
        self.assertEqual(source.to_request(16).prefix_tokens, 32)
        for value in (-1, 49, True):
            with self.assertRaises((WorkloadValidationError, ConfigurationError)):
                UserRequest(
                    request_id=0, prefill_len=48, decode_len=2, prefix_tokens=value
                )


if __name__ == "__main__":
    unittest.main()
