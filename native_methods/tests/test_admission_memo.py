"""Capacity memo invalidation and the native P queue tie-break exception."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "splitwise-sim"))
from native_admission_memo import AdmissionMemo
from native_method_cache import MethodKVCache
from kv_cache import Block


class AdmissionMemoTests(unittest.TestCase):
    def fixture(self):
        devices = [SimpleNamespace(cache=MethodKVCache(str(i), 64)) for i in range(8)]
        sim = SimpleNamespace(
            devices=devices,
            store=MethodKVCache("store", None),
            counters=Counter(),
            pending=[{"source_id": "0"}],
        )
        return sim, AdmissionMemo(sim), Block("wanted", 1, 16)

    def fill(self, device):
        blocks = [Block("busy", 4, 64)]
        device.cache.reserve(blocks, "active", 0)
        device.cache.publish(blocks, "active", 0)

    def test_alternate_p_becomes_selectable_without_kv_change(self):
        sim, memo, wanted = self.fixture()
        self.fill(sim.devices[0])
        world = memo.world()
        self.assertFalse(sim.devices[0].cache.can_reserve([wanted], "source"))
        self.assertTrue(sim.devices[1].cache.can_reserve([wanted], "source"))
        memo.record_failure("0", [wanted], sim.devices[4], [wanted], "source", world)
        memo.finish_wake()
        self.assertFalse(memo.skip_row("0", world))
        self.assertFalse(memo.skip_wake(world))

    def test_impossible_d_ignores_p_queue_until_any_dependency_changes(self):
        sim, memo, wanted = self.fixture()
        self.fill(sim.devices[4])
        world = memo.world()
        memo.record_failure("0", [wanted], sim.devices[4], [wanted], "source", world)
        memo.finish_wake()
        self.assertTrue(memo.skip_row("0", world))
        self.assertTrue(memo.skip_wake(world))
        for cache in [*(d.cache for d in sim.devices), sim.store]:
            cache.mutation_version += 1
            self.assertFalse(memo.skip_row("0", memo.world()))
            self.assertFalse(memo.skip_wake(memo.world()))
            cache.mutation_version -= 1
        memo.pending_added()
        self.assertFalse(memo.skip_wake(memo.world()))
        sim.devices[4].cache.release("active", 1)
        self.assertTrue(sim.devices[4].cache.can_reserve([wanted], "source"))
        self.assertFalse(memo.skip_row("0", memo.world()))

    def test_all_p_impossible_and_capacity_change_invalidates(self):
        sim, memo, wanted = self.fixture()
        for d in sim.devices[:4]:
            self.fill(d)
        memo.record_failure(
            "0", [wanted], sim.devices[4], [wanted], "source", memo.world()
        )
        memo.finish_wake()
        self.assertTrue(memo.skip_wake(memo.world()))
        sim.devices[0].cache.capacity_bytes = 80
        self.assertFalse(memo.skip_wake(memo.world()))


if __name__ == "__main__":
    unittest.main()
