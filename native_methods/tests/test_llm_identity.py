"""Lossless host identity memo: signed limits, aliasing, eviction and scopes."""

from array import array
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "LLMServingSim"))
from native_llm_components import PackedIdentityMemo


class IdentityMemoTests(unittest.TestCase):
    def test_exact_signed_limits_and_independent_lists(self):
        values = [-(2**63), -1, 0, 1, 2**63 - 1]
        memo = PackedIdentityMemo(Counter())
        first = memo.get("fixed-source", lambda: list(values))
        first[0] = 42
        hit = memo.get("fixed-source", lambda: self.fail("Unexpected recomputation"))
        self.assertIs(type(hit), list)
        self.assertEqual(hit, values)
        hit.clear()
        self.assertEqual(memo.get("fixed-source", lambda: []), values)

    def test_eviction_recomputes_exact_input_and_respects_byte_limit(self):
        capacity = sys.getsizeof(array("q", [0]))
        memo = PackedIdentityMemo(Counter(), byte_limit=capacity)
        calls = Counter()

        def build(key, values):
            calls[key] += 1
            return list(values)

        self.assertEqual(memo.get("a", lambda: build("a", [-17])), [-17])
        self.assertEqual(memo.get("b", lambda: build("b", [23])), [23])
        self.assertEqual(memo.get("a", lambda: build("a", [-17])), [-17])
        self.assertEqual(calls, {"a": 2, "b": 1})
        self.assertGreater(memo.counters["identity_hashes.memo_evictions"], 0)
        self.assertLessEqual(memo.evidence()["peak_allocated_array_bytes"], capacity)

    def test_overflow_fails_without_truncation_and_runtime_scopes_are_independent(self):
        for value in (-(2**63) - 1, 2**63):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PackedIdentityMemo(Counter()).get("overflow", lambda: [value])
        one, two = PackedIdentityMemo(Counter()), PackedIdentityMemo(Counter())
        self.assertEqual(one.get("same-id", lambda: [1]), [1])
        self.assertEqual(two.get("same-id", lambda: [2]), [2])
        disabled = PackedIdentityMemo(Counter(), byte_limit=0)
        self.assertEqual(disabled.get("oversize", lambda: [-17, 23]), [-17, 23])
        self.assertEqual(disabled.evidence()["allocated_array_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
