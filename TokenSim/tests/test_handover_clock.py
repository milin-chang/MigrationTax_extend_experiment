"""A completion-clock counterexample keeps the arrival-clock gate explicit."""

import unittest
import simpy
from TokenSim.latency.base import LatencyBackend
from tests.test_handover_p2p import engine_for, request


class ShortDecodeBackend(LatencyBackend):
    def estimate_step_latency(self, requests):
        return 0.001 if requests[0].is_prefill else 0.0001


def completion_clock_case(bandwidth):
    env = simpy.Environment()
    engine = engine_for(
        env,
        routing=True,
        capacity=64,
        bandwidth=bandwidth,
        backend=ShortDecodeBackend(),
    )
    requests = [
        request(env, engine, 0, ["a", "b"], outputs=2, conv="a", turn=0, prefix=0)
    ]

    def arrivals():
        yield env.timeout(0.002)
        requests.append(
            request(
                env, engine, 1, ["x", "y", "z"], outputs=2, conv="b", turn=0, prefix=0
            )
        )
        yield env.timeout(0.0015)
        requests.append(
            request(
                env, engine, 2, ["a", "b", "c"], outputs=2, conv="a", turn=1, prefix=32
            )
        )

    env.process(arrivals())
    for _ in range(5000):
        if env.peek() == float("inf"):
            break
        env.step()
    else:
        raise AssertionError("clock diagnostic failed to make progress")
    engine.raise_if_failed()
    if not all(r.is_done for r in requests):
        raise AssertionError("clock diagnostic left requests unfinished")
    return {
        "bandwidth": bandwidth,
        "routing": engine.router.observations,
        "requests": [
            {
                "id": r.id,
                "arrival_s": r.arrival_at,
                "completed_s": r.completed_at,
                "p_local_tokens": r.first_prefill_source.gpu_hit_tokens,
                "p_store_tokens": r.first_prefill_source.cpu_hit_tokens,
            }
            for r in requests
        ],
    }


class HandoverClockTest(unittest.TestCase):
    def test_completion_driven_states_are_not_claimed_arrival_clock_equivalent(self):
        slow = completion_clock_case(10000)
        fast = completion_clock_case(10**12)
        self.assertEqual(
            [r["arrival_s"] for r in slow["requests"]],
            [r["arrival_s"] for r in fast["requests"]],
        )
        self.assertNotEqual(
            slow["routing"][-1]["p_local_tokens"], fast["routing"][-1]["p_local_tokens"]
        )
        self.assertEqual(slow["routing"][-1]["clock"], "completion")


if __name__ == "__main__":
    unittest.main()
