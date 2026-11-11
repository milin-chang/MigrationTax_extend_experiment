"""Arrival cache decisions stay independent of bandwidth and real readiness."""

import unittest
import simpy
from tests.test_handover_p2p import engine_for, request
from tests.test_handover_clock import ShortDecodeBackend
from TokenSim.latency.base import LatencyBackend


def run_to_idle(env, engine):
    for _ in range(10000):
        if env.peek() == float("inf"):
            break
        env.step()
    else:
        raise AssertionError("arrival-clock flow did not advance")
    engine.raise_if_failed()


def arrival_pressure_case(bandwidth):
    env = simpy.Environment()
    engine = engine_for(
        env,
        flow=True,
        clock="arrival",
        block_size=1,
        capacity=64,
        bandwidth=bandwidth,
        backend=ShortDecodeBackend(),
    )
    a = [f"a{i}" for i in range(32)]
    requests = [request(env, engine, 0, a, outputs=2, conv="a", turn=0, prefix=0)]

    def arrivals():
        yield env.timeout(0.002)
        requests.append(
            request(
                env,
                engine,
                1,
                [f"x{i}" for i in range(48)],
                outputs=2,
                conv="b",
                turn=0,
                prefix=0,
            )
        )
        yield env.timeout(0.0015)
        requests.append(
            request(
                env,
                engine,
                2,
                a + [f"b{i}" for i in range(16)],
                outputs=2,
                conv="a",
                turn=1,
                prefix=32,
            )
        )

    env.process(arrivals())
    run_to_idle(env, engine)
    if not all(r.completion_event.triggered for r in requests):
        raise AssertionError("arrival-clock flow left requests unfinished")
    return engine, requests


class HandoverArrivalTest(unittest.TestCase):
    def test_long_prefill_cannot_be_overtaken_at_its_fixed_d(self):
        class ProportionalBackend(LatencyBackend):
            def estimate_step_latency(self, requests):
                return (
                    requests[0].step_tokens * 0.0001
                    if requests[0].is_prefill
                    else 0.0001
                )

        env = simpy.Environment()
        engine = engine_for(
            env,
            flow=True,
            clock="arrival",
            block_size=1,
            capacity=128,
            workers_per_phase=2,
            backend=ProportionalBackend(),
        )
        engine.router.arrival_state.d = [engine.workers[2]]
        first = request(
            env,
            engine,
            0,
            [f"a{i}" for i in range(60)],
            outputs=2,
            conv="a",
            turn=0,
            prefix=0,
        )
        later = []

        def arrive():
            yield env.timeout(0.0001)
            later.append(
                request(
                    env,
                    engine,
                    1,
                    [f"b{i}" for i in range(10)],
                    outputs=2,
                    conv="b",
                    turn=0,
                    prefix=0,
                )
            )

        env.process(arrive())
        run_to_idle(env, engine)
        second = later[0]
        self.assertLess(
            second.handover_prefill_finished_at, first.handover_prefill_finished_at
        )
        self.assertGreaterEqual(second.handoff_received_at, first.completed_at)
        self.assertTrue(
            first.completion_event.triggered and second.completion_event.triggered
        )

    def test_eight_workers_route_returning_prefixes_to_original_holders(self):
        env = simpy.Environment()
        engine = engine_for(
            env,
            flow=True,
            clock="arrival",
            block_size=1,
            capacity=128,
            workers_per_phase=4,
        )
        for i in range(4):
            request(
                env,
                engine,
                i,
                [f"a{i}:{j}" for j in range(32)],
                outputs=2,
                conv=str(i),
                turn=0,
                prefix=0,
            )
            run_to_idle(env, engine)
        initial = list(engine.router.observations)
        for i in range(4):
            request(
                env,
                engine,
                i + 4,
                [f"a{i}:{j}" for j in range(32)] + [f"b{i}:{j}" for j in range(16)],
                outputs=2,
                conv=str(i),
                turn=1,
                prefix=32,
            )
            run_to_idle(env, engine)
        self.assertEqual(len({r["prefill_worker"] for r in initial}), 4)
        self.assertEqual(len({r["decode_worker"] for r in initial}), 4)
        for first, last in zip(initial, engine.router.observations[4:]):
            self.assertEqual(
                (first["prefill_worker"], first["decode_worker"]),
                (last["prefill_worker"], last["decode_worker"]),
            )

    def test_bandwidth_changes_timing_but_not_arrival_states(self):
        slow, rs = arrival_pressure_case(10000)
        fast, rf = arrival_pressure_case(10**12)
        self.assertEqual(slow.router.observations, fast.router.observations)
        self.assertEqual(rs[-1].first_prefill_source.cpu_hit_tokens, 32)
        self.assertEqual(rf[-1].first_prefill_source.cpu_hit_tokens, 32)
        self.assertGreater(rs[-1].prefill_latency, rf[-1].prefill_latency)

    def test_hit_waits_for_producer_instead_of_using_future_kv(self):
        env = simpy.Environment()
        engine = engine_for(
            env, flow=True, clock="arrival", block_size=1, capacity=64, bandwidth=10000
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
        later = []

        def arrive():
            yield env.timeout(0.0002)
            later.append(
                request(
                    env,
                    engine,
                    1,
                    [f"a{i}" for i in range(32)] + [f"b{i}" for i in range(16)],
                    outputs=2,
                    conv="a",
                    turn=1,
                    prefix=32,
                )
            )

        env.process(arrive())
        run_to_idle(env, engine)
        second = later[0]
        self.assertEqual(engine.router.observations[-1]["p_local_tokens"], 32)
        self.assertEqual(engine.router.observations[-1]["d_local_tokens"], 32)
        self.assertEqual(second.first_prefill_source.gpu_hit_tokens, 32)
        self.assertGreaterEqual(second.handover_prefill_finished_at, first.completed_at)
        self.assertTrue(second.completion_event.triggered)


if __name__ == "__main__":
    unittest.main()
