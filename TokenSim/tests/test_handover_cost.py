"""Source cost checkpoints and the actual P-to-D first-output endpoint."""

import unittest
import simpy
from TokenSim.latency.handover import prefill_seconds, decode_seconds
from tests.test_handover_p2p import engine_for
from TokenSim.llm.llm_request import Request


class HandoverCostTest(unittest.TestCase):
    def test_source_context_checkpoints_and_chunked_runtime(self):
        for context in (1000, 8000, 32000, 120000):
            with self.subTest(context=context):
                # Independent arithmetic from the handover constants.
                pf = (13400000000 * context + 524288 * context**2) / 268800000000000
                st = 0.0029098 + 0.0000002072 * context
                self.assertAlmostEqual(prefill_seconds(context), pf)
                self.assertAlmostEqual(decode_seconds(context), st)
                env = simpy.Environment()
                engine = engine_for(
                    env,
                    flow=True,
                    capacity=context + 32,
                    bandwidth=10**12,
                    cost_model="pf_st",
                    overhead_s=0.001,
                    max_num_seqs=1,
                )
                # The micro-model has 4 bytes/token; compute is the supplied PF/ST adapter.
                hashes = [f"input:{i}" for i in range((context + 15) // 16)]
                req = Request(
                    id=0,
                    prefill_len=context,
                    decode_len=1,
                    block_size=16,
                    hash_ids=hashes,
                    chat_id="a",
                    turn=0,
                    prefix_tokens=0,
                )
                req.arrive(env)
                engine.add_requests([req])
                env.run()
                engine.raise_if_failed()
                expected = 0.001 + pf + st + context * 4 / 10**12
                self.assertAlmostEqual(req.prefill_latency, expected, places=9)
                self.assertAlmostEqual(req.handover_first_decode_service_s, st)
                self.assertTrue(req.completion_event.triggered)


if __name__ == "__main__":
    unittest.main()
