"""Explicit source PF/ST cost adapter for serial, cached-prompt experiments."""

from TokenSim.errors import ConfigurationError
from TokenSim.latency.base import LatencyBackend
from TokenSim.config.validation import number


def prefill_seconds(tokens):
    number(tokens, "prefill tokens")
    return (2 * 6.7e9 * tokens + 4 * 32 * 4096 * tokens * tokens) / (0.7 * 384e12)


def decode_seconds(context):
    number(context, "decode context")
    return 0.0029098 + 0.2072e-6 * context


class HandoverCostBackend(LatencyBackend):
    """PF(new tokens), telescoped across chunks; one ST(C) per output step.

    Cached attention follows the supplied PF(n) abstraction, not a calibrated
    attention-kernel model. Batching is rejected, rather than priced silently.
    """

    def estimate_step_latency(self, requests):
        if len(requests) != 1:
            raise ConfigurationError("PF/ST adapter requires max_num_seqs=1")
        req = requests[0]
        if req.is_prefill:
            total = req.prefill_compute_len
            start = max(0, req.num_computed_tokens - (req.prefill_len - total))
            end = min(total, start + req.step_tokens)
            return prefill_seconds(end) - prefill_seconds(min(start, total))
        if req.needs_recompute:
            return prefill_seconds(
                req.num_computed_tokens + req.step_tokens
            ) - prefill_seconds(req.num_computed_tokens)
        return decode_seconds(req.context_len)

    def describe(self):
        return {
            "backend": "handover_pf_st",
            "prefill": "PF(new tokens), cumulative across chunks",
            "decode": "ST(context before output)",
            "batching": "serial",
            "calibrated_gpu_profile": False,
        }
