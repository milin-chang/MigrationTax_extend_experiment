"""Classify observable prefix coverage without hiding mixed block states."""

from TokenSim.errors import ConfigurationError


def classify_prefix(prefix_tokens, p_local_tokens, p_store_tokens, d_local_tokens):
    values = (prefix_tokens, p_local_tokens, p_store_tokens, d_local_tokens)
    if any(type(value) is not int or value < 0 for value in values):
        raise ConfigurationError(
            "prefix coverage must contain nonnegative token counts"
        )
    if prefix_tokens == 0:
        return {"cell": None, "reason": "no inherited prefix"}
    if p_local_tokens >= prefix_tokens:
        p = "H"
    elif p_local_tokens or 0 < p_store_tokens < prefix_tokens:
        return {"cell": None, "reason": "mixed P hit/fetch/recompute coverage"}
    elif p_store_tokens >= prefix_tokens:
        p = "F"
    else:
        p = "R"
    if d_local_tokens >= prefix_tokens:
        d = "H"
    elif d_local_tokens:
        return {"cell": None, "reason": "partial D prefix coverage"}
    else:
        d = "F"
    return {"cell": p + d, "reason": "full-prefix coverage observed"}


def native_timing_observation(req):
    ready = getattr(req, "handoff_ready_at", None)
    if getattr(req, "handover_flow", False):
        return dict(
            native_ttft_s=req.prefill_latency,
            prefill_completion_after_arrival_s=req.handover_prefill_finished_at
            - req.arrival_at,
            handoff_ready_after_arrival_s=(
                None if ready is None else ready - req.arrival_at
            ),
            source_first_token_s=req.prefill_latency,
            source_endpoint_supported=True,
            endpoint="first output sampled at D after transfer; includes queue and configured overhead",
            original_native_endpoint_preserved=False,
        )
    return {
        "native_ttft_s": req.prefill_latency,
        "handoff_ready_after_arrival_s": (
            None if ready is None else ready - req.arrival_at
        ),
        "source_first_token_s": None,
        "source_endpoint_supported": False,
        "reason": "Native P already samples a token; a post-transfer first decode step is not implemented.",
    }
