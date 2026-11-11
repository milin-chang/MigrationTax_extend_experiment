"""Ours：HH/FH 用 PPD；HF 在 P 先 decode；FF 用 layer pipeline。

對應 §V、Fig. 6。此處實作 TTFT 決策與成本，不模擬完整 handoff。
"""

from dataclasses import replace

from ..model import CostModel, Turn, TurnResult
from ..PPD import evaluate as ppd


def evaluate(turn: Turn, model: CostModel) -> TurnResult:
    # HH、FH：D 已有前文，沿用 PPD 的當地 prefill。
    if turn.cache_state.decode_hit:
        ppd_result = ppd(turn, model)
        return replace(ppd_result, method="Ours")

    # HF：第一個 token 在 P 產生。完整 input KV 仍會搬到 D，
    # 只是這段搬移不再阻擋第一個 token；不能誤寫成零搬移。
    if turn.cache_state.prefill_hit:
        reference_seconds = model.reference_seconds(turn)
        return TurnResult(
            turn=turn,
            method="Ours",
            path="prefill_side_handoff",
            prefill_device="P",
            first_decode_device="P",
            store_to_prefill_tokens=0,
            prefill_to_decode_tokens=turn.context_tokens,
            ttft_seconds=reference_seconds,
            reference_seconds=reference_seconds,
        )

    # FF：store→P、prefill、P→D 拆成逐層 pipeline。
    store_to_prefill_tokens = turn.prefix_tokens
    prefill_to_decode_tokens = turn.context_tokens
    cache_ready_seconds = model.pipeline_cache_ready_seconds(
        turn, store_to_prefill_tokens, prefill_to_decode_tokens,
    )
    first_step_seconds = model.decode_step_seconds(turn.context_tokens)
    ttft_seconds = cache_ready_seconds + first_step_seconds
    reference_seconds = model.reference_seconds(turn)
    return TurnResult(
        turn=turn,
        method="Ours",
        path="layer_pipeline",
        prefill_device="P",
        first_decode_device="D",
        store_to_prefill_tokens=store_to_prefill_tokens,
        prefill_to_decode_tokens=prefill_to_decode_tokens,
        ttft_seconds=ttft_seconds,
        reference_seconds=reference_seconds,
    )
