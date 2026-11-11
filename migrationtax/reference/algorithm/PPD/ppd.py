"""MigrationTax 的 PPD baseline；不是原 PPD 的完整 serving system。

D 命中：直接在 D prefill。D 未命中：依本文假設從 store fetch，
走 Serial；不偷偷替換成 prefix recompute 或 direct-to-D 敏感度版本。
"""

from dataclasses import replace

from ..model import CostModel, Turn, TurnResult
from ..Serial import evaluate as serial


def evaluate(turn: Turn, model: CostModel) -> TurnResult:
    if not turn.cache_state.decode_hit:
        serial_result = serial(turn, model)
        return replace(serial_result, method="PPD", path="serial_fallback")

    reference_seconds = model.reference_seconds(turn)
    return TurnResult(
        turn=turn,
        method="PPD",
        path="decode_local_prefill",
        prefill_device="D",
        first_decode_device="D",
        store_to_prefill_tokens=0,
        prefill_to_decode_tokens=0,
        ttft_seconds=reference_seconds,
        reference_seconds=reference_seconds,
    )
