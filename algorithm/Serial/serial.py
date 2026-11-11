"""Serial baseline：store→P、prefill、P→D、decode 依序完成（Eq. 1）。"""

from ..model import CostModel, Turn, TurnResult, serial_transfer_tokens


def evaluate(turn: Turn, model: CostModel) -> TurnResult:
    store_to_prefill_tokens, prefill_to_decode_tokens = serial_transfer_tokens(turn)
    fetch_seconds = model.move_seconds(store_to_prefill_tokens)
    prefill_seconds = model.prefill_seconds(turn)
    send_seconds = model.move_seconds(prefill_to_decode_tokens)
    first_step_seconds = model.decode_step_seconds(turn.context_tokens)
    ttft_seconds = fetch_seconds + prefill_seconds + send_seconds + first_step_seconds
    reference_seconds = model.reference_seconds(turn)

    return TurnResult(
        turn=turn,
        method="Serial",
        path="serial",
        prefill_device="P",
        first_decode_device="D",
        store_to_prefill_tokens=store_to_prefill_tokens,
        prefill_to_decode_tokens=prefill_to_decode_tokens,
        ttft_seconds=ttft_seconds,
        reference_seconds=reference_seconds,
    )


def build_plan(turn: Turn, *, layers: int = 32):
    """Describe native operations without analytical completion times."""
    from ..native import build_plan as native_plan

    return native_plan("Serial", turn, layers=layers)
