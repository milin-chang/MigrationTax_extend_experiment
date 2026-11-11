"""讓三種方法接受相同的 Turn；指標只以 returning turns 為分母。"""

from collections.abc import Iterable
import math
from statistics import median

from .model import CacheState, CostModel, Turn, TurnResult
from .Serial import evaluate as serial
from .PPD import evaluate as ppd
from .MigrationTax import evaluate as ours


METHODS = {"Serial": serial, "PPD": ppd, "Ours": ours}


def compare_turns(turns: Iterable[Turn], model: CostModel) -> list[TurnResult]:
    """不重新計算 cache 狀態，不對不同方法另外抽樣請求。"""
    results = []
    for turn in turns:
        for evaluate in METHODS.values():
            result = evaluate(turn, model)
            results.append(result)
    return results


def summarize(results: Iterable[TurnResult], threshold: float = 2.0) -> dict:
    """彙整單一方法的結果；affected_share 用 0–1 比例表示。"""
    if not math.isfinite(threshold) or threshold <= 1:
        raise ValueError("threshold must be finite and greater than 1")
    returning_results = []
    affected_results = []
    for result in results:
        if not result.turn.is_returning:
            continue
        returning_results.append(result)
        if result.is_affected(threshold):
            affected_results.append(result)

    returning_count = len(returning_results)
    affected_count = len(affected_results)
    reference_total = sum(result.reference_seconds for result in returning_results)
    actual_total = sum(result.ttft_seconds for result in returning_results)

    # 沒有資料與受影響比例為 0 是不同情況，JSON 分別輸出 null 與 0。
    affected_share = None
    summed_ttft_inflation = None
    if returning_count > 0:
        affected_share = affected_count / returning_count
        summed_ttft_inflation = (actual_total - reference_total) / reference_total

    median_affected_slowdown = None
    if affected_count > 0:
        affected_slowdowns = [result.slowdown for result in affected_results]
        median_affected_slowdown = median(affected_slowdowns)

    by_cache_state = {}
    for state in CacheState:
        by_cache_state[state.value] = {"returning_turns": 0, "affected_turns": 0}
    for result in returning_results:
        state_counts = by_cache_state[result.turn.cache_state.value]
        state_counts["returning_turns"] += 1
    for result in affected_results:
        state_counts = by_cache_state[result.turn.cache_state.value]
        state_counts["affected_turns"] += 1

    return {
        "returning_turns": returning_count,
        "affected_turns": affected_count,
        "affected_share": affected_share,
        "median_affected_slowdown": median_affected_slowdown,
        "summed_ttft_inflation": summed_ttft_inflation,
        "by_cache_state": by_cache_state,
    }
