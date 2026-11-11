"""給定 returning turns 的共同狀態，重算三方法與 Fig. 4／7 統計。

本模組只接收 Turn，不讀 trace、作者 dump、目標比例或路由 RNG。
輸入須為非空的 returning-turn 清單；首次請求在呼叫前排除。
"""

from .algorithm import CacheState, CostModel, Turn
from .algorithm.comparison import METHODS, compare_turns, summarize


def evaluate_turns(turns: list[Turn], seed: int, model: CostModel) -> dict:
    """供作者 dump 與自行重建狀態共用的統計；不讀取任何參考答案。"""
    results = compare_turns(turns, model)
    metrics = {}
    for method in METHODS:
        method_results = []
        for result in results:
            if result.method == method:
                method_results.append(result)
        metrics[method] = summarize(method_results, threshold=2.0)

    returning_count = len(turns)
    fig4 = {}
    serial_counts = metrics["Serial"]["by_cache_state"]
    for state in CacheState:
        counts = serial_counts[state.value]
        cell_count = counts["returning_turns"]
        affected_count = counts["affected_turns"]
        within = None
        if cell_count > 0:
            within = 100 * affected_count / cell_count
        fig4[state.value] = {
            "returning_turns": cell_count, "affected_turns": affected_count,
            "cell_share_percent": 100 * cell_count / returning_count,
            "affected_within_cell_percent": within,
        }

    fig7 = {}
    for method, summary in metrics.items():
        stacks = {}
        for state in CacheState:
            affected_count = summary["by_cache_state"][state.value]["affected_turns"]
            stacks[state.value] = 100 * affected_count / returning_count
        fig7[method] = {
            "affected_turns": summary["affected_turns"],
            "affected_share_percent": 100 * summary["affected_turns"] / returning_count,
            "stack_contribution_percent": stacks,
        }

    # 早期開發是對齊 Fig 4、Fig 7 所以這編繪有
    return {
        "seed": seed, "returning_turns": returning_count,
        "fig4": fig4, "fig7": fig7,
    }

