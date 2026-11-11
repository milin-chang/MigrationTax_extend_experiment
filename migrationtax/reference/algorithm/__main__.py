"""從專案根目錄執行：python3 -m algorithm --demo。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .comparison import METHODS, compare_turns, summarize
from .model import CostModel, Turn, TurnResult


def load_turns(path: Path) -> list[Turn]:
    """只接受明確標註 cache_state 的標準 JSONL，不猜原始 trace 語意。"""
    turns = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each line must be a JSON object")
                turn = Turn(**row)
                turns.append(turn)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return turns


def build_parser() -> argparse.ArgumentParser:
    """集中定義命令列選項；參數名稱與預設值維持不變。"""
    parser = argparse.ArgumentParser(description="比較 MigrationTax 的三種 TTFT 方法")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true", help="執行內附五個假設案例")
    inputs.add_argument("--input", type=Path, help="明確標註 cache_state 的標準 JSONL")
    parser.add_argument("--output", type=Path, help="保存逐輪結果、設定與摘要為 JSON")
    parser.add_argument("--details", action="store_true", help="顯示每輪三方法的 TTFT")
    parser.add_argument(
        "--bandwidth-gb-s", type=float, default=100.0,
        help="每裝置 ingress，十進位 GB/s",
    )
    parser.add_argument(
        "--kv-bytes-per-token", type=int, default=524_000,
        help="正文 0.524 MB = 524000 bytes",
    )
    parser.add_argument("--threshold", type=float, default=2.0, help="affected 的嚴格大於門檻")
    return parser


def build_report(
    turns: list[Turn],
    results: list[TurnResult],
    model: CostModel,
    input_path: Path,
    synthetic_demo: bool,
    threshold: float,
) -> dict:
    """分別彙整三種方法，再組合設定、統計與逐輪輸出。"""
    metrics = {}
    for method_name in METHODS:
        method_results = []
        for result in results:
            if result.method == method_name:
                method_results.append(result)
        metrics[method_name] = summarize(method_results, threshold)

    serialized_results = [result.to_dict(threshold) for result in results]
    return {
        "scope": "Analytic TTFT with supplied fixed cache states; not full trace reproduction.",
        "input": str(input_path),
        "synthetic_demo": synthetic_demo,
        "cost_model": asdict(model),
        "time_unit": "seconds",
        "affected_threshold": threshold,
        "input_turns": len(turns),
        "metrics": metrics,
        "results": serialized_results,
    }


def print_summary(report: dict) -> None:
    """輸出每種方法的 returning-turn 統計。"""
    title = "Analytic TTFT | fixed cache states"
    if report["synthetic_demo"]:
        title += " | synthetic demo"
    print(title)

    input_turns = report["input_turns"]
    threshold = report["affected_threshold"]
    print(f"Input turns: {input_turns} | threshold: >{threshold:g}x")
    print(f"{'Method':<9} {'Returning':>10} {'Affected':>10} {'Share':>10}")

    for method_name, summary in report["metrics"].items():
        share = summary["affected_share"]
        if share is None:
            share_label = "N/A"
        else:
            share_label = f"{share:.1%}"

        returning_count = summary["returning_turns"]
        affected_count = summary["affected_turns"]
        print(
            f"{method_name:<9} {returning_count:>10} "
            f"{affected_count:>10} {share_label:>10}"
        )


def print_details(results: list[TurnResult]) -> None:
    """逐輪結果只在顯示時轉成毫秒，保存的時間仍為秒。"""
    print(f"\n{'Request':<14} {'State':<6} {'Method':<8} {'TTFT ms':>10} {'Slowdown':>10}  Path")
    for result in results:
        request_label = result.turn.request_id or "-"
        state_label = result.turn.cache_state.value
        ttft_milliseconds = result.ttft_seconds * 1000
        print(
            f"{request_label:<14} {state_label:<6} "
            f"{result.method:<8} {ttft_milliseconds:>10.3f} "
            f"{result.slowdown:>9.3f}x  {result.path}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        input_path = Path(__file__).parent / "examples" / "four_states.jsonl"
    else:
        input_path = args.input

    try:
        model = CostModel(
            bandwidth_gb_s=args.bandwidth_gb_s,
            kv_bytes_per_token=args.kv_bytes_per_token,
        )
        turns = load_turns(input_path)
        results = compare_turns(turns, model)
        report = build_report(
            turns=turns,
            results=results,
            model=model,
            input_path=input_path,
            synthetic_demo=args.demo,
            threshold=args.threshold,
        )
        if args.output:
            report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report_json, encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print_summary(report)
    if args.demo or args.details:
        print_details(results)
    if args.output:
        print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
