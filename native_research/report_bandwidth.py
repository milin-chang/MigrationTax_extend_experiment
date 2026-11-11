"""Export audited bandwidth runs and paired affected shares without rerunning."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "native_research/runs/servegen7b-bw-s0-8000-01"
OUTPUT = ROOT / "native_research/reports/servegen7b-bw-s0-8000-01"
METHODS = ("Serial", "PPD", "MigrationTax")


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def completed(run):
    records = []
    for condition in sorted(run.glob("*/*/seed-0/condition.json")):
        output = condition.parent
        receipt = output.with_name(output.name + "-job")
        if not (receipt / "audit.json").is_file():
            continue
        audit = read(receipt / "audit.json")
        if audit["status"] != "passed":
            continue
        info, summary = read(condition), read(output / "summary.json")
        if info["status"] != "completed" or summary["status"] != "completed":
            continue
        with (output / "requests.jsonl").open() as stream:
            requests = {row["source_id"]: row for row in map(json.loads, stream)}
        assert len(requests) == summary["completed"] == summary["input_requests"]
        native = read(next((output / "native").glob("result_*.json")))
        config = read(output / "effective-config.json")
        values = list(requests.values())
        ttft = [r["native_prefill_latency_s"] for r in values]
        d_ttft = [r["d_output_sampled_latency_s"] for r in values]
        assert all(value is not None and value > 0 for value in d_ttft)
        lifecycle = [r["native_total_time_s"] for r in values]
        tpot = [
            r["native_decode_time_sum_s"] / max(1, r["generated"] - 1) for r in values
        ]
        duration = summary["simulated_end_s"]
        records.append(
            dict(
                bandwidth_label=config["bandwidth_study"]["label"],
                method=info["method"],
                bandwidth_gb_per_s=config["bandwidth_study"]["bandwidth_bytes_per_s"]
                / 1e9,
                reference=config["bandwidth_study"]["reference"],
                requests=len(values),
                returning_requests=sum(r["source_turn"] > 0 for r in values),
                model_id=config["hardware_alignment"]["model_id"],
                model_dtype=config["hardware_alignment"]["model_dtype"],
                tensor_parallel=config["hardware_alignment"]["tensor_parallel"],
                physical_gpus=config["hardware_alignment"]["total_gpus"],
                block_tokens=config["simulation"]["block_size"],
                input_selection=config["bandwidth_study"].get(
                    "input_selection",
                    "先排除超過單 GPU 容量的整段對話，再取按原到達時間排序的前 8,000 筆；沒有截短 token、重縮時間或調整負載。",
                ),
                mean_ttft_s=statistics.mean(ttft),
                median_ttft_s=native["prefill_time"]["p50"],
                p95_ttft_s=sorted(ttft)[math.ceil(0.95 * len(ttft)) - 1],
                p99_ttft_s=native["prefill_time"]["p99"],
                max_ttft_s=max(ttft),
                mean_d_first_sampled_s=statistics.mean(d_ttft),
                p95_d_first_sampled_s=sorted(d_ttft)[math.ceil(0.95 * len(d_ttft)) - 1],
                mean_d_handoff_available_s=statistics.mean(
                    r["d_output_visible_latency_s"] for r in values
                ),
                mean_lifecycle_s=statistics.mean(lifecycle),
                p50_lifecycle_s=native["request_time"]["p50"],
                p99_lifecycle_s=native["request_time"]["p99"],
                max_lifecycle_s=max(lifecycle),
                mean_tpot_s=statistics.mean(tpot),
                p50_tpot_s=native["decode_time"]["p50"],
                p99_tpot_s=native["decode_time"]["p99"],
                max_tpot_s=max(tpot),
                output_tokens_per_s=sum(r["generated"] for r in values) / duration,
                requests_per_s=len(values) / duration,
                operator_match_type_counts=native.get("operator_match_type_counts", {}),
                wall_seconds=audit["wall_seconds"],
                peak_rss_gib=audit["peak_rss_bytes"] / 2**30,
                input_sha256=info["input_sha256"],
                source=str(output),
                rows=requests,
            )
        )
    return records


def pair(records, destination):
    reference = next((r for r in records if r["reference"]), None)
    if reference is None:
        return []
    comparisons = []
    for result in records:
        if result["reference"]:
            write_csv(
                destination / "reference-Serial-requests.csv",
                list(result["rows"].values()),
            )
            continue
        assert result["input_sha256"] == reference["input_sha256"]
        assert result["rows"].keys() == reference["rows"].keys()
        exported, returning = [], []
        for source, row in result["rows"].items():
            baseline = reference["rows"][source]
            assert (
                row["source_turn"] == baseline["source_turn"]
                and row["arrival_s"] == baseline["arrival_s"]
            )
            is_returning = row["source_turn"] > 0
            ratio_d = (
                row["d_output_sampled_latency_s"]
                / baseline["d_output_sampled_latency_s"]
            )
            ratio_client = (
                row["native_prefill_latency_s"] / baseline["native_prefill_latency_s"]
            )
            item = dict(
                **row,
                reference_d_latency_s=baseline["d_output_sampled_latency_s"],
                reference_client_ttft_s=baseline["native_prefill_latency_s"],
                slowdown_d=ratio_d,
                paired_d_endpoint="first native output sampled on decode worker",
                slowdown_client=ratio_client,
                affected_d=(
                    (
                        row["d_output_sampled_latency_s"]
                        > 2 * baseline["d_output_sampled_latency_s"]
                    )
                    if is_returning
                    else None
                ),
                affected_client=(
                    (
                        row["native_prefill_latency_s"]
                        > 2 * baseline["native_prefill_latency_s"]
                    )
                    if is_returning
                    else None
                ),
                returning=is_returning,
            )
            exported.append(item)
            if is_returning:
                returning.append(item)
        n = len(returning)
        assert n == result["returning_requests"] > 0
        comparisons.append(
            dict(
                bandwidth_label=result["bandwidth_label"],
                bandwidth_gb_per_s=result["bandwidth_gb_per_s"],
                method=result["method"],
                returning_requests=n,
                affected_d_count=sum(r["affected_d"] for r in returning),
                affected_d_percent=100 * sum(r["affected_d"] for r in returning) / n,
                affected_client_count=sum(r["affected_client"] for r in returning),
                affected_client_percent=100
                * sum(r["affected_client"] for r in returning)
                / n,
                decode_miss_at_arrival_count=sum(
                    r["decode_miss_at_original_arrival"] for r in returning
                ),
                decode_miss_at_arrival_percent=100
                * sum(r["decode_miss_at_original_arrival"] for r in returning)
                / n,
                zero_d_prefix_at_arrival_percent=100
                * sum(
                    r["inherited_prefix_tokens"] > 0
                    and r["d_local_tokens_at_original_arrival"] == 0
                    for r in returning
                )
                / n,
                partial_d_prefix_at_arrival_percent=100
                * sum(
                    0
                    < r["d_local_tokens_at_original_arrival"]
                    < r["inherited_prefix_tokens"]
                    for r in returning
                )
                / n,
                d_choice_changed_after_arrival_count=sum(
                    r["decode_choice_changed_before_admission"] for r in returning
                ),
            )
        )
        write_csv(
            destination
            / f'{result["bandwidth_label"]}-{result["method"]}-requests.csv',
            exported,
        )
    return comparisons


def render(records, comparisons):
    first = records[0] if records else None
    population = first["requests"] if first else "待完成"
    setup = (
        f'Seed 0；{first["model_id"]} {first["model_dtype"]}；H100 SXM；'
        f'TP{first["tensor_parallel"]}／PP1；4P＋4D 共 {first["physical_gpus"]} GPU；'
        f'{first["block_tokens"]}-token KV block。'
        if first
        else "尚無完成且稽核通過的條件。"
    )
    lines = [
        f"# TokenSim：ServeGen {population} 筆通訊頻寬實驗",
        "",
        setup,
        "",
        f"已完成並通過稽核：{len(records)}/10 組。以下只納入完成且稽核通過的條件。",
        "",
        "通訊頻寬使用十進位 GB/s、單向；HBM 頻寬固定 3,352 GB/s。高頻寬 Serial 參考為 1e15 bytes/s。",
        "",
    ]
    for title, key in (
        ("Affected share：第一個真正由 D 採樣的輸出", "affected_d_percent"),
        ("Affected share：使用者首 token", "affected_client_percent"),
        (
            "Decode miss share：原始到達時的 GPU resident prefix",
            "decode_miss_at_arrival_percent",
        ),
    ):
        lines += [
            "## " + title,
            "",
            "| 通訊頻寬 GB/s | Serial % | PPD % | MigrationTax % |",
            "|---:|---:|---:|---:|",
        ]
        for bandwidth in (50, 100, 200):
            group = {
                r["method"]: r[key]
                for r in comparisons
                if r["bandwidth_gb_per_s"] == bandwidth
            }
            values = [f"{group[m]:.6f}" if m in group else "pending" for m in METHODS]
            lines.append(f"| {bandwidth} | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "## 請求與整批指標",
        "",
        "| 頻寬 GB/s | 方法 | Avg client TTFT s | TTFT P95 s | Avg D-first s | Avg lifecycle s | Avg TPOT s/token | Output tokens/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        records, key=lambda r: (r["bandwidth_gb_per_s"], METHODS.index(r["method"]))
    ):
        label = "reference" if row["reference"] else str(int(row["bandwidth_gb_per_s"]))
        values = [
            f"{row[k]:.6f}"
            for k in (
                "mean_ttft_s",
                "p95_ttft_s",
                "mean_d_first_sampled_s",
                "mean_lifecycle_s",
                "mean_tpot_s",
                "output_tokens_per_s",
            )
        ]
        lines.append(
            "| " + label + " | " + row["method"] + " | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "## 統計口徑",
        "",
        (
            f'- {first["requests"]:,} 筆請求含 {first["returning_requests"]:,} 個 returning turns；'
            + first["input_selection"]
            if first
            else "- 請求數與取樣規則待完成條件輸出。"
        ),
        "- Affected 使用同一個 source request 對照 Serial 高頻寬參考；延遲嚴格大於參考的兩倍才算 affected，分母為全部 returning turns。",
        "- D-first 主指標採真正的第一個 D worker 輸出事件，從原始到達計算。本子集輸出至少 4 tokens，全部請求都有 D 採樣。HF 的前兩個輸出可在 P 產生，因此第一個 D 輸出可能是整段回覆的第三個 token。KV 交接／ownership 時間另作 D availability proxy 保存，不取代 D 採樣事件。",
        "- Client TTFT 保留原生使用者首輸出時間，HF 可在 P 先輸出。兩種 endpoint 分開報告。",
        "- Decode miss 來自原始到達時對每個 D 的快取快照，再對應實際目的 D；不是 store load 或 admission 後的命中。到達時候選 D 與最後目的 D 的差異也保留於 CSV。",
        "- 上表 decode miss 採整段 inherited prefix 未完整命中，包含部分 prefix 命中。comparison.csv 與 results.json 另列零覆蓋與部分覆蓋比例，保留 block 粒度造成的差異。",
        "- Cache 狀態、路由與排隊由各條件獨立演化，不假設 miss share 在方法或頻寬間相同。",
        "- 原生完整生命週期含寫回；TPOT 為每請求平均 token 間隔的算術平均；吞吐量從時間 0 至最後完整完成，包含到達空檔。",
        "- 指定 Llama-7B FP16 在此固定 H100 vLLM 資料包缺少 FP16 量測；缺少的 operator 使用原生 analytical fallback。每組的實際 match_type 計數保存在 results.json 與原生結果。",
        "- 使用既有 fork 的 cache-aware P 路由、D affinity 與完整寫回；P2P 使用可提供明確數值延遲的 MooncakeConnector。與提供 HTML 的原版問題情境有差異，預期百分比不作通過門檻。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = completed(args.run_root)
    comparisons = pair(records, args.output_dir)
    exported = [
        {key: value for key, value in row.items() if key != "rows"} for row in records
    ]
    document = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        completed=len(records),
        expected=10,
        complete=len(records) == 10,
        runs=exported,
        comparisons=comparisons,
        affected_d_endpoint="first_d_output_sampled_at_s minus original arrival",
    )
    (args.output_dir / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    (args.output_dir / "results.md").write_text(render(exported, comparisons))
    write_csv(args.output_dir / "comparison.csv", comparisons)
    flat = [
        {k: v for k, v in row.items() if k != "operator_match_type_counts"}
        for row in exported
    ]
    write_csv(args.output_dir / "native-metrics.csv", flat)
    if document["complete"]:
        from native_research.plot_bandwidth import draw

        draw(document, args.output_dir)
    print(
        f'Audited conditions: {len(records)}/10; report: {args.output_dir / "results.md"}'
    )


if __name__ == "__main__":
    main()
