"""Export only audited five-stage link-study runs and trace-specific pairs."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from .aligned_audit import ROOT, read
from .report_bandwidth import write_csv

METHODS = ("Serial", "PPD", "MigrationTax")
LINK_ORDER = ("nvl4", "pcie6", "ib400")


def paired_rows(records, reference):
    if records.keys() != reference.keys():
        raise ValueError("Source IDs differ from the trace-specific reference")
    result = []
    for source, row in records.items():
        baseline = reference[source]
        if any(
            row[key] != baseline[key]
            for key in ("arrival_s", "source_turn", "generated")
        ):
            raise ValueError("Paired source workload differs")
        returning = row["source_turn"] > 0
        sampled = row["d_output_sampled_latency_s"]
        reference_sample = baseline["d_output_sampled_latency_s"]
        if reference_sample is None:
            raise ValueError("Serial reference lacks a D sample")
        result.append(
            dict(
                **row,
                returning=returning,
                reference_d_first_s=baseline["d_output_sampled_latency_s"],
                reference_client_ttft_s=baseline["native_prefill_latency_s"],
                affected_d=(
                    (sampled > 2 * reference_sample)
                    if returning and sampled is not None
                    else None
                ),
                d_sample_missing=sampled is None,
                affected_client=(
                    (
                        row["native_prefill_latency_s"]
                        > 2 * baseline["native_prefill_latency_s"]
                    )
                    if returning
                    else None
                ),
            )
        )
    return result


def collect(design):
    completed = []
    for job in read(design / "jobs.json"):
        output, receipt = ROOT / job["output"], ROOT / job["job"]
        if (
            not (receipt / "audit.json").exists()
            or read(receipt / "audit.json")["status"] != "passed"
        ):
            continue
        summary, condition = read(output / "summary.json"), read(
            output / "condition.json"
        )
        if summary["status"] != "completed" or condition["status"] != "completed":
            raise ValueError("Audited job has inconsistent completion status")
        rows = {
            row["source_id"]: row
            for row in map(
                json.loads, (output / "requests.jsonl").read_text().splitlines()
            )
        }
        if (
            len(rows) != job["expected_requests"]
            or condition["input_sha256"] != job["input_sha256"]
        ):
            raise ValueError("Job input or output population differs")
        values = list(rows.values())
        ttft = [row["native_prefill_latency_s"] for row in values]
        d_first = [
            row["d_output_sampled_latency_s"]
            for row in values
            if row["d_output_sampled_latency_s"] is not None
        ]
        if any(not math.isfinite(value) for value in d_first):
            raise ValueError("Completed request has a nonfinite D sample")
        if any(
            row["d_output_sampled_latency_s"] is None
            and (row["generated"] > 2 or row["first_d_output_visible_at_s"] is None)
            for row in values
        ):
            raise ValueError("Unexplained missing D sample")
        native = read(next((output / "native").glob("result_*.json")))
        audit = read(receipt / "audit.json")
        completed.append(
            dict(
                id=job["id"],
                phase=job["phase"],
                trace=job["trace"],
                link=job["bandwidth_label"],
                method=job["method"],
                reference=job["reference"],
                input_sha256=job["input_sha256"],
                bandwidth_gb_per_s=job["bandwidth_bytes_per_s"] / 1e9,
                requests=len(values),
                returning=sum(row["source_turn"] > 0 for row in values),
                mean_ttft_s=statistics.mean(ttft),
                p50_ttft_s=statistics.median(ttft),
                p95_ttft_s=sorted(ttft)[math.ceil(0.95 * len(values)) - 1],
                p99_ttft_s=native["prefill_time"]["p99"],
                max_ttft_s=max(ttft),
                mean_d_first_observed_s=statistics.mean(d_first) if d_first else None,
                observed_d_sample_requests=len(d_first),
                missing_d_sample_requests=len(values) - len(d_first),
                mean_lifecycle_s=statistics.mean(
                    row["native_total_time_s"] for row in values
                ),
                mean_tpot_s=statistics.mean(
                    row["native_decode_time_sum_s"] / max(1, row["generated"] - 1)
                    for row in values
                ),
                output_tokens_per_s=sum(row["generated"] for row in values)
                / summary["simulated_end_s"],
                operator_match_type_counts=native["operator_match_type_counts"],
                wall_seconds=audit["wall_seconds"],
                peak_rss_gib=audit["peak_rss_bytes"] / 2**30,
                output=job["output"],
                rows=rows,
            )
        )
    return completed


def render(records, comparisons, destination):
    lines = [
        "# TokenSim：五階段連線實驗",
        "",
        "llama-7b／FP16、H100 SXM 80GB、TP8、4P＋4D，共 64 張模擬 GPU；seed 0。",
        "",
        f"已完成並通過稽核：{len(records)}／22 組。",
        "",
        "| 階段 | 已完成 | 規劃組數 |",
        "|---:|---:|---:|",
    ]
    for phase, expected in enumerate((2, 4, 4, 6, 6)):
        lines.append(
            f"| {phase} | {sum(row['phase'] == phase for row in records)} | {expected} |"
        )
    for trace in ("mooncake", "servegen"):
        lines += [
            "",
            f"## {trace}：Affected share",
            "",
            "D-first 嚴格大於同請求 Serial inf 參考兩倍；分母為 returning turns。",
            "",
            "| 連線／GB/s | Serial % | PPD % | MigrationTax % |",
            "|---|---:|---:|---:|",
        ]
        for link, bandwidth in (("nvl4", 450), ("pcie6", 128), ("ib400", 50)):
            group = {
                row["method"]: row
                for row in comparisons
                if row["trace"] == trace and row["link"] == link
            }
            values = []
            for method in METHODS:
                row = group.get(method)
                if row is None:
                    values.append("pending")
                elif row["missing_d_samples"]:
                    values.append(
                        f'{row["affected_d_lower_percent"]:.6f}～{row["affected_d_upper_percent"]:.6f}（缺 {row["missing_d_samples"]}）'
                    )
                else:
                    values.append(f'{row["affected_d_percent"]:.6f}')
            lines.append(f"| {link}／{bandwidth} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Decode miss share",
        "",
        "原始到達時，實際目的 D 的 inherited prefix 未完整命中，包含部分覆蓋。",
        "",
        "| Trace | 連線 | 方法 | Returning turns | Decode miss % |",
        "|---|---|---|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f'| {row["trace"]} | {row["link"]} | {row["method"]} | {row["returning_requests"]} | {row["decode_miss_percent"]:.6f} |'
        )
    lines += [
        "",
        "## 原生請求指標",
        "",
        "| 階段 | Trace | 連線 | 方法 | Avg TTFT s | P95 TTFT s | Avg lifecycle s | Avg TPOT s/token |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        values = [
            f"{row[key]:.6f}"
            for key in ("mean_ttft_s", "p95_ttft_s", "mean_lifecycle_s", "mean_tpot_s")
        ]
        lines.append(
            f'| {row["phase"]} | {row["trace"]} | {row["link"]} | {row["method"]} | '
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## 統計與模型口徑",
        "",
        "- Mooncake 為 shifted v1 全量 12,031 筆；ServeGen 為 primary v1 全量 39,698 筆，不排除對話；sanity 為 Mooncake 前 200 筆。",
        "- 各 Trace 共用一份全量 Serial inf 參考，依同一 source_id 配對；sanity 不混入正式統計。",
        "- D-first 為原始到達至第一個實際 D 採樣事件；client TTFT 另列。",
        "- Cache miss 使用原始到達快照，對應最後目的 D，不使用 admission 後的 d_admit_hit。",
        "- 連線服務使用峰值 bytes/s 與固定延遲；目前 Mooncake KV 路徑不套用 catalog 的 ramp 或 collective efficiency。",
        "- TP 群組內維持 nvlink4；GPU HBM 維持 3,352 GB/s。",
        "- 所有方法的 compute 實際 match type 另列 JSON；FP16 缺少量測時使用原生 analytical 模型，未完成實機校準。",
        "- 對照圖表預期百分比不作驗收門檻。",
        "",
        "[CSV 索引](csv-index.json)／[比較表](comparison.csv)／[完整數值](results.json)。",
        "",
    ]
    (destination / "results.md").write_text("\n".join(lines))


def export(design, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = collect(design)
    references = {row["trace"]: row for row in records if row["reference"]}
    comparisons, index = [], []
    for record in records:
        rows = list(record["rows"].values())
        if record["phase"] != 0 and not record["reference"]:
            baseline = references.get(record["trace"])
            if baseline is None:
                continue
            if baseline["input_sha256"] != record["input_sha256"]:
                raise ValueError("Finite and reference inputs differ")
            rows = paired_rows(record["rows"], baseline["rows"])
            returning = [row for row in rows if row["returning"]]
            n = len(returning)
            missing = sum(row["d_sample_missing"] for row in returning)
            affected = sum(row["affected_d"] is True for row in returning)
            comparisons.append(
                dict(
                    trace=record["trace"],
                    link=record["link"],
                    method=record["method"],
                    bandwidth_gb_per_s=record["bandwidth_gb_per_s"],
                    returning_requests=n,
                    affected_d_count=affected,
                    missing_d_samples=missing,
                    affected_d_percent=100 * affected / n if not missing else None,
                    affected_d_lower_percent=100 * affected / n,
                    affected_d_upper_percent=100 * (affected + missing) / n,
                    affected_client_percent=100
                    * sum(row["affected_client"] for row in returning)
                    / n,
                    decode_miss_percent=100
                    * sum(row["decode_miss_at_original_arrival"] for row in returning)
                    / n,
                    zero_d_prefix_percent=100
                    * sum(
                        row["inherited_prefix_tokens"] > 0
                        and row["d_local_tokens_at_original_arrival"] == 0
                        for row in returning
                    )
                    / n,
                    partial_d_prefix_percent=100
                    * sum(
                        0
                        < row["d_local_tokens_at_original_arrival"]
                        < row["inherited_prefix_tokens"]
                        for row in returning
                    )
                    / n,
                )
            )
        filename = record["id"] + "-requests.csv"
        write_csv(destination / filename, rows)
        index.append(
            dict(
                id=record["id"],
                phase=record["phase"],
                trace=record["trace"],
                csv=filename,
                rows=len(rows),
            )
        )
    comparisons.sort(
        key=lambda row: (
            row["trace"],
            LINK_ORDER.index(row["link"]),
            METHODS.index(row["method"]),
        )
    )
    exported = [
        {key: value for key, value in row.items() if key != "rows"} for row in records
    ]
    document = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        completed=len(records),
        expected=22,
        complete=len(records) == 22,
        runs=exported,
        comparisons=comparisons,
    )
    (destination / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    (destination / "csv-index.json").write_text(json.dumps(index, indent=2) + "\n")
    write_csv(destination / "comparison.csv", comparisons)
    if not comparisons:
        (destination / "comparison.csv").write_text(
            "trace,link,method,returning_requests,affected_d_percent\n"
        )
    write_csv(
        destination / "native-metrics.csv",
        [
            {k: v for k, v in row.items() if k != "operator_match_type_counts"}
            for row in exported
        ],
    )
    render(exported, comparisons, destination)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.design, args.report_dir)
    print(f'Audited five-stage runs: {result["completed"]}/22')


if __name__ == "__main__":
    main()
