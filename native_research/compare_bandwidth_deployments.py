"""Compare TP1 and TP8 on identical completed ServeGen source requests."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics

from .report_bandwidth import completed, write_csv


def waits(rows):
    values = list(rows.values())
    parent = [
        (
            max(0.0, rows[row["previous_source_id"]]["completed_at"] - row["arrival_s"])
            if row["previous_source_id"] is not None
            else 0.0
        )
        for row in values
    ]
    return dict(
        parent=statistics.mean(parent),
        admission=statistics.mean(row["native_budget_wait_s"] for row in values),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp1-run", type=Path, required=True)
    parser.add_argument("--tp8-run", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    old = {
        (row["bandwidth_label"], row["method"]): row for row in completed(args.tp1_run)
    }
    new = {
        (row["bandwidth_label"], row["method"]): row for row in completed(args.tp8_run)
    }
    if len(old) != 10 or old.keys() != new.keys():
        raise ValueError("Both deployments require all ten audited conditions")
    comparisons = []
    for key, baseline in old.items():
        current = new[key]
        if baseline["tensor_parallel"] != 1 or current["tensor_parallel"] != 8:
            raise ValueError("Expected TP1 and TP8 deployments")
        if baseline["input_sha256"] != current["input_sha256"]:
            raise ValueError("Input hashes differ")
        if any(
            baseline[field] != current[field]
            for field in ("model_id", "model_dtype", "block_tokens")
        ):
            raise ValueError("Model, precision or KV block geometry differs")
        if (
            baseline["requests"] != current["requests"]
            or baseline["rows"].keys() != current["rows"].keys()
        ):
            raise ValueError("Paired populations differ")
        for source, row in baseline["rows"].items():
            paired = current["rows"][source]
            if any(
                row[field] != paired[field]
                for field in (
                    "arrival_s",
                    "source_turn",
                    "generated",
                    "previous_source_id",
                )
            ):
                raise ValueError("Paired source workloads differ")
        bw, nw = waits(baseline["rows"]), waits(current["rows"])
        comparisons.append(
            dict(
                bandwidth=key[0],
                method=key[1],
                requests=current["requests"],
                input_sha256=current["input_sha256"],
                tp1_mean_ttft_s=baseline["mean_ttft_s"],
                tp8_mean_ttft_s=current["mean_ttft_s"],
                ttft_reduction_percent=100
                * (1 - current["mean_ttft_s"] / baseline["mean_ttft_s"]),
                tp1_p95_ttft_s=baseline["p95_ttft_s"],
                tp8_p95_ttft_s=current["p95_ttft_s"],
                tp1_mean_parent_wait_s=bw["parent"],
                tp8_mean_parent_wait_s=nw["parent"],
                tp1_mean_admission_wait_s=bw["admission"],
                tp8_mean_admission_wait_s=nw["admission"],
                tp1_after_admission_to_first_token_s=baseline["mean_ttft_s"]
                - bw["parent"]
                - bw["admission"],
                tp8_after_admission_to_first_token_s=current["mean_ttft_s"]
                - nw["parent"]
                - nw["admission"],
                tp1_mean_lifecycle_s=baseline["mean_lifecycle_s"],
                tp8_mean_lifecycle_s=current["mean_lifecycle_s"],
                tp1_mean_tpot_s=baseline["mean_tpot_s"],
                tp8_mean_tpot_s=current["mean_tpot_s"],
            )
        )
    comparisons.sort(
        key=lambda row: (
            row["bandwidth"] == "reference",
            (
                float("inf")
                if row["bandwidth"] == "reference"
                else float(row["bandwidth"].removeprefix("bw"))
            ),
            ("Serial", "PPD", "MigrationTax").index(row["method"]),
        )
    )
    destination = args.report_dir
    write_csv(destination / "tp1-vs-tp8.csv", comparisons)
    lines = [
        "# 相同 8,000 筆輸入：TP1 與 TP8 部署比較",
        "",
        "兩套部署使用同一份 ServeGen seed 0 輸入，模型、精度、KV block、batch 上限與 KV 傳輸頻寬條件相同。",
        "TP1 為 8 張模擬 GPU；TP8 為 64 張模擬 GPU，並使用固定 NVLink 處理群組內通訊。",
        "這是增加 GPU 資源後的部署比較，不能宣稱同 GPU 預算下的加速或效率提升。",
        "",
        "## TTFT",
        "",
        "| 頻寬 GB/s | 方法 | TP1 Avg TTFT s | TP8 Avg TTFT s | TTFT 降低 % | TP1 P95 s | TP8 P95 s |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        values = [
            f"{row[key]:.6f}"
            for key in (
                "tp1_mean_ttft_s",
                "tp8_mean_ttft_s",
                "ttft_reduction_percent",
                "tp1_p95_ttft_s",
                "tp8_p95_ttft_s",
            )
        ]
        lines.append(
            f'| {row["bandwidth"].removeprefix("bw")} | {row["method"]} | '
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## 等待拆分",
        "",
        "各欄為每請求平均秒數。",
        "",
        "| 頻寬 GB/s | 方法 | TP1 前輪等待 s | TP8 前輪等待 s | TP1 准入等待 s | TP8 准入等待 s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        values = [
            f"{row[key]:.6f}"
            for key in (
                "tp1_mean_parent_wait_s",
                "tp8_mean_parent_wait_s",
                "tp1_mean_admission_wait_s",
                "tp8_mean_admission_wait_s",
            )
        ]
        lines.append(
            f'| {row["bandwidth"].removeprefix("bw")} | {row["method"]} | '
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "TTFT 起點是原始到達；前輪等待與准入預算等待依每筆事件重新計算。",
        "通過准入至首 token 的剩餘時間仍包含原生排程、計算與傳輸，沒有被當成純 prefill。",
        "完整生命週期與 TPOT 的對照另列於 [CSV](tp1-vs-tp8.csv)。",
        "",
        "計算時間使用原生 analytical 模型，兩套結果都未經實機校準。",
        "",
    ]
    (destination / "TP1_VS_TP8.md").write_text("\n".join(lines))
    document = dict(
        comparisons=comparisons,
        same_input=True,
        equal_gpu_budget=False,
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (destination / "tp1-vs-tp8.json").write_text(json.dumps(document, indent=2) + "\n")
    print(destination / "TP1_VS_TP8.md")


if __name__ == "__main__":
    main()
