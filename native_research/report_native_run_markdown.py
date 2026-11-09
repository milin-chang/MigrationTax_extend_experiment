"""Recompute completed native-run metrics and write Markdown, JSON and CSV.

No simulator is started. Standard-library-only CLI; also works on a restored
Release archive with explicit --run-root and --output-dir arguments.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import statistics

METHODS = ("Serial", "PPD", "MigrationTax")
METRICS = (
    ("平均 TTFT ↓", "秒", "mean_ttft_s", False),
    ("TTFT 中位數 ↓", "秒", "median_ttft_s", False),
    ("TTFT P95 ↓", "秒", "p95_ttft_s", False),
    ("平均完整生命週期，含寫回 ↓", "秒", "mean_lifecycle_s", False),
    ("平均 TPOT ↓", "ms/token", "mean_tpot_ms", False),
    ("整批輸出吞吐量 ↑", "tokens/s", "output_tokens_per_s", True),
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    require(bool(values), "Empty metric population")
    require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid elapsed time")
    ordered = sorted(values)
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95_nearest_rank=ordered[math.ceil(0.95 * len(values)) - 1],
        max=max(values),
    )


def splitwise_tpot(output, requests):
    """Use output-emitting native events; exclude subsequent KV writeback."""
    emitted, first, last = Counter(), {}, {}
    path = output / "native-nodes.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            count = event["source_outputs_emitted"]
            if not count:
                continue
            source = event["source_id"]
            require(source in requests and count > 0, "Unknown output source")
            at = event["completed_at_s"]
            emitted[source] += count
            first[source] = min(first.get(source, at), at)
            last[source] = max(last.get(source, at), at)
    values = []
    for source, row in requests.items():
        require(emitted[source] == row["generated"], "Output emission count mismatch")
        require(
            math.isclose(first[source], row["first_at_s"], abs_tol=1e-8, rel_tol=0),
            "First output timestamp mismatch",
        )
        require(
            last[source] <= row["completed_at_s"] + 1e-8,
            "Output after writeback completion",
        )
        if row["generated"] == 1:
            require(
                first[source] == last[source], "Single output has multiple timestamps"
            )
            values.append(0.0)
        else:
            values.append((last[source] - first[source]) / (row["generated"] - 1))
    return values, dict(
        path=path.name,
        sha256=digest(path),
        definition="(last emitting event - first emitting event)/(outputs - 1); singleton 0",
    )


def collect(output, job, hardware):
    condition, summary, config = [
        read(output / name)
        for name in ("condition.json", "summary.json", "effective-config.json")
    ]
    receipts = output.with_name(output.name + "-job")
    receipt, audit = read(receipts / "job.json"), read(receipts / "audit.json")
    require(
        condition["status"] == summary["status"] == "completed", "Incomplete condition"
    )
    require(condition["limit"] is None, "Limited run cannot enter full-run summary")
    require(
        receipt["status"] == audit["status"] == "passed" and receipt["returncode"] == 0,
        "Missing successful receipts",
    )
    require(
        not receipt["timed_out"] and not audit["failures"], "Failed audit or timeout"
    )
    require(
        not summary["rejected"] and not summary["unfinished"],
        "Rejected or unfinished sources",
    )
    require(
        config["hardware_alignment"] == hardware, "Hardware differs from frozen plan"
    )
    require(condition["input_sha256"] == job["input_sha256"], "Input hash mismatch")
    for key in ("backend", "method", "seed"):
        require(condition[key] == job[key], "Condition metadata mismatch: " + key)
    requests = {}
    with (output / "requests.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["source_id"] not in requests, "Duplicate source ID")
            require(row["status"] == "completed", "Unfinished request")
            requests[row["source_id"]] = row
    count = job["expected_requests"]
    require(
        set(requests) == {str(i) for i in range(count)}, "Missing or unknown sources"
    )
    require(
        summary["completed"] == summary["input_requests"] == count,
        "Population mismatch",
    )
    tokens = sum(row["generated"] for row in requests.values())
    require(tokens == job["expected_output_tokens"], "Output token count mismatch")
    backend = job["backend"]
    if backend == "tokensim":
        ttft = [r["native_prefill_latency_s"] for r in requests.values()]
        lifecycle = [r["native_total_time_s"] for r in requests.values()]
        tpot = [
            r["native_decode_time_sum_s"] / max(1, r["generated"] - 1)
            for r in requests.values()
        ]
        for row in requests.values():
            if row["generated"] == 1:
                require(
                    row["native_decode_time_sum_s"] == 0,
                    "Singleton has decode intervals",
                )
            require(
                math.isclose(
                    row["completed_at"] - row["arrival_s"],
                    row["native_total_time_s"],
                    rel_tol=0,
                    abs_tol=1e-8,
                ),
                "Lifecycle timestamp mismatch",
            )
        native_paths = list((output / "native").glob("result_*.json"))
        require(len(native_paths) == 1, "Expected one native TokenSim result")
        native = read(native_paths[0])
        require(
            native["termination_reason"] == "completed"
            and native["request_count"] == count,
            "Native exporter incomplete",
        )
        for key, value in (("p50", statistics.median(tpot)), ("max", max(tpot))):
            require(
                math.isclose(
                    value, native["decode_time"][key], rel_tol=0, abs_tol=1e-9
                ),
                "TPOT differs from native distribution",
            )
        tpot_source = dict(
            path="requests.jsonl",
            definition="native_decode_time_sum_s / max(1, generated - 1); singleton 0",
            native_result=str(native_paths[0].relative_to(output)),
            native_sha256=digest(native_paths[0]),
        )
    elif backend == "splitwise":
        ttft = [r["derived_ttft_s"] for r in requests.values()]
        lifecycle = [r["derived_latency_s"] for r in requests.values()]
        for row in requests.values():
            require(
                row["derived_ttft_s"] == row["first_at_s"] - row["arrival_s"],
                "TTFT timestamp mismatch",
            )
            require(
                row["derived_latency_s"] == row["completed_at_s"] - row["arrival_s"],
                "Lifecycle timestamp mismatch",
            )
        tpot, tpot_source = splitwise_tpot(output, requests)
    else:
        raise ValueError("Unsupported native backend: " + backend)
    require(
        all(first <= full + 1e-8 for first, full in zip(ttft, lifecycle)),
        "TTFT exceeds full lifecycle",
    )
    distributions = dict(
        ttft_s=describe(ttft), lifecycle_s=describe(lifecycle), tpot_s=describe(tpot)
    )
    duration = summary["simulated_end_s"]
    require(math.isfinite(duration) and duration > 0, "Invalid observation window")
    metrics = dict(
        mean_ttft_s=distributions["ttft_s"]["mean"],
        median_ttft_s=distributions["ttft_s"]["median"],
        p95_ttft_s=distributions["ttft_s"]["p95_nearest_rank"],
        mean_lifecycle_s=distributions["lifecycle_s"]["mean"],
        mean_tpot_ms=distributions["tpot_s"]["mean"] * 1000,
        output_tokens_per_s=tokens / duration,
    )
    return dict(
        backend=backend,
        trace=job["trace"],
        method=job["method"],
        seed=job["seed"],
        requests=count,
        output_tokens=tokens,
        single_output_requests=sum(r["generated"] == 1 for r in requests.values()),
        input_sha256=job["input_sha256"],
        observation_start_s=0,
        observation_end_s=duration,
        metrics=metrics,
        distributions=distributions,
        tpot_source=tpot_source,
        source_directory=str(output),
        requests_sha256=digest(output / "requests.jsonl"),
        condition_sha256=digest(output / "condition.json"),
        receipt_sha256=digest(receipts / "job.json"),
    )


def render(rows, hardware, command):
    text = [
        "# TP8 seed 0 模擬結果",
        "",
        "模型／GPU：Llama2-70B、FP16 權重與 KV、H100、TP8／PP1、4P＋4D，共 64 GPU。",
        "Mooncake 使用 shifted v1；ServeGen 使用 primary v1。以下均為 seed 0，12 組完整執行。",
        "",
        "粗體表示同一模擬器、同一 Trace、同列的最佳值；並列最佳全部標示。依未四捨五入原值比較。",
        "",
    ]
    for backend, title in (("splitwise", "SplitwiseSim"), ("tokensim", "TokenSim")):
        for trace, trace_title in (("mooncake", "Mooncake"), ("servegen", "ServeGen")):
            group = {
                r["method"]: r
                for r in rows
                if r["backend"] == backend and r["trace"] == trace
            }
            require(set(group) == set(METHODS), "Incomplete method group")
            require(
                len({r["input_sha256"] for r in group.values()}) == 1,
                "Methods have different inputs",
            )
            text += [
                f"### {title}／{trace_title}",
                "",
                "| 指標 | 單位 | Serial | PPD | MigrationTax |",
                "|---|---|---:|---:|---:|",
            ]
            for label, unit, key, larger in METRICS:
                best = (max if larger else min)(
                    r["metrics"][key] for r in group.values()
                )
                values = []
                for method in METHODS:
                    value = group[method]["metrics"][key]
                    formatted = f"{value:,.6f}"
                    values.append(
                        "**" + formatted + "**" if value == best else formatted
                    )
                text.append(
                    "| " + label + " | " + unit + " | " + " | ".join(values) + " |"
                )
            text += [
                "",
                f"每方法完成 {group['Serial']['requests']:,} 筆請求，輸出 {group['Serial']['output_tokens']:,} tokens。",
                "",
            ]
    text += [
        "## 統計口徑",
        "",
        "- TTFT 從 Trace 原始到達計算至首 token，包含前輪相依與准入等待；納入初輪與 returning 請求。",
        "- TTFT P95 採 nearest rank：排序後第 ceil(0.95 × N) 筆。",
        "- 完整生命週期從原始到達計算至最後寫回完成。",
        "- TPOT 先計算每請求首 token 後的平均 token 間隔，再對所有完成請求取算術平均；單輸出請求記為 0，不按 token 數加權。",
        "- TokenSim TPOT 使用保存的 native decode interval sum；SplitwiseSim 使用原生輸出事件的最後／首 token 時間差，排除後續寫回。JSON 保存來源、計算公式與統計請求數。",
        "- 整批輸出吞吐量＝輸出 tokens 總數／從時間 0 至最後寫回完成的模擬時間，包含到達空檔。微小差異不代表最大可持續吞吐量提升。",
        "- 僅 seed 0，不是四個 seeds 的中位數；先前不同硬體或輸入版本的表格不混入本次結果。",
        "- 完成與資料完整性核對不代表實機計時校準；兩套模擬器仍有不同計時、排程與准入假設。",
        "",
        "## 重新產生 Markdown",
        "",
        "以下 CLI 只重新讀取已完成結果，不啟動模擬。",
        "",
        "```bash",
        command,
        "```",
        "",
        "輸出：`results.md`、`results.json`、`results.csv`。JSON 保存完整精度與來源 SHA-256。",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("native_research/runs/tp8-aligned-s0-shifted-v1-01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("native_research/reports/tp8-aligned-s0-shifted-v1-01"),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print output paths without printing the Markdown",
    )
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
    run = args.run_root.resolve()
    destination = args.output_dir.resolve()
    require(
        run != destination and run not in destination.parents,
        "Report directory must be outside immutable run folder",
    )
    plan = read(run / "controller/plan.snapshot.json")
    runner = read(run / "controller/runner.json")
    require(runner["status"] == "completed", "Campaign is not complete")
    require(
        len(plan["jobs"]) == 12 and {j["seed"] for j in plan["jobs"]} == {0},
        "This report requires the completed 12-condition seed-0 matrix",
    )
    rows = []
    for job in plan["jobs"]:
        output = (
            run / job["backend"] / job["trace"] / job["method"] / f'seed-{job["seed"]}'
        )
        rows.append(collect(output, job, plan["hardware_alignment"]))
        print(
            f'Checked {job["backend"]}/{job["trace"]}/{job["method"]}/seed-0',
            flush=True,
        )
    for trace in ("mooncake", "servegen"):
        require(
            len({r["input_sha256"] for r in rows if r["trace"] == trace}) == 1,
            "Cross-backend input mismatch",
        )
    rows.sort(
        key=lambda r: (
            ("splitwise", "tokensim").index(r["backend"]),
            ("mooncake", "servegen").index(r["trace"]),
            METHODS.index(r["method"]),
        )
    )
    command = shlex.join(
        [
            "python3",
            "native_research/report_native_run_markdown.py",
            "--run-root",
            str(args.run_root),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    markdown = render(rows, plan["hardware_alignment"], command)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.md").write_text(markdown)
    document = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Twelve completed TP8 seed-0 native conditions",
        hardware_alignment=plan["hardware_alignment"],
        run_root=str(run),
        plan_sha256=digest(run / "controller/plan.snapshot.json"),
        generator_sha256=digest(Path(__file__)),
        tpot_population="All completed source requests; single-output requests contribute zero; arithmetic mean of per-request means",
        rows=rows,
    )
    (destination / "results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    )
    flat = [
        {
            **{
                k: r[k]
                for k in (
                    "backend",
                    "trace",
                    "method",
                    "seed",
                    "requests",
                    "output_tokens",
                    "single_output_requests",
                )
            },
            **r["metrics"],
        }
        for r in rows
    ]
    with (destination / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if not args.quiet:
        print(markdown)
    print(f'Saved {destination/"results.md"}', flush=True)


if __name__ == "__main__":
    main()
