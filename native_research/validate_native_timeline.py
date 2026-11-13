"""Retain CLI baseline/observed runs and verify streaming P/D diagnostics."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from native_methods.inputs import ROOT
from native_research.verify_timeline_csv import verify_csv


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_requests(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def native_result(path):
    data = json.loads(next((path / "native").glob("result_*.json")).read_text())
    data.pop("simulator_wall_time")
    return data


def one_case(directory, tp, trace, method, count):
    design = ROOT / f"config/five-stage-llama7b-h100-sxm80gb-tp{tp}-s0"
    input_path = (
        ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp8-s0/inputs/mooncake-s0.jsonl"
        if trace == "mooncake"
        else ROOT / "inputs/servegen-s0-first8000.jsonl"
    )
    case = directory / f"tp{tp}" / trace / "nvl4" / method
    case.mkdir(parents=True)
    outputs = {}
    elapsed = {}
    for enabled in (False, True):
        name = "recorded" if enabled else "baseline"
        output = case / "seed-0" if enabled else case / "baseline/seed-0"
        command = [
            sys.executable,
            "-m",
            "native_methods.worker",
            "--backend",
            "tokensim",
            "--method",
            method,
            "--trace",
            trace,
            "--seed",
            "0",
            "--input",
            str(input_path),
            "--limit",
            str(count),
            "--hardware-profile",
            str(design / "hardware.json"),
            "--experiment-config",
            str(design / "study_nvl4.json"),
            "--output",
            str(output),
        ]
        if enabled:
            command.append("--record-timeline")
        started = time.monotonic()
        with (case / f"{name}-stdout.log").open("x") as stdout, (
            case / f"{name}-stderr.log"
        ).open("x") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                ),
            )
        elapsed[name] = time.monotonic() - started
        save(
            case / f"{name}-cli.json",
            dict(
                command=command,
                returncode=result.returncode,
                wall_seconds=elapsed[name],
                output=str(output),
            ),
        )
        if result.returncode:
            raise RuntimeError(
                f"{name} CLI failed; see {case / (name + '-stderr.log')}"
            )
        condition = json.loads((output / "condition.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        if (
            condition["status"] != "completed"
            or summary["status"] != "completed"
            or summary["completed"] != count
        ):
            raise ValueError("CLI workload did not complete")
        outputs[name] = output
    baseline, observed = outputs["baseline"], outputs["recorded"]
    baseline_rows, observed_rows = load_requests(
        baseline / "requests.jsonl"
    ), load_requests(observed / "requests.jsonl")
    by_id = lambda rows: {r["source_id"]: r for r in rows}
    if by_id(baseline_rows) != by_id(observed_rows):
        raise ValueError("Per-request records changed when timeline was enabled")
    if native_result(baseline) != native_result(observed):
        raise ValueError("Native metrics changed beyond simulator wall-clock time")
    if json.loads((baseline / "summary.json").read_text()) != json.loads(
        (observed / "summary.json").read_text()
    ):
        raise ValueError("Native execution counters or completion changed")
    checks = verify_csv(observed, tp)
    return dict(
        tp=tp,
        trace=trace,
        method=method,
        status="passed",
        requests=count,
        returning=sum(r["source_turn"] > 0 for r in observed_rows),
        baseline=str(baseline.relative_to(ROOT)),
        observed=str(observed.relative_to(ROOT)),
        wall_seconds=elapsed,
        native_metrics_equal=True,
        request_records_equal=True,
        request_sha256={
            key: sha256((path / "requests.jsonl").read_bytes()).hexdigest()
            for key, path in outputs.items()
        },
        csv_checks=checks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "native_research/runs/native-timeline-validation-s0-128-01",
    )
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument(
        "--tp", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 4, 8]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("Serial", "PPD", "MigrationTax"),
        default=["Serial", "PPD", "MigrationTax"],
    )
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", started_utc=datetime.now(timezone.utc).isoformat(), cases=[]
    )
    save(directory / "validation.json", report)
    try:
        for tp in args.tp:
            for trace in ("mooncake", "servegen"):
                for method in args.methods:
                    report["current"] = dict(tp=tp, trace=trace, method=method)
                    save(directory / "validation.json", report)
                    result = one_case(directory, tp, trace, method, args.requests)
                    report["cases"].append(result)
                    save(directory / "validation.json", report)
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in ("tp", "trace", "method", "status", "requests")
                            }
                        ),
                        flush=True,
                    )
        report.update(
            status="passed", completed_utc=datetime.now(timezone.utc).isoformat()
        )
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save(directory / "validation.json", report)
    lines = [
        "# P／D 時間線 CLI 驗證",
        "",
        f"狀態：{report['status']}。",
        "",
        "每組以相同輸入與 seed 分別關閉／開啟紀錄，逐筆比對 requests.jsonl、完成計數及原生結果 JSON（僅排除 simulator_wall_time）。",
        "",
        "本測試使用小批資料驗證功能，不取代正式全集實驗。",
        "",
        "| TP | Trace | 方法 | 請求數 | Returning | 檢查 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in report["cases"]:
        lines.append(
            f'| {row["tp"]} | {row["trace"]} | {row["method"]} | {row["requests"]} | {row["returning"]} | {row["status"]} |'
        )
    lines += [
        "",
        "每組的 seed-0/ 內保存 request-timeline.csv、worker-steps.csv、kv-transfers.csv、timeline-schema.json 及原始收據與來源快照。",
        "",
    ]
    (directory / "VALIDATION.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                directory=str(directory),
                status=report["status"],
                pairs=len(report["cases"]),
            )
        )
    )


if __name__ == "__main__":
    main()
