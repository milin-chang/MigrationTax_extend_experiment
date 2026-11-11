"""Run a configured Mooncake-only link study using the existing native controller."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .aligned_audit import ROOT, guards, read
from .report_bandwidth import write_csv
from .report_five_stage_links import collect, paired_rows

METHODS = ("Serial", "PPD", "MigrationTax")
LINKS = (("nvl4", 450), ("pcie6", 128), ("ib400", 50))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def export(design, destination):
    destination.mkdir(parents=True, exist_ok=True)
    jobs = read(design / "jobs.json")
    records = collect(design)
    reference = next((row for row in records if row["reference"]), None)
    comparisons, index = [], []
    for record in records:
        rows = list(record["rows"].values())
        if not record["reference"] and reference is not None:
            if reference["input_sha256"] != record["input_sha256"]:
                raise ValueError("Finite and reference input hashes differ")
            rows = paired_rows(record["rows"], reference["rows"])
            returning = [row for row in rows if row["returning"]]
            n = len(returning)
            missing = sum(row["d_sample_missing"] for row in returning)
            affected = sum(row["affected_d"] is True for row in returning)
            comparisons.append(dict(
                link=record["link"], method=record["method"],
                bandwidth_gb_per_s=record["bandwidth_gb_per_s"],
                returning_requests=n, affected_d_count=affected,
                missing_d_samples=missing,
                affected_d_percent=100 * affected / n if not missing else None,
                affected_d_lower_percent=100 * affected / n,
                affected_d_upper_percent=100 * (affected + missing) / n,
                affected_client_percent=100 * sum(row["affected_client"] for row in returning) / n,
                decode_miss_percent=100 * sum(row["decode_miss_at_original_arrival"] for row in returning) / n,
            ))
        filename = record["id"] + "-requests.csv"
        write_csv(destination / filename, rows)
        index.append(dict(id=record["id"], csv=filename, rows=len(rows)))
    exported = [{key: value for key, value in row.items() if key != "rows"} for row in records]
    document = dict(created_utc=timestamp(), completed=len(records), expected=len(jobs),
                    complete=len(records) == len(jobs), runs=exported, comparisons=comparisons)
    save(destination / "results.json", document)
    save(destination / "csv-index.json", index)
    write_csv(destination / "native-metrics.csv", [
        {key: value for key, value in row.items() if key != "operator_match_type_counts"}
        for row in exported
    ])
    write_csv(destination / "comparison.csv", comparisons)
    hardware = read(design / "experiment.json")["hardware"]
    lines = [
        "# TokenSim：Mooncake 連線實驗", "",
        f'llama-7b／FP16、H100 SXM 80GB、TP{hardware["tensor_parallel"]}／PP1、4P＋4D，共 {hardware["total_gpus"]} 張模擬 GPU；seed 0。',
        "", f"已完成並通過稽核：{len(records)}／{len(jobs)} 組。", "",
        "## 階段與完成狀態", "",
        "| 階段 | 條件 | 已完成／規劃 |", "|---:|---|---:|",
    ]
    for phase, description in [(1, "Serial inf reference＋450 GB/s 三方法"),
                               (3, "128 GB/s 三方法"), (4, "50 GB/s 三方法")]:
        lines.append(f'| {phase} | {description} | {sum(r["phase"] == phase for r in records)}／{sum(j["phase"] == phase for j in jobs)} |')
    lines += ["", "## Affected share", "",
              "D-first 嚴格大於同請求 Serial inf 參考兩倍；分母為全部 returning turns。", "",
              "| 連線／GB/s | Serial % | PPD % | MigrationTax % |", "|---|---:|---:|---:|"]
    for link, bandwidth in LINKS:
        group = {row["method"]: row for row in comparisons if row["link"] == link}
        available = [row["affected_d_percent"] for row in group.values() if row["affected_d_percent"] is not None]
        best = min(round(value, 6) for value in available) if available else None
        values = []
        for method in METHODS:
            row = group.get(method)
            if row is None:
                values.append("pending")
            elif row["missing_d_samples"]:
                values.append(f'{row["affected_d_lower_percent"]:.6f}～{row["affected_d_upper_percent"]:.6f}（缺 {row["missing_d_samples"]}）')
            else:
                value = f'{row["affected_d_percent"]:.6f}'
                values.append(f"**{value}**" if round(row["affected_d_percent"], 6) == best else value)
        lines.append(f"| {link}／{bandwidth} | " + " | ".join(values) + " |")
    lines += ["", "## 請求與整批指標", "",
              "| 連線 | 方法 | Avg TTFT s | TTFT P50 s | TTFT P95 s | TTFT Max s | Avg lifecycle s | Avg TPOT s/token | Output tokens/s |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("mean_ttft_s", "p50_ttft_s", "p95_ttft_s", "max_ttft_s", "mean_lifecycle_s", "mean_tpot_s", "output_tokens_per_s")
    for row in records:
        peers = [r for r in records if r["link"] == row["link"] and not r["reference"]]
        values = []
        for key in keys:
            value = f"{row[key]:.6f}"
            if peers and not row["reference"]:
                best = (max if key == "output_tokens_per_s" else min)(round(r[key], 6) for r in peers)
                if round(row[key], 6) == best:
                    value = f"**{value}**"
            values.append(value)
        lines.append(f'| {row["link"]} | {row["method"]} | ' + " | ".join(values) + " |")
    lines += ["", "## 統計口徑", "",
              "- 同一份 Mooncake shifted v1 seed 0 全量 12,031 筆，其中 3,959 個 returning turns；輸出 4,122,048 tokens，不篩選、不截短、不重縮時間。",
              "- 本輪僅階段 1／3／4，共 10 組；沒有 ServeGen，也沒有 sanity 模擬。",
              "- Client TTFT 與實際 D 首次採樣分開記錄；cache miss 使用原始到達快照。",
              "- 生命週期包含寫回；吞吐量的時間窗包含到達空檔；TPOT 先逐請求計算，再取請求平均。",
              f'- 本輪 TP{hardware["tensor_parallel"]} 共 {hardware["total_gpus"]} 張 GPU；4P＋4D 固定時，各 TP 的 GPU 總數不同，屬不同 GPU 預算的部署對照。',
              "- HBM 固定 3,352 GB/s；改變 KV 通訊頻寬。TP1 無跨 rank 的 TP collective；TP2／4／8 的群組內 collective 沿用固定 NVLink 設定。",
              "- FP16 operator 計時沿用原生 analytical fallback；計時來源與執行耗時另見 results.json。",
              "- 同一連線內，延遲與 affected share 取最低、吞吐取最高者加粗；依六位小數比較，並列一併標示；reference 不參與排名。", ""]
    (destination / "results.md").write_text("\n".join(lines))
    return document


def execute(design, run, report, status):
    index = read(design / "execution/index.json")
    status.update(status="starting", execution_started_utc=timestamp(), total_jobs=index["total_jobs"], completed_jobs=0,
                  design=str(design), report=str(report), pid=os.getpid())
    save(run / "status.json", status)
    for batch in index["batches"]:
        plan = ROOT / batch["plan"]
        guards(read(plan))
        directory = run / "controllers" / batch["id"] / "controller"
        directory.parent.mkdir(parents=True, exist_ok=True)
        status.update(status="running", batch=batch["id"], phase=batch["phase"], updated_utc=timestamp())
        save(run / "status.json", status)
        with (directory.parent / "controller.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-m", "native_research.run_bandwidth",
                "--plan", str(plan), "--directory", str(directory)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["controller_pid"] = child.pid
            previous = -1
            while child.poll() is None:
                receipts = sum((ROOT / job["job"] / "audit.json").exists() for job in read(design / "jobs.json"))
                if receipts != previous:
                    result = export(design, report)
                    status["completed_jobs"] = result["completed"]
                    previous = receipts
                status["updated_utc"] = timestamp()
                save(run / "status.json", status)
                time.sleep(10)
        result = export(design, report)
        status["completed_jobs"] = result["completed"]
        if child.returncode:
            raise RuntimeError(f'Batch {batch["id"]} exited {child.returncode}; inspect preserved controller and job receipts')
    if status["completed_jobs"] != index["total_jobs"]:
        raise RuntimeError("Not every formal condition passed its final audit")
    status.update(status="completed", ended_utc=timestamp())
    save(run / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, help="Wait for this study to complete before starting any native jobs")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    design, run, report = args.design.resolve(), args.run_root.resolve(), args.report_dir.resolve()
    if args.report_only:
        print(json.dumps({k: v for k, v in export(design, report).items() if k not in ("runs", "comparisons")}))
        return
    run.mkdir(parents=True, exist_ok=True)
    if (run / "status.json").exists():
        parser.error("Refusing to overwrite an existing study")
    status = dict(started_utc=timestamp())
    try:
        if args.after_run:
            dependency = args.after_run.resolve()
            if dependency == run:
                raise ValueError("A study cannot depend on itself")
            started_wait = time.monotonic()
            status.update(status="queued_waiting_for_previous_study", total_jobs=len(read(design / "jobs.json")),
                          completed_jobs=0, pid=os.getpid(), design=str(design),
                          report=str(report), after_run=str(dependency))
            while True:
                path = dependency / "status.json"
                previous = read(path) if path.exists() else {}
                if previous.get("status") == "completed":
                    if previous.get("completed_jobs") != previous.get("total_jobs") or not previous.get("total_jobs"):
                        raise ValueError("Prerequisite completion counts are inconsistent")
                    break
                if previous.get("status") in {"failed", "stopped_pending", "stopped"}:
                    raise RuntimeError("Prerequisite study did not complete; inspect its preserved receipts")
                status.update(updated_utc=timestamp(), dependency_status=previous.get("status", "not_started"))
                save(run / "status.json", status)
                time.sleep(10)
            status["dependency_wait_seconds"] = time.monotonic() - started_wait
        execute(design, run, report, status)
    except Exception as error:
        status.update(status="failed", error=repr(error), ended_utc=timestamp())
        save(run / "status.json", status)
        raise


if __name__ == "__main__":
    main()
