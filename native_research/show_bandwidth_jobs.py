"""Read-only status for the ServeGen communication-bandwidth matrix."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "native_research/runs/servegen7b-bw-s0-8000-01"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "native_research/plans/servegen7b-bandwidth-s0-8000.json",
    )
    args = parser.parse_args()
    plan = read(args.plan)
    run = (ROOT / plan["jobs"][0]["output"]).parents[2]
    journal = (
        read(run / "controller/runner.json")
        if (run / "controller/runner.json").exists()
        else dict(status="not_started", jobs=[])
    )
    states = {job["id"]: job for job in journal["jobs"]}
    print(
        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), journal["status"]
    )
    print(
        "STATE       GB/s       METHOD        COMPLETED       CPU  ELAPSED     PEAK_GiB"
    )
    for job in plan["jobs"]:
        state = states.get(job["id"], {}).get("status", "pending")
        receipt = ROOT / job["job"]
        count, elapsed, peak, cpu = 0, "-", "-", "-"
        entry = states.get(job["id"], {})
        if entry.get("cpu_ids"):
            cpu = str(entry["cpu_ids"][0])
        if (receipt / "running.json").exists():
            elapsed = (
                f'{(time.time() - (receipt / "running.json").stat().st_mtime)/60:.1f}m'
            )
        if entry.get("sampled_peak_rss_bytes"):
            peak = f'{entry["sampled_peak_rss_bytes"]/2**30:.2f}'
        progress = ROOT / job["output"] / "progress.json"
        if progress.exists():
            count = read(progress)["completed"]
        if (receipt / "job.json").exists():
            closed = read(receipt / "job.json")
            elapsed = f'{closed["wall_seconds"]/60:.1f}m'
            peak = f'{closed["peak_aggregate_rss_bytes"]/2**30:.2f}'
        if state == "completed":
            count = job["expected_requests"]
        label = (
            "reference"
            if job["reference"]
            else str(int(job["bandwidth_bytes_per_s"] / 1e9))
        )
        print(
            f'{state:11} {label:10} {job["method"]:13} {count:5}/{job["expected_requests"]:<8} {cpu:4} {elapsed:11} {peak}'
        )


if __name__ == "__main__":
    main()
