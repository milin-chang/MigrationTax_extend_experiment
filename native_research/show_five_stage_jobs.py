"""Read-only status of the five-stage H100 link experiment."""

from datetime import datetime
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp8-s0"
RUN = ROOT / "native_research/runs/five-stage-llama7b-h100-tp8-s0-01"


def read(path):
    return json.loads(path.read_text())


def main():
    status = (
        read(RUN / "status.json")
        if (RUN / "status.json").exists()
        else dict(status="not_started", completed_jobs=0)
    )
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(
        f'{status["status"]}; completed {status.get("completed_jobs", 0)}/22; phase {status.get("phase", "-")}'
    )
    if status.get("batch"):
        print("Execution batch:", status["batch"])
    print(
        "PHASE STATE       TRACE     LINK    METHOD        COMPLETED     ELAPSED PEAK_GiB"
    )
    for job in read(DESIGN / "jobs.json"):
        output, receipt = ROOT / job["output"], ROOT / job["job"]
        state, count, elapsed, peak = "pending", 0, "-", "-"
        if (receipt / "running.json").exists():
            state = "running"
            elapsed = f'{(time.time() - (receipt / "running.json").stat().st_mtime) / 60:.1f}m'
        if (output / "progress.json").exists():
            count = read(output / "progress.json")["completed"]
        if (receipt / "job.json").exists():
            closed = read(receipt / "job.json")
            state = "closed"
            elapsed = f'{closed["wall_seconds"] / 60:.1f}m'
            peak = f'{closed["peak_aggregate_rss_bytes"] / 2**30:.2f}'
        if (receipt / "audit.json").exists():
            audit = read(receipt / "audit.json")
            state = "completed" if audit["status"] == "passed" else "failed"
            count = audit.get("requests", count)
        print(
            f'{job["phase"]:5} {state:11} {job["trace"]:9} {job["bandwidth_label"]:7} {job["method"]:13} {count:5}/{job["expected_requests"]:<6} {elapsed:7} {peak}'
        )
    if status.get("error"):
        print("Error:", status["error"])
    print(
        "Read-only, non-atomic snapshots; completed requires the final audit receipt."
    )


if __name__ == "__main__":
    main()
