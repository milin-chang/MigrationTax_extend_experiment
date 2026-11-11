"""Run the frozen matrix, refresh partial reports and finalize after closure."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "native_research/runs/servegen7b-bw-s0-8000-01"
PLAN = ROOT / "native_research/plans/servegen7b-bandwidth-s0-8000.json"


def save(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def report(log, run, destination):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "native_research.report_bandwidth",
            "--run-root",
            str(run),
            "--output-dir",
            str(destination),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "native_research/reports/servegen7b-bw-s0-8000-01",
    )
    args = parser.parse_args()
    run, plan = args.run_root.resolve(), args.plan.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "orchestrator.json").exists():
        parser.error("Refusing to overwrite an existing run")
    status = dict(
        status="starting",
        started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        plan=str(plan),
        automatic_report=True,
    )
    status_path = run / "orchestrator.json"
    save(status_path, status)
    with (run / "controller.log").open("x") as log, (run / "report.log").open(
        "a"
    ) as reports:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "native_research.run_bandwidth",
                "--plan",
                str(plan),
                "--directory",
                str(run / "controller"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # The controller inherited the full CPU mask and selects CPUs 0..11.
        # Report generation uses the spare CPU without changing worker affinity.
        os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
        previous = None
        while process.poll() is None:
            journal = run / "controller/runner.json"
            if journal.exists():
                try:
                    data = json.loads(journal.read_text())
                    count = sum(job["status"] == "completed" for job in data["jobs"])
                    if count != previous:
                        report(reports, run, args.report_dir)
                        reports.flush()
                        previous = count
                    status.update(
                        status="running",
                        controller_pid=process.pid,
                        completed=count,
                        updated_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    save(status_path, status)
                except (OSError, ValueError):
                    pass
            time.sleep(5)
        code = report(reports, run, args.report_dir)
        final_journal = json.loads((run / "controller/runner.json").read_text())
        count = sum(job["status"] == "completed" for job in final_journal["jobs"])
        status.update(
            status="completed" if process.returncode == code == 0 else "failed",
            completed=count,
            controller_returncode=process.returncode,
            report_returncode=code,
            ended_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(status_path, status)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
