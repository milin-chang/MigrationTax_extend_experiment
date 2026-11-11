"""Run five frozen phases after a prerequisite matrix and export each delivery."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .aligned_audit import ROOT, audit, guards, read
from .report_five_stage_links import collect, export


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def controller_directory(run, batch):
    return run / "controllers" / batch / "controller"


def sanity(design, run):
    rows = [row for row in collect(design) if row["phase"] == 0]
    by_link = {row["link"]: row for row in rows}
    if set(by_link) != {"nvl4", "ib400"}:
        raise RuntimeError("Sanity phase lacks its two audited populations")
    fast, slow = by_link["nvl4"], by_link["ib400"]
    if (
        fast["input_sha256"] != slow["input_sha256"]
        or fast["rows"].keys() != slow["rows"].keys()
    ):
        raise RuntimeError("Sanity inputs differ")
    deltas = [
        slow["rows"][source]["d_output_sampled_latency_s"]
        - row["d_output_sampled_latency_s"]
        for source, row in fast["rows"].items()
    ]
    result = dict(
        status=(
            "passed"
            if slow["mean_d_first_observed_s"] > fast["mean_d_first_observed_s"]
            else "failed"
        ),
        requests=len(deltas),
        fast_mean_d_first_s=fast["mean_d_first_observed_s"],
        slow_mean_d_first_s=slow["mean_d_first_observed_s"],
        paired_slow_later_count=sum(delta > 1e-12 for delta in deltas),
        mean_paired_delta_s=sum(deltas) / len(deltas),
        limitation="Queue and cache evolution can make individual pairs nonmonotone; no historical affected-share targets are asserted",
    )
    save(run / "sanity.json", result)
    if result["status"] != "passed":
        raise RuntimeError(
            "The low-bandwidth sanity mean did not increase; retaining diagnostics and stopping new phases"
        )


def execute(args, status):
    design, run, report = (
        args.design.resolve(),
        args.run_root.resolve(),
        args.report_dir.resolve(),
    )
    index = read(design / "execution/index.json")
    status.update(
        status="queued_waiting_for_previous_matrix",
        total_jobs=index["total_jobs"],
        completed_jobs=len(collect(design)) if args.resume else 0,
        design=str(design),
    )
    save(run / "status.json", status)
    while args.after_run is not None:
        path = args.after_run / "postprocess.json"
        previous = read(path) if path.exists() else {}
        if previous.get("status") == "completed":
            break
        if previous.get("status") == "failed":
            raise RuntimeError("Prerequisite matrix postprocessing failed")
        time.sleep(10)
    available = os.sched_getaffinity(0)
    for phase in index["phases"]:
        number = phase["phase"]
        plan_path = ROOT / phase["plan"]
        plan = read(plan_path)
        guards(plan)
        batch = phase.get("batch", f"phase-{number}")
        directory = controller_directory(run, batch)
        legacy = directory.parent / "runner.json"
        closed = directory / "runner.json"
        previous = closed if closed.exists() else legacy
        if (
            args.resume
            and previous.exists()
            and read(previous)["status"] == "completed"
        ):
            for job in plan["jobs"]:
                result = audit(plan, job)
                if result["status"] != "passed":
                    raise RuntimeError(result)
            document = export(design, report)
            status["completed_jobs"] = document["completed"]
            status["completed_batches"] = status.get("completed_batches", 0) + 1
            if number == 0:
                sanity(design, run)
            save(run / "status.json", status)
            continue
        directory.parent.mkdir(parents=True, exist_ok=True)
        status.update(
            status="running", phase=number, batch=batch, phase_jobs=phase["count"]
        )
        save(run / "status.json", status)
        with (directory.parent / "controller.log").open("x") as log:
            os.sched_setaffinity(0, available)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "native_research.run_bandwidth",
                    "--plan",
                    str(plan_path),
                    "--directory",
                    str(directory),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            os.sched_setaffinity(0, [max(available)])
            last_count = None
            while process.poll() is None:
                path = directory / "runner.json"
                if path.exists():
                    journal = read(path)
                    count = sum(job["status"] == "completed" for job in journal["jobs"])
                    if count != last_count:
                        document = export(design, report)
                        status["completed_jobs"] = document["completed"]
                        last_count = count
                    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
                    save(run / "status.json", status)
                time.sleep(5)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Phase {number} controller exited {process.returncode}"
                )
        for job in plan["jobs"]:
            result = audit(plan, job)
            if result["status"] != "passed":
                raise RuntimeError(result)
        document = export(design, report)
        status["completed_jobs"] = document["completed"]
        if number == 0:
            sanity(design, run)
        if batch == "03-mooncake-ib400" or number == 2:
            mooncake_delivery = batch == "03-mooncake-ib400"
            destination = report / (
                "delivery-mooncake-complete"
                if mooncake_delivery
                else "delivery-after-phase-2"
            )
            destination.mkdir()
            for path in report.iterdir():
                if path.is_file() and path.suffix in (".csv", ".json", ".md"):
                    shutil.copy2(path, destination / path.name)
            save(
                run
                / (
                    "first-delivery.json"
                    if mooncake_delivery
                    else "phase2-delivery.json"
                ),
                dict(
                    status="available",
                    completed_jobs=document["completed"],
                    report=str(destination / "results.md"),
                    csv_index=str(destination / "csv-index.json"),
                    scope=(
                        "All Mooncake links"
                        if mooncake_delivery
                        else "Mooncake complete plus ServeGen fast-link results"
                    ),
                ),
            )
        status["completed_batches"] = status.get("completed_batches", 0) + 1
        status["completed_logical_phases"] = [
            p
            for p, expected in enumerate((2, 4, 4, 6, 6))
            if sum(row["phase"] == p for row in document["runs"]) == expected
        ]
        status["completed_phases"] = len(status["completed_logical_phases"])
        save(run / "status.json", status)
    from .plot_five_stage_links import draw

    draw(document, report)
    status.update(
        status="completed",
        completed_jobs=document["completed"],
        ended_utc=datetime.now(timezone.utc).isoformat(),
        report=str(report / "results.md"),
    )
    save(run / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--after-run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if (args.run_root / "status.json").exists() and not args.resume:
        parser.error("Refusing to overwrite an existing five-stage run")
    if (
        args.resume
        and (args.run_root / "status.json").exists()
        and read(args.run_root / "status.json")["status"] != "failed"
    ):
        parser.error(
            "Resume requires a failed coordinator; do not duplicate active jobs"
        )
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    try:
        execute(args, status)
    except Exception as error:
        status.update(
            status="failed",
            error=repr(error),
            ended_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(args.run_root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
