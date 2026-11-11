"""Run a frozen native matrix with one CPU per job and memory-admitted concurrency.

No simulation timing lives here. Receipts stay separate from worker outputs.
A correctness/resource failure stops new admissions while existing jobs finish.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from .aligned_audit import ROOT, audit, digest, guards, read
from .show_jobs import tail

GIB = 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def estimate(job):
    """Conservative admission estimates, not physical memory limits or measurements."""
    if "memory_estimate_bytes" in job:
        value = job["memory_estimate_bytes"]
        if type(value) is not int or not 0 < value <= 8 * GIB:
            raise ValueError("Job memory estimate must be positive and at most 8 GiB")
        return value
    if job.get("limit"):
        return GIB
    if job["backend"] == "splitwise":
        return GIB
    if job["trace"] == "mooncake":
        return 2 * GIB
    return int((3.5 if job["method"] == "MigrationTax" else 3) * GIB)


def priority(job):
    return (
        not job.get("reference", False),
        ("Serial", "PPD", "MigrationTax").index(job["method"]),
        job["bandwidth_bytes_per_s"],
    )


def admission(available, minimum, running, candidate):
    # Available RAM already excludes live RSS: reserve only the estimated remaining growth.
    growth = sum(
        max(0, max(r["estimate"], int(r["peak"] * 1.35) + GIB // 4) - r["rss"])
        for r in running
    )
    needed = minimum + growth + estimate(candidate) + GIB // 4
    return dict(
        allowed=available >= needed,
        host_available_bytes=available,
        growth_reserved_bytes=growth,
        next_job_estimate_bytes=estimate(candidate),
        minimum_host_available_bytes=minimum,
        sampling_margin_bytes=GIB // 4,
    )


def free_cpu(cpus, active):
    busy = {r["cpu"] for r in active}
    for cpu in cpus:
        if cpu in busy:
            continue
        with Path(f"/tmp/migrationtax-native-cpu-{cpu}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
        return cpu
    return None


def tree_rss(process):
    result = 0
    try:
        root = psutil.Process(process.pid)
        for child in [root, *root.children(recursive=True)]:
            try:
                result += child.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return result


def start(job, cpu, directory, decision):
    command = [
        sys.executable,
        "-m",
        "native_research.named_job",
        "--output",
        job["job"],
        "--name",
        "bw7b-" + job["id"],
        "--cpus",
        str(cpu),
        "--",
        *job["worker_argv"],
    ]
    log = (directory / (job["id"] + ".launcher.log")).open("x")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    print(
        json.dumps(dict(id=job["id"], status="started", cpu=cpu, pid=process.pid)),
        flush=True,
    )
    return dict(
        job=job,
        cpu=cpu,
        process=process,
        estimate=estimate(job),
        rss=0,
        peak=0,
        decision=decision,
    )


def execute(plan, directory, journal, cpus):
    pending = sorted(plan["jobs"], key=priority)
    active = []
    entries = {e["id"]: e for e in journal["jobs"]}
    minimum = plan["resources"]["minimum_host_available_bytes"]
    stopping = False

    def stop_admissions(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_admissions)
    signal.signal(signal.SIGINT, stop_admissions)
    last_guard = 0
    while pending or active:
        for item in list(active):
            job, process = item["job"], item["process"]
            entry = entries[job["id"]]
            if process.poll() is None:
                item["rss"] = tree_rss(process)
                item["peak"] = max(item["peak"], item["rss"])
                entry.update(rss_bytes=item["rss"], sampled_peak_rss_bytes=item["peak"])
                continue
            active.remove(item)
            try:
                result = audit(plan, job)
            except Exception as error:
                result = dict(status="failed", failures=[repr(error)])
            if process.returncode or entry.get("resource_failure"):
                result["status"] = "failed"
                result.setdefault("failures", []).append(
                    entry.get("resource_failure")
                    or f"launcher exit {process.returncode}"
                )
            receipt_dir = ROOT / job["job"]
            if receipt_dir.exists():
                write(receipt_dir / "audit.json", result)
            entry.update(
                status="completed" if result["status"] == "passed" else "failed",
                ended_utc=now(),
                audit=result,
                launcher_returncode=process.returncode,
            )
            print(
                json.dumps(dict(id=job["id"], status=entry["status"], audit=result)),
                flush=True,
            )
            if result["status"] != "passed":
                stopping = True

        available = psutil.virtual_memory().available
        if active and available < minimum + GIB // 4:
            # Abort a job before exhausting the host reserve; preserve its partial artifacts.
            stopping = True
            victim = max(active, key=lambda r: r["rss"])
            entry = entries[victim["job"]["id"]]
            if not entry.get("resource_failure"):
                entry["resource_failure"] = "host_memory_reserve_pressure"
                running = ROOT / victim["job"]["job"] / "running.json"
                try:
                    os.killpg(read(running)["pid"], signal.SIGKILL)
                except (OSError, ValueError, KeyError):
                    pass

        if time.monotonic() - last_guard > 60 and not stopping:
            try:
                guards(plan)
            except Exception as error:
                journal["error"] = repr(error)
                stopping = True
            last_guard = time.monotonic()
        stopping = stopping or (directory / "stop-after-current").exists()
        if not stopping:
            for job in list(pending):
                cpu = free_cpu(cpus, active)
                if cpu is None:
                    break
                decision = admission(
                    psutil.virtual_memory().available, minimum, active, job
                )
                if not decision["allowed"]:
                    continue
                try:
                    guards(plan)
                    item = start(job, cpu, directory, decision)
                except Exception as error:
                    entries[job["id"]].update(status="failed", error=repr(error))
                    stopping = True
                    break
                active.append(item)
                pending.remove(job)
                entries[job["id"]].update(
                    status="running",
                    started_utc=now(),
                    cpu_ids=[cpu],
                    launcher_pid=item["process"].pid,
                    admission=decision,
                )

        journal.update(
            status="draining" if stopping and active else "running",
            sampled_utc=now(),
            host_available_bytes=available,
            active_jobs=len(active),
            pending_jobs=len(pending),
            new_admissions_stopped=stopping,
        )
        write(directory / "runner.json", journal)
        if stopping and not active:
            break
        if pending or active:
            time.sleep(1)
    completed = sum(e["status"] == "completed" for e in entries.values())
    failed = (
        any(e["status"] == "failed" for e in entries.values()) or "error" in journal
    )
    journal.update(
        status=(
            "completed"
            if completed == len(entries)
            else "failed" if failed else "stopped_pending"
        ),
        ended_utc=now(),
        completed_jobs=completed,
        active_jobs=0,
    )
    write(directory / "runner.json", journal)
    print(
        json.dumps(
            dict(status=journal["status"], completed=completed, total=len(entries))
        ),
        flush=True,
    )
    return 0 if journal["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "runner.json").exists():
            parser.error(
                "Controller already has a journal; inspect existing receipts instead of overwriting"
            )
        plan = read(ROOT / args.plan)
        guards(plan)
        cpus = plan["resources"]["planned_cpu_pool"]
        if (
            len(cpus) > 12
            or not cpus
            or not set(cpus).issubset(os.sched_getaffinity(0))
        ):
            parser.error("CPU pool must contain at most 12 available logical CPUs")
        if plan["resources"]["logical_cpus_per_job"] != 1:
            parser.error("This controller requires one CPU per job")
        for job in plan["jobs"]:
            if (ROOT / job["output"]).exists() or (ROOT / job["job"]).exists():
                parser.error("Refusing to overwrite condition: " + job["id"])
        extra = [
            "native_research/run_bandwidth.py",
            "native_research/named_job.py",
            "native_research/aligned_audit.py",
            "native_research/show_jobs.py",
        ]
        extra += [
            str(path.relative_to(ROOT))
            for suffix in ("*.yaml", "*.json")
            for path in (ROOT / "TokenSim/data").rglob(suffix)
        ]
        plan["execution_locks"] = {name: digest(ROOT / name) for name in extra}
        write(directory / "plan.snapshot.json", plan)
        journal = dict(
            status="starting",
            started_utc=now(),
            controller_pid=os.getpid(),
            plan_sha256=digest(directory / "plan.snapshot.json"),
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            python=sys.executable,
            pythonpath=os.environ.get("PYTHONPATH"),
            memory_policy="8 GiB native RLIMIT_AS and sampled RSS limit; reserve >=2 GiB at admission, monitor host pressure; abort before reserve exhaustion",
            admission_estimates=plan["resources"].get(
                "admission_estimates",
                "Unmeasured TP8 full workloads; conservative per-backend/trace estimates plus live peak growth reserve",
            ),
            jobs=[
                dict(
                    id=job["id"], status="pending", output=job["output"], job=job["job"]
                )
                for job in plan["jobs"]
            ],
        )
        os.sched_setaffinity(0, cpus)
        write(directory / "runner.json", journal)
        raise SystemExit(execute(plan, directory, journal, cpus))


if __name__ == "__main__":
    main()
