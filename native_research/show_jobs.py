"""Read approved job receipts and telemetry without starting or changing jobs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "native_research/runs/resource-parallel-runner-01/fleet-plans.json"
PLAN = ROOT / "native_research/plans/tp8-aligned-current.json"
ACTIVE = {"RUNNING", "STARTING", "STALE", "READING"}


def document(path):
    for attempt in range(3):
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            if attempt < 2:
                time.sleep(0.02)
    return {}


def tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    except OSError:
        pass
    return {}


def option(command, name, fallback="-"):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return fallback


def load_jobs(manifest):
    """Accept a direct plan or the historical registry of multiple plans."""
    jobs = {
        str((ROOT / p).resolve()): ({}, {}) for p in manifest.get("primary_jobs", [])
    }
    unreadable = []
    plans = [manifest]
    for name in manifest.get("plans", []):
        plan = document(ROOT / name)
        if not plan:
            unreadable.append(name)
        plans.append(plan)
    for plan in plans:
        for entry in plan.get("jobs", []):
            # Keep receipts outside the worker's output: the worker creates it exclusively.
            output = ROOT / entry["output"]
            job = ROOT / entry.get("job", str(output.with_name(output.name + "-job")))
            jobs[str(job.resolve())] = (entry, plan.get("hardware_alignment", {}))
    return jobs, unreadable


def observe(job, metadata, hardware=None):
    running = document(job / "running.json")
    receipt = document(job / "job.json")
    audit = document(job / "audit.json")
    output = ROOT / metadata.get("output", str(job.parent / "output"))
    condition = document(output / "condition.json")
    config = document(output / "effective-config.json")
    summary = document(output / "summary.json")
    progress = document(output / "progress.json")
    telemetry_path = job / "telemetry.jsonl"
    telemetry = tail(telemetry_path)
    command = running.get("command", metadata.get("worker_argv", []))
    count = summary.get("completed", progress.get("completed", 0))
    total = summary.get("input_requests", progress.get("input_requests"))
    total = (
        total
        or metadata.get("expected_requests")
        or condition.get("input", {}).get("requests")
    )
    expected = metadata.get("expected_requests", total)
    actual_hardware = config.get("hardware_alignment", {})
    hardware_match = (actual_hardware == hardware) if hardware and config else None
    input_match = (
        condition.get("input_sha256") == metadata["input_sha256"]
        if metadata.get("input_sha256") and condition
        else None
    )
    if (job / "job.json").exists():
        if not receipt:
            status = "READING"
        elif receipt.get("termination_reason") == "user_requested_stop":
            status = "STOPPED"
        elif receipt.get("status") == "stopped_by_user":
            status = "USER_STOPPED"
        elif receipt.get("returncode") != 0 or receipt.get("status") == "failed":
            status = "FAILED"
        elif condition.get("status") == "unsupported":
            status = "UNSUPPORTED"
        elif (
            summary.get("status") == "completed"
            and total is not None
            and count == total == expected
            and not summary.get("rejected")
            and not summary.get("unfinished")
            and condition.get("status") == "completed"
        ):
            status = "DONE"
        else:
            status = "INCOMPLETE"
    elif running:
        try:
            age = (
                time.time()
                - (telemetry_path if telemetry else job / "running.json")
                .stat()
                .st_mtime
            )
            status = ("RUNNING" if telemetry else "STARTING") if age <= 15 else "STALE"
        except OSError:
            status = "READING"
    elif (job / "running.json").exists():
        status = "READING"
    elif summary or condition or config or progress or telemetry:
        status = "UNCONFIRMED"
    else:
        status = "PENDING"
    if status in {"DONE", "RUNNING", "STARTING", "UNCONFIRMED"}:
        if hardware_match is False or input_match is False:
            status = "MISMATCH"
        elif status == "DONE" and hardware and hardware_match is None:
            status = "INCOMPLETE"
    if status == "DONE" and audit.get("status") == "failed":
        status = "AUDIT_FAILED"
    return dict(
        state=status,
        backend=metadata.get("backend", option(command, "--backend")),
        trace=metadata.get("trace", option(command, "--trace")),
        method=metadata.get("method", option(command, "--method")),
        revision=config.get("method_semantics_version", "-"),
        seed=str(metadata.get("seed", option(command, "--seed"))),
        completed=count,
        total=expected,
        reported_total=total,
        hardware_match=hardware_match,
        input_match=input_match,
        audit_status=audit.get("status"),
        cpus=running.get("cpu_ids", receipt.get("cpu_ids", metadata.get("cpus", []))),
        elapsed_s=receipt.get("wall_seconds", telemetry.get("elapsed_wall_s")),
        rss_bytes=None if receipt else telemetry.get("aggregate_rss_bytes"),
        peak_rss_bytes=receipt.get("peak_aggregate_rss_bytes"),
        returncode=receipt.get("returncode"),
        termination_reason=receipt.get("termination_reason"),
        job=str(job.relative_to(ROOT)) if job.is_relative_to(ROOT) else str(job),
    )


def duration(seconds):
    if seconds is None:
        return "-"
    minutes, seconds = divmod(int(seconds), 60)
    return (
        f"{minutes // 60}h{minutes % 60:02}m"
        if minutes >= 60
        else f"{minutes}m{seconds:02}s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all", action="store_true", help="Include finished and pending jobs"
    )
    selection.add_argument(
        "--active", action="store_true", help="Show only active or stale observations"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print structured observations"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--plan", type=Path, help="Read a plan (paths are relative to the repository)"
    )
    source.add_argument("--registry", type=Path, help="Read a historical registry")
    source.add_argument(
        "--legacy", action="store_true", help="Read the historical registry"
    )
    args = parser.parse_args()
    path = args.plan or args.registry or (REGISTRY if args.legacy else PLAN)
    manifest = document(ROOT / path)
    if not manifest:
        parser.exit(1, f"Cannot read job plan/registry: {path}\n")
    jobs, unreadable = load_jobs(manifest)
    observations = []
    for name, (metadata, hardware) in sorted(jobs.items()):
        observations.append(observe(Path(name), metadata, hardware))
    # Direct plans show their entire matrix; retain active-only defaults for old registries.
    show_all = args.all or ("jobs" in manifest and not args.active)
    selected = (
        observations if show_all else [r for r in observations if r["state"] in ACTIVE]
    )
    counts = dict(Counter(r["state"] for r in observations))
    if unreadable:
        status = "UNCONFIRMED"
    elif observations and counts.get("PENDING") == len(observations):
        status = "NOT_STARTED"
    elif observations and counts.get("DONE") == len(observations):
        status = "COMPLETED"
    elif any(r["state"] not in ACTIVE | {"DONE", "PENDING"} for r in observations):
        status = "ATTENTION"
    else:
        status = "IN_PROGRESS" if observations else "EMPTY"
    if args.json:
        print(
            json.dumps(
                dict(
                    sampled_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    non_atomic_samples=True,
                    unreadable_plans=unreadable,
                    scope=str(path),
                    status=status,
                    counts=counts,
                    total_jobs=len(observations),
                    hardware_alignment=manifest.get("hardware_alignment"),
                    resources=manifest.get("resources"),
                    jobs=selected,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"Scope: {path}\nStatus: {status}  Total: {len(observations)}")
    print("  ".join(f"{state}: {count}" for state, count in sorted(counts.items())))
    hardware = manifest.get("hardware_alignment", {})
    if hardware:
        print(
            f"Planned: {hardware['model_id']} / {hardware['model_dtype']} / "
            f"{hardware['gpu_id']} / TP{hardware['tensor_parallel']} / "
            f"{hardware['total_gpus']} GPUs"
        )
    for trace, info in manifest.get("inputs", {}).items():
        print(
            f"Input: {trace} {info.get('version', '-')} / {info.get('requests', '-')} requests"
        )
    resources = manifest.get("resources", {})
    if resources:
        print(
            f"CPU plan: {resources.get('logical_cpus_per_job', '-')} logical CPU/job; "
            f"pool {','.join(map(str, resources.get('planned_cpu_pool', [])))} (actual CPUs below)"
        )
    revisions = {
        "native-full-model-mixed-v3": "mixed-v3",
        "native-fused-first-v2": "phase-v2",
    }
    table = [
        [
            "STATE",
            "BACKEND",
            "REV",
            "TRACE",
            "METHOD",
            "SEED",
            "COMPLETED",
            "CPU",
            "WALL",
            "RAM_GiB",
            "PEAK_GiB",
        ]
    ]
    for r in selected:
        count = f"{r['completed']:,}"
        if r["total"]:
            count += f"/{r['total']:,} ({r['completed'] / r['total']:.0%})"
        table.append(
            [
                r["state"],
                r["backend"],
                revisions.get(r["revision"], r["revision"]),
                r["trace"],
                r["method"],
                r["seed"],
                count,
                ",".join(map(str, r["cpus"])) or "-",
                duration(r["elapsed_s"]),
                (
                    f"{r['rss_bytes'] / 1024**3:.2f}"
                    if r["rss_bytes"] is not None
                    else "-"
                ),
                (
                    f"{r['peak_rss_bytes'] / 1024**3:.2f}"
                    if r["peak_rss_bytes"] is not None
                    else "-"
                ),
            ]
        )
    widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    for row in table:
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    if any(r["state"] == "STALE" for r in selected):
        print(
            "STALE: no fresh telemetry for 15s; process state is unconfirmed, not necessarily stopped."
        )
    if unreadable:
        print(
            "Some plans are being updated or unavailable; retry:", ", ".join(unreadable)
        )
    print(
        "WALL is real elapsed time, not simulated time. Progress is the latest saved request count."
    )
    print(
        "Read-only, non-atomic snapshots. DONE requires process/workload receipts and plan identity checks;"
    )
    print(
        "it is not a full scientific acceptance audit. UNCONFIRMED means missing process receipts."
    )


if __name__ == "__main__":
    main()
