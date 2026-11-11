"""Independent resource-admitted native job; no fixed job-count ceiling.

Each job has an exclusive name and one or two logical CPUs, an 8 GiB limit, and no formal
wall deadline. Other jobs are observed through fresh shared-file telemetry,
which works across PID namespaces. Combined samples are explicitly non-atomic.
"""

import argparse
from contextlib import ExitStack
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time
import psutil

MEMORY_BYTES = 8 * 1024**3
ROOT = Path(__file__).resolve().parents[1]


def peer_sample(job):
    job = ROOT / job
    if (job / "job.json").exists():
        return dict(job=str(job), status="closed", rss_bytes=0)
    path = job / "telemetry.jsonl"
    if not path.exists():
        return dict(job=str(job), status="not_started", rss_bytes=0)
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            record = json.loads(stream.read().splitlines()[-1])
        age = max(0.0, time.time() - path.stat().st_mtime)
        return dict(
            job=str(job),
            status="fresh" if age <= 15 else "stale",
            rss_bytes=record["aggregate_rss_bytes"],
            sample_age_s=age,
        )
    except (OSError, ValueError, IndexError, KeyError) as error:
        return dict(
            job=str(job),
            status="unavailable",
            rss_bytes=None,
            reason=type(error).__name__,
        )


def run(command, output, name, cpus, peers, seconds=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Use a plain job name")
    if (
        len(cpus) not in (1, 2)
        or len(set(cpus)) != len(cpus)
        or not set(cpus).issubset(os.sched_getaffinity(0))
    ):
        raise ValueError("Specify one or two distinct available logical CPUs")
    if seconds is not None and seconds <= 0:
        raise ValueError("Diagnostic deadline must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    peers = sorted({Path(p).resolve() for p in peers} - {output})
    launcher_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = peak_vms = peak_combined = 0
    timed_out, reason = False, None
    with ExitStack() as locks:
        for label in [name, *(f"cpu-{cpu}" for cpu in sorted(cpus))]:
            lock = locks.enter_context(
                Path(f"/tmp/migrationtax-native-{label}.lock").open("a")
            )
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        os.sched_setaffinity(0, cpus)

        def constrain():
            os.sched_setaffinity(0, cpus)
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
            os.setsid()

        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open(
            "w"
        ) as stderr, (output / "telemetry.jsonl").open("w") as telemetry:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=constrain,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(len(cpus)),
                    "OPENBLAS_NUM_THREADS": str(len(cpus)),
                    "MKL_NUM_THREADS": str(len(cpus)),
                    "NUMEXPR_NUM_THREADS": str(len(cpus)),
                    "WANDB_MODE": "disabled",
                    "WANDB_DISABLED": "true",
                },
            )
            (output / "running.json").write_text(
                json.dumps(
                    dict(
                        pid=process.pid,
                        command=command,
                        cpu_ids=cpus,
                        memory_limit_bytes=MEMORY_BYTES,
                        name=name,
                        launcher_sha256=launcher_hash,
                        timeout_seconds=seconds,
                    )
                )
                + "\n"
            )
            next_sample = 0.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= next_sample:
                    rss = vms = cpu_time = 0
                    try:
                        root = psutil.Process(process.pid)
                        for child in [root, *root.children(recursive=True)]:
                            try:
                                memory, timing = child.memory_info(), child.cpu_times()
                                rss += memory.rss
                                vms += memory.vms
                                cpu_time += timing.user + timing.system
                            except psutil.Error:
                                pass
                    except psutil.Error:
                        pass
                    peak_rss, peak_vms = max(peak_rss, rss), max(peak_vms, vms)
                    samples = [peer_sample(p) for p in peers]
                    complete = all(
                        p["status"] in ("fresh", "closed", "not_started")
                        for p in samples
                    )
                    combined = (
                        rss + sum(p["rss_bytes"] for p in samples) if complete else None
                    )
                    if combined is not None:
                        peak_combined = max(peak_combined, combined)
                    telemetry.write(
                        json.dumps(
                            dict(
                                elapsed_wall_s=elapsed,
                                cpu_seconds_live=cpu_time,
                                aggregate_rss_bytes=rss,
                                aggregate_vms_bytes=vms,
                                peers=samples,
                                combined_observed_rss_bytes=combined,
                                host_available_bytes=psutil.virtual_memory().available,
                            )
                        )
                        + "\n"
                    )
                    telemetry.flush()
                    next_sample = elapsed + 1
                    if rss > MEMORY_BYTES:
                        reason = "aggregate_memory_limit"
                        os.killpg(process.pid, signal.SIGKILL)
                if seconds is not None and elapsed > seconds and process.poll() is None:
                    reason, timed_out = "explicit_diagnostic_deadline", True
                    os.killpg(process.pid, signal.SIGKILL)
                time.sleep(0.2)
            code = process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    unchanged = sha256(Path(__file__).read_bytes()).hexdigest() == launcher_hash
    receipt = dict(
        command=command,
        returncode=code,
        status="passed" if code == 0 and unchanged else "failed",
        timed_out=timed_out,
        timeout_seconds=seconds,
        wall_seconds=time.monotonic() - started,
        cpu_ids=cpus,
        memory_limit_bytes=MEMORY_BYTES,
        cpu_seconds=after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime,
        max_child_rss_kib=after.ru_maxrss,
        peak_aggregate_rss_bytes=peak_rss,
        peak_aggregate_vms_bytes=peak_vms,
        termination_reason=reason or "child_exit",
        name=name,
        concurrency_policy="resource-admitted; no fixed job-count cap",
        peers=[str(p) for p in peers],
        peak_combined_observed_rss_bytes=peak_combined,
        combined_sampling="Own tree plus fresh per-job samples; ages retained; not atomic",
        launcher_sha256=launcher_hash,
        launcher_unchanged=unchanged,
        telemetry="telemetry.jsonl",
    )
    (output / "job.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--peer-jobs", type=Path, nargs="*", default=[])
    parser.add_argument("--seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    receipt = run(
        command, args.output, args.name, args.cpus, args.peer_jobs, args.seconds
    )
    print(json.dumps(receipt))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
