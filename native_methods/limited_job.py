"""Single native job: two CPUs, 8 GiB, telemetry, optional wall deadline.

Formal conditions have no wall-time ceiling by default. Diagnostic deadlines
are explicit options. Long duration or absent request completions alone do
not trigger termination; memory limits and child exit remain enforced.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import psutil

MEMORY_BYTES = 8 * 1024**3


def run(command, output, seconds=None):
    if seconds is not None and seconds <= 0:
        raise ValueError("Diagnostic deadline must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    peak_rss = 0
    peak_vms = 0
    reason = None
    timed_out = False
    with Path("/tmp/migrationtax-research-single-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

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
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "WANDB_MODE": "disabled",
                    "WANDB_DISABLED": "true",
                },
            )
            (output / "running.json").write_text(
                json.dumps(
                    dict(
                        pid=process.pid,
                        command=command,
                        timeout_seconds=seconds,
                        cpu_ids=cpus,
                        memory_limit_bytes=MEMORY_BYTES,
                    )
                )
                + "\n"
            )
            next_sample = 0.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= next_sample:
                    rss = vms = cpu = 0.0
                    try:
                        root = psutil.Process(process.pid)
                        for child in [root] + root.children(recursive=True):
                            try:
                                mem = child.memory_info()
                                times = child.cpu_times()
                                rss += mem.rss
                                vms += mem.vms
                                cpu += times.user + times.system
                            except psutil.Error:
                                pass
                    except psutil.Error:
                        pass
                    peak_rss = max(peak_rss, rss)
                    peak_vms = max(peak_vms, vms)
                    telemetry.write(
                        json.dumps(
                            dict(
                                elapsed_wall_s=elapsed,
                                cpu_seconds_live=cpu,
                                aggregate_rss_bytes=int(rss),
                                aggregate_vms_bytes=int(vms),
                            )
                        )
                        + "\n"
                    )
                    telemetry.flush()
                    next_sample = elapsed + 5
                    if rss > MEMORY_BYTES:
                        reason = "aggregate_memory_limit"
                        os.killpg(process.pid, signal.SIGKILL)
                if seconds is not None and elapsed > seconds and process.poll() is None:
                    reason = "explicit_diagnostic_deadline"
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                time.sleep(0.2)
            code = process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(
        command=command,
        returncode=code,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - started,
        timeout_seconds=seconds,
        cpu_ids=cpus,
        memory_limit_bytes=MEMORY_BYTES,
        cpu_seconds=(
            after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        ),
        max_child_rss_kib=after.ru_maxrss,
        peak_aggregate_rss_bytes=int(peak_rss),
        peak_aggregate_vms_bytes=int(peak_vms),
        termination_reason=reason or "child_exit",
        status="passed" if code == 0 else "failed",
        telemetry="telemetry.jsonl",
    )
    (output / "job.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Optional diagnostic deadline; omitted means no wall-time ceiling",
    )
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("A child command is required")
    receipt = run(command, a.output, a.seconds)
    print(json.dumps(receipt))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
