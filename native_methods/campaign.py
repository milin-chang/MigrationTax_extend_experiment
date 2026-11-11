"""Run formal native conditions sequentially; preserve semantic failure status."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from .limited_job import run
from .inputs import ROOT
from .reuse import completed_conditions, copy_verified

BACKENDS = ("splitwise", "tokensim", "llmservingsim")
AVAILABLE_BACKENDS = (*BACKENDS, "simai")
METHODS = ("Serial", "PPD", "MigrationTax")
FORMAL_TARGET_CONDITIONS = len(BACKENDS) * len(METHODS) * 2 * 4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument(
        "--backends",
        nargs="+",
        choices=AVAILABLE_BACKENDS,
        default=BACKENDS,
        help="Default: three active backends; simai is retained for explicit historical replay",
    )
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument(
        "--traces",
        nargs="+",
        choices=("mooncake", "servegen"),
        default=["mooncake", "servegen"],
    )
    p.add_argument(
        "--reuse-completed",
        type=Path,
        help="Verify input/environment/code before copying completed full conditions",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional diagnostic deadline; formal default is no wall-time ceiling",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, choices=range(4), default=list(range(4))
    )
    p.add_argument(
        "--stop-file",
        type=Path,
        help="Finish current condition, then defer remaining conditions if this file exists",
    )
    p.add_argument(
        "--max-new-conditions",
        type=int,
        help="Return to the job scheduler after this many new conditions; verified reuse does not count",
    )
    a = p.parse_args()
    a.output = a.output.resolve()
    a.inputs = a.inputs.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    conditions = []
    for backend in a.backends:
        for trace in a.traces:
            for method in a.methods:
                for seed in a.seeds:
                    source = a.inputs / f"{trace}-s{seed}.jsonl"
                    conditions.append(
                        dict(
                            backend=backend,
                            trace=trace,
                            method=method,
                            seed=seed,
                            input=str(source),
                            input_sha256=sha256(source.read_bytes()).hexdigest(),
                            expected_requests=sum(
                                1 for s in source.open() if s.strip()
                            ),
                            status="planned",
                        )
                    )
    manifest = dict(
        status="running",
        formal_target_conditions=FORMAL_TARGET_CONDITIONS,
        active_goal_backends=list(BACKENDS),
        deferred_by_user_scope=["simai"],
        selected_conditions=len(conditions),
        conditions=conditions,
        analytical_results_used_as_native=False,
        timeout_seconds=a.timeout_seconds,
        cpu_limit=2,
        memory_limit_bytes=8 * 1024**3,
    )
    target = a.output / "campaign.json"

    def save():
        target.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    failed = {}
    previous = completed_conditions(a.reuse_completed) if a.reuse_completed else {}
    newly_run = 0
    for c in conditions:
        if (a.stop_file and a.stop_file.exists()) or (
            a.max_new_conditions is not None and newly_run >= a.max_new_conditions
        ):
            c.update(status="deferred_by_scheduling")
            save()
            continue
        if c["backend"] in failed:
            c.update(status="not_run_after_failure", cause=failed[c["backend"]])
            save()
            continue
        path = a.output / c["backend"] / c["trace"] / c["method"] / f"seed-{c['seed']}"
        old = previous.get((c["backend"], c["trace"], c["method"], c["seed"]))
        proof = copy_verified(old, c, path) if old else None
        if proof:
            c.update(
                status="completed",
                completed=c["expected_requests"],
                reused_from=proof["source_campaign"],
                job=str((path / "job/job.json").relative_to(a.output)),
                output=str((path / "output").relative_to(a.output)),
            )
            save()
            print(
                f"{c['backend']} {c['trace']} {c['method']} seed{c['seed']}: verified completed evidence reused",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "-m",
            "native_methods.worker",
            "--backend",
            c["backend"],
            "--method",
            c["method"],
            "--trace",
            c["trace"],
            "--seed",
            str(c["seed"]),
            "--input",
            c["input"],
            "--output",
            str(path / "output"),
        ]
        receipt = run(command, path / "job", a.timeout_seconds)
        newly_run += 1
        c["job"] = str((path / "job/job.json").relative_to(a.output))
        c["output"] = str((path / "output").relative_to(a.output))
        state = path / "output/condition.json"
        result = json.loads(state.read_text()) if state.exists() else {}
        if receipt["timed_out"]:
            c["status"] = "timeout"
        elif receipt["returncode"] != 0:
            c["status"] = "failed"
        else:
            c["status"] = result.get("status", "missing_condition_status")
        summary = path / "output/summary.json"
        if summary.exists():
            values = json.loads(summary.read_text())
            for key in ("input_requests", "completed", "rejected", "unfinished"):
                if key in values:
                    c[key] = values[key]
            if (
                values.get("status") == "completed"
                and values.get("completed") != c["expected_requests"]
            ):
                c["status"] = "incomplete"
        if c["status"] not in ("completed", "unsupported"):
            failed[c["backend"]] = c["output"]
        save()
        print(
            f"{c['backend']} {c['trace']} {c['method']} seed{c['seed']}: {c['status']}",
            flush=True,
        )
    manifest["status"] = (
        "completed"
        if all(c["status"] == "completed" for c in conditions)
        else "partial"
    )
    manifest["formal_matrix_complete"] = (
        len(conditions) == FORMAL_TARGET_CONDITIONS
        and set(a.backends) == set(BACKENDS)
        and manifest["status"] == "completed"
    )
    # Native run counts do not verify metrics, audits, documents or reruns.
    manifest["overall_goal_complete"] = False
    save()


if __name__ == "__main__":
    main()
