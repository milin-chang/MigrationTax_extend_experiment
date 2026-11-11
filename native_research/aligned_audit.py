"""Read-only close checks for an aligned native condition."""

from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def guards(plan):
    locks = dict(plan["source_sha256"])
    for info in plan["inputs"].values():
        locks[info["path"]] = info["sha256"]
        locks[info["raw_source"]["path"]] = info["raw_source"]["sha256"]
    locks.update(plan.get("execution_locks", {}))
    for name, expected in locks.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen file changed: " + name)


def audit(plan, job):
    output = ROOT / job["output"]
    condition = read(output / "condition.json")
    summary = read(output / "summary.json")
    config = read(output / "effective-config.json")
    receipt = read(ROOT / job["job"] / "job.json")
    failures = []

    def check(ok, name):
        if not ok:
            failures.append(name)

    check(
        receipt["returncode"] == 0 and receipt["status"] == "passed", "process receipt"
    )
    check(
        receipt["timeout_seconds"] is None and not receipt["timed_out"],
        "no wall timeout",
    )
    check(len(receipt["cpu_ids"]) == 1, "one CPU")
    check(receipt["memory_limit_bytes"] == 8 * 1024**3, "8 GiB limit")
    check(condition["status"] == summary["status"] == "completed", "completion status")
    check(condition["limit"] == job.get("limit"), "full/diagnostic scope")
    check(condition["input_sha256"] == job["input_sha256"], "input hash")
    check(condition.get("raw_source_unchanged") is True, "raw source unchanged")
    check(
        condition.get("analytical_evaluate_used_for_timing") is False,
        "native timing guard",
    )
    check(condition.get("author_answers_read") is False, "input provenance guard")
    check(config["hardware_alignment"] == plan["hardware_alignment"], "shared hardware")
    check(
        config["hardware_memory_bandwidth_bytes_per_s"] == 3352e9,
        "unchanged HBM bandwidth",
    )
    check(
        config["communication_link"]["bandwidth_per_direction_bytes_per_s"]
        == job["bandwidth_bytes_per_s"],
        "communication bandwidth",
    )
    check(config["simulation"]["block_size"] == 64, "64-token block geometry")
    hardware = plan["hardware_alignment"]
    check(
        config["cluster"]["num_workers"] == hardware["total_gpus"], "physical GPU ranks"
    )
    tp = hardware["tensor_parallel"]
    check(config["simulation"]["handover_tp_groups"] == (tp > 1), "TP handover mode")
    if tp > 1:
        groups = config["tensor_parallel_handover"]
        replicas = groups["replica_ranks"]
        ranks = [rank for group in replicas.values() for rank in group]
        check(
            groups["enabled"] and groups["physical_gpus"] == hardware["total_gpus"],
            "TP physical inventory",
        )
        check(
            len(replicas) == 8 and all(len(group) == tp for group in replicas.values()),
            "four P and four D TP replicas",
        )
        check(len(set(ranks)) == hardware["total_gpus"], "unique TP ranks")
    check(
        summary["completed"] == summary["input_requests"] == job["expected_requests"],
        "population",
    )
    check(not summary["rejected"] and not summary["unfinished"], "rejected/unfinished")
    check(summary["native_scheduler_executed"] is True, "native scheduler")
    inputs = [json.loads(line) for line in (ROOT / job["input"]).open()]
    inputs = inputs[: job.get("limit")]
    seen, generated = set(), 0
    for line in (output / "requests.jsonl").open():
        row = json.loads(line)
        source_id = int(row["source_id"])
        if source_id in seen or not 0 <= source_id < len(inputs):
            failures.append("duplicate/unknown source ID")
            continue
        seen.add(source_id)
        original = inputs[source_id]
        check(row["generated"] == original["output_tokens"], f"output {source_id}")
        check(row["arrival_s"] == original["arrival_seconds"], f"arrival {source_id}")
        check(row["status"] == "completed", f"status {source_id}")
        check(
            row["first_d_output_visible_at_s"] >= row["arrival_s"],
            f"D first output {source_id}",
        )
        check(
            row["first_d_output_visible_at_s"] <= row["completed_at"] + 1e-8,
            f"D output before completion {source_id}",
        )
        completed_at = row[
            "completed_at_s" if job["backend"] == "splitwise" else "completed_at"
        ]
        check(
            math.isfinite(completed_at) and completed_at >= row["arrival_s"],
            f"time {source_id}",
        )
        generated += row["generated"]
    check(len(seen) == job["expected_requests"], "unique completed requests")
    check(generated == job["expected_output_tokens"], "total output tokens")
    before, after = (
        condition["initial_source_sha256"],
        condition["imported_source_sha256"],
    )
    for name, expected in before.items():
        check(after.get(name) == expected, "source changed during run: " + name)
    for name, expected in after.items():
        check(
            digest(output / "source-snapshot" / name) == expected, "snapshot: " + name
        )
    if job["backend"] == "tokensim":
        for device in read(output / "native-end-state.json"):
            check(device["gpu_status"][1] == 0, f"rank {device['id']} GPU owners")
            check(
                all(
                    not device[key]
                    for key in ("waiting", "running", "loading", "incoming", "pinned")
                ),
                f"rank {device['id']} pending",
            )
        for info in read(output / "native-admission-budget.json")["devices"].values():
            check(info["occupied_at_end"] == 0, "admission budget")
        if job["method"] == "MigrationTax":
            check(
                read(output / "native-migration-evidence.json")[
                    "store_leases_remaining"
                ]
                == 0,
                "store leases",
            )
    else:
        for name, info in summary["native_cache_accounting"].items():
            check(
                info["reserved_bytes"] == info["pinned_bytes"] == 0,
                "cache owners: " + name,
            )
    guards(plan)
    return dict(
        status="passed" if not failures else "failed",
        failures=failures,
        requests=len(seen),
        output_tokens=generated,
        wall_seconds=receipt["wall_seconds"],
        peak_rss_bytes=receipt["peak_aggregate_rss_bytes"],
    )
