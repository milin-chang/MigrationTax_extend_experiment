"""Reaudit completed conditions and independently reconstruct paired metrics."""

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import statistics

from .aligned_audit import ROOT, audit, guards, read


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def requests(output):
    return {
        row["source_id"]: row
        for row in map(json.loads, (output / "requests.jsonl").read_text().splitlines())
    }


def verify(run, destination):
    plan = read(run / "controller/plan.snapshot.json")
    guards(plan)
    reference_job = next(job for job in plan["jobs"] if job["reference"])
    reference = requests(ROOT / reference_job["output"])
    checks, decomposition, native_matches = [], [], []
    receipts = []
    for job in plan["jobs"]:
        result = audit(plan, job)
        if result["status"] != "passed":
            raise ValueError(result)
        output = ROOT / job["output"]
        index = requests(output)
        if index.keys() != reference.keys():
            raise ValueError("Paired source IDs differ")
        rows = list(index.values())
        parent = [
            (
                max(
                    0.0,
                    index[row["previous_source_id"]]["completed_at"] - row["arrival_s"],
                )
                if row["previous_source_id"] is not None
                else 0.0
            )
            for row in rows
        ]
        budget = [row["native_budget_wait_s"] for row in rows]
        ttft = [row["native_prefill_latency_s"] for row in rows]
        after = [t - p - b for t, p, b in zip(ttft, parent, budget)]
        if min(after) < -1e-8:
            raise ValueError("TTFT components overlap or have inconsistent clocks")
        decomposition.append(
            dict(
                bandwidth=job["bandwidth_label"],
                method=job["method"],
                requests=len(rows),
                mean_ttft_s=statistics.mean(ttft),
                median_ttft_s=statistics.median(ttft),
                mean_parent_wait_s=statistics.mean(parent),
                mean_admission_wait_s=statistics.mean(budget),
                mean_after_admission_to_client_token_s=statistics.mean(after),
                parent_wait_requests=sum(value > 1e-8 for value in parent),
                admission_wait_requests=sum(value > 1e-8 for value in budget),
                max_admission_wait_s=max(budget),
            )
        )
        native = read(next((output / "native").glob("result_*.json")))
        native_matches.append(native["operator_match_type_counts"])
        if (
            native["parallel_tp_collective_latency"] <= 0
            or native["parallel_sync_event_count"] <= 0
        ):
            raise ValueError("No TP collective events executed")
        receipts.append(read(ROOT / job["job"] / "job.json"))
        if job["reference"]:
            continue
        paired = csv_rows(
            destination / f'{job["bandwidth_label"]}-{job["method"]}-requests.csv'
        )
        if len(paired) != len(rows) or len({row["source_id"] for row in paired}) != len(
            rows
        ):
            raise ValueError("Paired CSV population differs")
        returning = affected_d = affected_client = 0
        for row in paired:
            actual, baseline = index[row["source_id"]], reference[row["source_id"]]
            if (
                float(row["reference_d_latency_s"])
                != baseline["d_output_sampled_latency_s"]
            ):
                raise ValueError("D reference endpoint differs")
            if (
                float(row["reference_client_ttft_s"])
                != baseline["native_prefill_latency_s"]
            ):
                raise ValueError("Client reference endpoint differs")
            if actual["source_turn"] == 0:
                continue
            returning += 1
            d = (
                actual["d_output_sampled_latency_s"]
                > 2 * baseline["d_output_sampled_latency_s"]
            )
            client = (
                actual["native_prefill_latency_s"]
                > 2 * baseline["native_prefill_latency_s"]
            )
            if (row["affected_d"] == "True") != d or (
                row["affected_client"] == "True"
            ) != client:
                raise ValueError("Paired affected classification differs")
            affected_d += d
            affected_client += client
        expected = sum(row["source_turn"] > 0 for row in rows)
        if returning != expected:
            raise ValueError("Returning denominator differs")
        checks.append(
            dict(
                bandwidth=job["bandwidth_label"],
                method=job["method"],
                returning=returning,
                affected_d_count=affected_d,
                affected_client_count=affected_client,
            )
        )
    state = read(run / "orchestrator.json")
    elapsed = (
        datetime.fromisoformat(state["ended_utc"])
        - datetime.fromisoformat(state["started_utc"])
    ).total_seconds()
    verification = dict(
        status="passed",
        conditions=len(plan["jobs"]),
        requests_per_condition=len(reference),
        returning_per_condition=sum(
            row["source_turn"] > 0 for row in reference.values()
        ),
        output_tokens_per_condition=sum(row["generated"] for row in reference.values()),
        paired_csv_rechecks=checks,
        all_native_compute_matches_analytical=all(
            set(counts) == {"analytical"} for counts in native_matches
        ),
        all_TP_collectives_executed=True,
        frozen_plan_guard="passed",
        wall_seconds=elapsed,
        min_job_wall_seconds=min(row["wall_seconds"] for row in receipts),
        max_job_wall_seconds=max(row["wall_seconds"] for row in receipts),
        peak_rss_gib=max(row["peak_aggregate_rss_bytes"] for row in receipts) / 2**30,
        report_sha256=sha256((destination / "results.md").read_bytes()).hexdigest(),
        ttft_decomposition=decomposition,
    )
    (destination / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    write_csv(destination / "ttft-decomposition.csv", decomposition)
    lines = [
        "# TTFT 等待拆分",
        "",
        "各欄均為每請求平均值，單位為秒。",
        "",
        "TTFT＝前輪等待＋准入預算等待＋通過准入至首 token。",
        "通過准入至首 token 仍包含原生排程、計算與傳輸。",
        "",
        "| 頻寬 GB/s | 方法 | Avg TTFT s | 前輪等待 s | 准入等待 s | 通過准入至首 token s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in decomposition:
        values = [
            f"{row[key]:.6f}"
            for key in (
                "mean_ttft_s",
                "mean_parent_wait_s",
                "mean_admission_wait_s",
                "mean_after_admission_to_client_token_s",
            )
        ]
        lines.append(
            f'| {row["bandwidth"].removeprefix("bw")} | {row["method"]} | '
            + " | ".join(values)
            + " |"
        )
    (destination / "TTFT_DECOMPOSITION.md").write_text("\n".join(lines) + "\n")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root, args.report_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "conditions",
                    "requests_per_condition",
                    "wall_seconds",
                    "peak_rss_gib",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
