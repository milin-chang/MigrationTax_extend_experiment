"""Check observed P/D endpoints against native request records and CSV steps."""

import csv
import json
import math


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Inconsistent observed endpoint: {a} != {b}")


def verify_csv(output, tp):
    raw = {
        r["source_id"]: r
        for r in map(json.loads, (output / "requests.jsonl").read_text().splitlines())
    }
    timeline = rows(output / "request-timeline.csv")
    steps = rows(output / "worker-steps.csv")
    transfers = rows(output / "kv-transfers.csv")
    schema = json.loads((output / "timeline-schema.json").read_text())
    if (
        schema["status"] != "completed"
        or schema["unfinished_requests"]
        or schema["unfinished_transfers"]
    ):
        raise ValueError("Timeline is incomplete")
    if len(timeline) != len(raw) or {r["source_id"] for r in timeline} != raw.keys():
        raise ValueError("Missing or duplicate request timeline rows")
    step_ids = set()
    for step in steps:
        if step["status"] not in ("completed", "no_compute", "load_only"):
            raise ValueError("An observed step failed")
        if step["step_id"] in step_ids:
            raise ValueError("Duplicate step identity")
        step_ids.add(step["step_id"])
        ids = json.loads(step["request_ids"])
        if len(ids) != int(step["batch_size"]) or int(step["tp_group_size"]) != tp:
            raise ValueError("Batch or TP group cardinality is inconsistent")
        if set(ids) != json.loads(step["request_phases"]).keys():
            raise ValueError("Step phase membership differs")
        close(
            float(step["load_end_s"]) - float(step["load_start_s"]), step["load_wait_s"]
        )
        if step["first_compute_start_s"]:
            if float(step["first_compute_start_s"]) + 1e-8 < float(step["load_end_s"]):
                raise ValueError("Compute precedes blocking load completion")
            close(
                float(step["last_compute_end_s"])
                - float(step["first_compute_start_s"]),
                step["compute_span_s"],
            )
    by_source = {}
    for step in steps:
        for rid in json.loads(step["request_ids"]):
            by_source.setdefault(rid, []).append(step)
    for row in timeline:
        native = raw[row["source_id"]]
        if row["status"] != "completed":
            raise ValueError("Unexpected noncompleted request")
        if not row["d_budget_acquired_s"]:
            raise ValueError("Completed request lacks a D budget acquisition timestamp")
        close(row["d_budget_acquired_s"], row["submitted_s"])
        if (
            native["plan_branch"] != "decode_local_prefill"
            and not row["p_budget_acquired_s"]
        ):
            raise ValueError("P participant lacks a P budget acquisition timestamp")
        if row["p_budget_acquired_s"] and not (
            float(row["admission_started_s"])
            <= float(row["p_budget_acquired_s"])
            <= float(row["d_budget_acquired_s"])
        ):
            raise ValueError("Budget acquisition timestamps are out of order")
        close(row["ttft_s"], native["native_prefill_latency_s"])
        close(
            float(row["first_client_output_at_s"]) - float(row["arrival_s"]),
            row["ttft_s"],
        )
        close(
            float(row["submitted_s"]) - float(row["parent_ready_s"]),
            row["admission_wait_s"],
        )
        close(row["admission_wait_s"], native["native_budget_wait_s"])
        close(row["completed_at_s"], native["completed_at"])
        close(row["lifecycle_s"], native["native_total_time_s"])
        for prefix in ("p", "d"):
            if row[f"{prefix}_scheduler_enter_s"] and float(
                row[f"{prefix}_scheduler_enter_s"]
            ) < float(row[f"{prefix}_queue_enter_s"]):
                raise ValueError("Scheduler insertion precedes worker message arrival")
        if native["first_d_output_sampled_at_s"] is None:
            if row["first_d_output_at_s"]:
                raise ValueError("A missing D sample was replaced")
        else:
            close(row["first_d_output_at_s"], native["first_d_output_sampled_at_s"])
        if native["plan_branch"] == "decode_local_prefill" and row["p_prefill_start_s"]:
            raise ValueError("D-local request has fabricated P execution")
        own_steps = by_source[row["source_id"]]
        for prefix, role in (("p", "prefill"), ("d", "decode")):
            compute = [
                s for s in own_steps if s["role"] == role and s["first_compute_start_s"]
            ]
            if compute:
                close(
                    row[f"{prefix}_first_compute_start_s"],
                    min(float(s["first_compute_start_s"]) for s in compute),
                )
            if prefix == "p" and compute:
                close(
                    row["p_last_compute_end_s"],
                    max(float(s["last_compute_end_s"]) for s in compute),
                )
        prompt_steps = [
            s
            for s in own_steps
            if s["role"] == "prefill"
            and s["first_compute_start_s"]
            and json.loads(s["request_phases"])[row["source_id"]] == "prefill"
        ]
        if prompt_steps:
            close(
                row["p_prefill_start_s"],
                min(float(s["first_compute_start_s"]) for s in prompt_steps),
            )
            close(
                row["p_last_prefill_compute_end_s"],
                max(float(s["last_compute_end_s"]) for s in prompt_steps),
            )
            close(
                row["p_prefill_end_s"],
                max(float(s["step_completed_at_s"]) for s in prompt_steps),
            )
        if row["d_first_decode_step_end_s"]:
            candidates = [
                s
                for s in by_source[row["source_id"]]
                if s["role"] == "decode"
                and json.loads(s["request_phases"])[row["source_id"]] == "decode"
                and s["last_compute_end_s"]
            ]
            close(
                row["d_first_decode_step_end_s"],
                min(float(s["last_compute_end_s"]) for s in candidates),
            )
    for transfer in transfers:
        if transfer["status"] != "completed":
            raise ValueError("Incomplete explicit transfer")
        close(
            float(transfer["service_started_s"]) - float(transfer["queued_at_s"]),
            transfer["queue_wait_s"],
        )
        close(
            float(transfer["service_ended_s"]) - float(transfer["service_started_s"]),
            transfer["service_s"],
        )
    return dict(
        requests=len(timeline),
        steps=len(steps),
        explicit_transfers=len(transfers),
        status="passed",
    )
