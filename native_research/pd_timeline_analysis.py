"""Verify request clocks and stream native step diagnostics with bounded memory."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
import statistics

from .aligned_audit import ROOT, read
from .plot_ttft_decomposition import collect

TPS = (1, 4, 8)
PREFIX = "mooncake7b-h100-tp{}-pd-timeline-s0-02"
STEP_PARTS = (
    "load_wait_s",
    "precompute_wait_s",
    "compute_service_s",
    "between_compute_wait_s",
    "postcompute_wait_s",
)
DELTAS = {
    "p_message_wait": ("p_scheduler_enter_s", "p_queue_enter_s"),
    "p_scheduler_to_compute": ("p_first_compute_start_s", "p_scheduler_enter_s"),
    "p_prompt_span": ("p_prefill_end_s", "p_prefill_start_s"),
    "p_prompt_kernel_window": ("p_last_prefill_compute_end_s", "p_prefill_start_s"),
    "d_message_wait": ("d_scheduler_enter_s", "d_queue_enter_s"),
    "d_scheduler_to_compute": ("d_first_compute_start_s", "d_scheduler_enter_s"),
    "d_queue_to_compute": ("d_first_compute_start_s", "d_queue_enter_s"),
    "d_queue_to_output": ("first_d_output_at_s", "d_queue_enter_s"),
    "d_queue_to_decode_end": ("d_first_decode_step_end_s", "d_queue_enter_s"),
}


def number(row, name):
    return None if row[name] == "" else float(row[name])


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-10, abs_tol=1e-8
    ):
        raise ValueError(f"Clock mismatch: {actual} != {expected}")


def difference(row, end, start):
    a, b = number(row, end), number(row, start)
    if a is None or b is None:
        return None
    result = a - b
    if not math.isfinite(result) or result < -1e-8:
        raise ValueError(f"Invalid interval {start} -> {end}: {row['source_id']}")
    return max(0.0, result)


def request_intervals(row):
    """The P-then-D gate split also handles a successful D-only fast path."""
    values = {key: difference(row, *fields) for key, fields in DELTAS.items()}
    if not row["d_budget_acquired_s"]:
        raise ValueError("Missing D budget timestamp")
    if not row["p_budget_acquired_s"] and row["plan_branch"] != "decode_local_prefill":
        raise ValueError("Missing required P budget timestamp")
    values["gate_p_wait"] = (
        difference(row, "p_budget_acquired_s", "admission_started_s") or 0.0
    )
    start = (
        "p_budget_acquired_s" if row["p_budget_acquired_s"] else "admission_started_s"
    )
    values["gate_d_after_p_wait"] = difference(row, "d_budget_acquired_s", start)
    values["after_admission"] = difference(
        row, "first_client_output_at_s", "submitted_s"
    )
    close(number(row, "submitted_s"), number(row, "d_budget_acquired_s"))
    close(
        values["gate_p_wait"] + values["gate_d_after_p_wait"],
        number(row, "admission_wait_s"),
    )
    close(
        number(row, "parent_wait_s")
        + number(row, "admission_wait_s")
        + values["after_admission"],
        number(row, "ttft_s"),
    )
    return values


def describe(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, maximum=None)
    return dict(
        n=len(values),
        mean=statistics.mean(values),
        p50=statistics.median(values),
        p95=values[math.ceil(0.95 * len(values)) - 1],
        maximum=values[-1],
    )


def request_summary(run, base):
    output = ROOT / run["output"]
    with (output / "request-timeline.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    native = {
        str(r["source_id"]): r
        for r in map(json.loads, (output / "requests.jsonl").read_text().splitlines())
    }
    if len(rows) != 12031 or {r["source_id"] for r in rows} != native.keys():
        raise ValueError("Timeline IDs/count differ from raw requests")
    series = defaultdict(list)
    branches = Counter()
    for row in rows:
        if row["status"] != "completed":
            raise ValueError("Incomplete timeline")
        original = native[row["source_id"]]
        close(number(row, "ttft_s"), original["native_prefill_latency_s"])
        close(number(row, "arrival_s"), original["arrival_s"])
        close(number(row, "completed_at_s"), original["completed_at"])
        close(
            number(row, "parent_wait_s"),
            number(row, "parent_ready_s") - original["arrival_s"],
        )
        if int(row["source_turn"]) != original["source_turn"]:
            raise ValueError("Request turn differs")
        branches[row["plan_branch"]] += 1
        for key, value in request_intervals(row).items():
            if value is not None:
                series[key].append(value)
        if row["d_first_decode_batch_size"]:
            series["d_first_decode_batch_size"].append(
                float(row["d_first_decode_batch_size"])
            )
    result = dict(base, condition_id=run["id"], output=run["output"])
    for key in (
        "p50_ttft_s",
        "p95_ttft_s",
        "max_ttft_s",
        "mean_lifecycle_s",
        "mean_tpot_s",
    ):
        result[key] = run[key]
    for key, values in series.items():
        for stat, value in describe(values).items():
            result[f"{key}_{stat}"] = value
    close(
        result["gate_p_wait_mean"] + result["gate_d_after_p_wait_mean"],
        result["mean_admission_wait_s"],
    )
    result["p_budget_observed_n"] = sum(bool(r["p_budget_acquired_s"]) for r in rows)
    result["d_budget_observed_n"] = sum(bool(r["d_budget_acquired_s"]) for r in rows)
    result["branches_json"] = json.dumps(branches, sort_keys=True)
    return result, rows


def stream_steps(path, metadata, sample_id=None):
    accumulators = {}
    samples = []
    for role in ("prefill", "decode"):
        accumulators[role] = dict(
            n=0, sums=Counter(), maxima=Counter(), batches=Counter()
        )
    sample_counts = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "completed":
                raise ValueError(f"Incomplete step: {path}")
            a = accumulators[row["role"]]
            a["n"] += 1
            numeric = {key: float(row[key]) for key in STEP_PARTS}
            close(
                sum(numeric.values()),
                float(row["step_completed_at_s"]) - float(row["scheduled_at_s"]),
            )
            if min(numeric.values()) < -1e-8:
                raise ValueError("Negative step duration")
            numeric["batch_size"] = int(row["batch_size"])
            numeric["batch_tokens"] = int(row["batch_tokens"])
            numeric["blocking_load_nonzero"] = numeric["load_wait_s"] > 0
            for key, value in numeric.items():
                a["sums"][key] += value
                a["maxima"][key] = max(value, a["maxima"][key])
            a["batches"][numeric["batch_size"]] += 1
            if (
                sample_id
                and sample_counts[row["role"]] < 12
                and sample_id in json.loads(row["request_ids"])
            ):
                samples.append(dict(metadata, **row))
                sample_counts[row["role"]] += 1
    summaries, histogram = [], []
    for role, a in accumulators.items():
        record = dict(metadata, role=role, steps=a["n"])
        if not a["n"]:
            raise ValueError(f"No {role} steps")
        for key, total in a["sums"].items():
            record[f"{key}_mean"] = total / a["n"]
            record[f"{key}_max"] = a["maxima"][key]
        summaries.append(record)
        histogram += [
            dict(metadata, role=role, batch_size=size, steps=count)
            for size, count in sorted(a["batches"].items())
        ]
    return summaries, histogram, samples


def stream_transfers(path, metadata, sample_id=None):
    groups = {}
    samples = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "completed":
                raise ValueError("Incomplete explicit transfer")
            close(
                float(row["service_ended_s"]) - float(row["service_started_s"]),
                float(row["service_s"]),
            )
            close(
                float(row["service_started_s"]) - float(row["queued_at_s"]),
                float(row["queue_wait_s"]),
            )
            a = groups.setdefault(row["kind"], Counter())
            a["transfers"] += 1
            for key in ("bytes", "queue_wait_s", "service_s"):
                a[key] += float(row[key])
            if sample_id == row["source_id"] and len(samples) < 64:
                samples.append(dict(metadata, **row))
    records = [dict(metadata, kind=kind, **a) for kind, a in sorted(groups.items())]
    return records, samples


def collect_tp(tp):
    source = ROOT / "native_research/reports" / PREFIX.format(tp)
    bases, checks = collect(source, tp)
    runs = read(source / "results.json")["runs"]
    rows_by_condition, summaries = {}, []
    for run in runs:
        base = next(
            b for b in bases if (b["link"], b["method"]) == (run["link"], run["method"])
        )
        summary, rows = request_summary(run, base)
        summaries.append(summary)
        rows_by_condition[run["id"]] = rows
    finite450 = [r for r in runs if r["link"] == "nvl4"]
    eligible = None
    for run in finite450:
        ids = {
            r["source_id"]
            for r in rows_by_condition[run["id"]]
            if int(r["source_turn"]) > 0
            and r["p_prefill_start_s"]
            and r["d_first_compute_start_s"]
        }
        eligible = ids if eligible is None else eligible & ids
    serial = next(r for r in finite450 if r["method"] == "Serial")
    population = sorted(
        (r for r in rows_by_condition[serial["id"]] if r["source_id"] in eligible),
        key=lambda r: (float(r["ttft_s"]), r["source_id"]),
    )
    sample_id = population[len(population) // 2]["source_id"]
    steps, histogram, transfers, sample_steps, sample_transfers, sample_requests = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for run, summary, check in zip(runs, summaries, checks):
        output = ROOT / run["output"]
        metadata = dict(tp=tp, link=run["link"], method=run["method"])
        selected = sample_id if run["link"] == "nvl4" else None
        a, b, c = stream_steps(output / "worker-steps.csv", metadata, selected)
        steps += a
        histogram += b
        sample_steps += c
        a, b = stream_transfers(output / "kv-transfers.csv", metadata, selected)
        transfers += a
        sample_transfers += b
        if selected:
            sample_requests += [
                dict(metadata, **r)
                for r in rows_by_condition[run["id"]]
                if r["source_id"] == selected
            ]
        schema = read(output / "timeline-schema.json")
        if schema["status"] != "completed":
            raise ValueError("Incomplete timeline receipt")
        check.update(
            timeline_requests=12031,
            worker_steps=sum(r["steps"] for r in steps[-2:]),
            p_budget_observed=summary["p_budget_observed_n"],
            d_budget_observed=summary["d_budget_observed_n"],
            request_timeline_sha256=sha256(
                (output / "request-timeline.csv").read_bytes()
            ).hexdigest(),
        )
        print(
            f"Verified TP{tp} {run['link']} {run['method']}: {check['worker_steps']:,} steps",
            flush=True,
        )
    return dict(
        requests=summaries,
        steps=steps,
        batch_histogram=histogram,
        transfers=transfers,
        sample_requests=sample_requests,
        sample_steps=sample_steps,
        sample_transfers=sample_transfers,
        checks=checks,
        selection=dict(
            tp=tp,
            source_id=sample_id,
            eligible_returning=len(population),
            rule="Upper median by Serial 450 GB/s TTFT among returning IDs with P/D observations in all three methods; same ID across methods",
        ),
    )
