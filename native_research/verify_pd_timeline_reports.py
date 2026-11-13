"""Cross-check compact P/D reports against receipts, raw requests and archives."""

import csv
from hashlib import sha256
import json
from zipfile import ZipFile

from .aligned_audit import ROOT, read
from .pd_timeline_analysis import PREFIX, TPS, close, request_intervals
from .plot_mooncake_link_study import digest


def verify_tp(tp):
    target = ROOT / "native_research/reports" / PREFIX.format(tp)
    data = read(target / "pd-analysis.json")
    jobs = {r["id"]: r for r in read(ROOT / "config" / PREFIX.format(tp) / "jobs.json")}
    evidence = []
    with ZipFile(target / "request-timelines.zip") as bundle:
        if len(bundle.namelist()) != 20:
            raise ValueError("Expected ten complete CSV/schema pairs")
        for check, summary in zip(data["checks"], data["requests"]):
            output = ROOT / summary["output"]
            schema = read(output / "timeline-schema.json")
            steps = sum(
                r["steps"]
                for r in data["steps"]
                if (r["link"], r["method"]) == (summary["link"], summary["method"])
            )
            if (
                schema["counts"]["worker_steps"] != steps
                or schema["counts"]["requests"] != 12031
            ):
                raise ValueError("Streaming counts differ from writer receipt")
            if schema["unfinished_requests"] or schema["unfinished_transfers"]:
                raise ValueError("Unfinished observations")
            native = {
                str(r["source_id"]): r
                for r in map(
                    json.loads, (output / "requests.jsonl").read_text().splitlines()
                )
            }
            with (output / "request-timeline.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    r = native[row["source_id"]]
                    request_intervals(row)
                    close(
                        float(row["first_client_output_at_s"])
                        - float(row["arrival_s"]),
                        float(row["ttft_s"]),
                    )
                    close(float(row["admission_wait_s"]), r["native_budget_wait_s"])
                    close(float(row["lifecycle_s"]), r["native_total_time_s"])
                    close(
                        float(row["first_client_output_at_s"]),
                        r["first_client_output_at_s"],
                    )
                    if r["first_d_output_sampled_at_s"] is None:
                        if row["first_d_output_at_s"]:
                            raise ValueError("Fabricated D output timestamp")
                    else:
                        close(
                            float(row["first_d_output_at_s"]),
                            r["first_d_output_sampled_at_s"],
                        )
                    parent = r["previous_source_id"]
                    ready = (
                        r["arrival_s"]
                        if parent is None
                        else max(r["arrival_s"], native[str(parent)]["completed_at"])
                    )
                    close(float(row["parent_ready_s"]), ready)
            baseline = (
                ROOT
                / jobs[summary["condition_id"]]["baseline_output"]
                / "requests.jsonl"
            )
            recorded_hash, baseline_hash = digest(output / "requests.jsonl"), digest(
                baseline
            )
            if recorded_hash != baseline_hash:
                raise ValueError("Observation changed full request records")
            for filename in ("request-timeline.csv", "timeline-schema.json"):
                archived = bundle.read(
                    f"{summary['link']}/{summary['method']}/{filename}"
                )
                if sha256(archived).hexdigest() != digest(output / filename):
                    raise ValueError(
                        "Published request archive differs from raw recording"
                    )
            if check["request_timeline_sha256"] != digest(
                output / "request-timeline.csv"
            ):
                raise ValueError("Analysis input changed")
            evidence.append(
                dict(
                    condition=summary["condition_id"],
                    status="passed",
                    requests=12031,
                    streamed_steps=steps,
                    request_records_identical=True,
                    recorded_requests_sha256=recorded_hash,
                    baseline_requests_sha256=baseline_hash,
                    baseline=str(baseline.relative_to(ROOT)),
                    request_timeline_sha256=check["request_timeline_sha256"],
                )
            )
    verification = read(target / "pd-verification.json")
    for figure in verification["figures"]:
        if digest(target / figure["file"]) != figure["sha256"]:
            raise ValueError("Figure changed after generation")
    result = dict(
        status="passed",
        tp=tp,
        conditions=10,
        requests_per_condition=12031,
        full_request_logs_identical=10,
        checks=evidence,
        request_archive_sha256=digest(target / "request-timelines.zip"),
    )
    (target / "pd-integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    groups = [verify_tp(tp) for tp in TPS]
    target = ROOT / "native_research/reports/mooncake-pd-timeline-tp1-4-8-s0-02"
    validation_root = (
        ROOT / "native_research/runs/native-timeline-budget-fix-validation-s0-128-01"
    )
    validation = dict(
        unit_and_coverage=read(validation_root / "final-verification.json"),
        cli_on_off=read(validation_root / "validation.json"),
    )
    (target / "native-observer-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    result = dict(
        status="passed", conditions=30, full_request_logs_identical=30, groups=groups
    )
    (target / "pd-integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Verified 30 conditions, 360,930 request observations, archives and figure hashes; all 30 request logs byte-identical to uninstrumented baselines."
    )


if __name__ == "__main__":
    main()
