"""Freeze fresh full Mooncake matrices with streaming P/D diagnostics."""

from copy import deepcopy
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from native_methods.hardware_profile import load_profile
from native_methods.inputs import load_requests
from .aligned_audit import ROOT, guards, read
from .prepare_mooncake_link_study import digest, relative, write


def prepare(tp, revision="01"):
    name = f"mooncake7b-h100-tp{tp}-pd-timeline-s0-{revision}"
    design = ROOT / "config" / name
    run = ROOT / "native_research/runs" / name
    report = ROOT / "native_research/reports" / name
    original = ROOT / f"config/five-stage-llama7b-h100-sxm80gb-tp{tp}-s0"
    old_run = f"native_research/runs/five-stage-llama7b-h100-tp{tp}-s0-01"
    old_report = old_run.replace("/runs/", "/reports/")
    if any(p.exists() for p in (design, run, report)):
        raise FileExistsError(f"Existing experiment: {name}")

    def remap(value):
        if isinstance(value, str):
            return (
                value.replace(relative(original), relative(design))
                .replace(old_run, relative(run))
                .replace(old_report, relative(report))
            )
        if isinstance(value, list):
            return [remap(x) for x in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    design.mkdir(parents=True)
    for filename in ("hardware.json", "cluster.json", "metrics.json"):
        shutil.copy2(original / filename, design / filename)
    write(design / "links.json", remap(read(original / "links.json")))
    for link in ("inf", "nvl4", "pcie6", "ib400"):
        shutil.copytree(original / f"data_{link}", design / f"data_{link}")
        shutil.copy2(
            original / f"transfer_{link}.json", design / f"transfer_{link}.json"
        )
        write(
            design / f"study_{link}.json", remap(read(original / f"study_{link}.json"))
        )
    (design / "inputs").mkdir()
    source = original / "inputs/mooncake-s0.jsonl"
    target = design / "inputs/mooncake-s0.jsonl"
    shutil.copy2(source, target)
    rows = load_requests(target, "mooncake")
    if (
        len(rows),
        sum(r["turn"] > 0 for r in rows),
        sum(r["output_tokens"] for r in rows),
    ) != (12031, 3959, 4122048):
        raise ValueError("Full Mooncake workload changed")
    if digest(source) != digest(target):
        raise ValueError("Copied input differs")
    profile = load_profile(design / "hardware.json")
    if profile.tensor_parallel != tp:
        raise ValueError("Hardware TP differs")
    manifest = deepcopy(read(original / "input-manifest.json")["mooncake"])
    manifest.update(
        path=relative(target), copied_from=relative(source), sha256=digest(target)
    )
    write(design / "input-manifest.json", {"mooncake": manifest})
    jobs = []
    for old in read(original / "jobs.json"):
        if old["trace"] != "mooncake" or old["phase"] not in (1, 3, 4):
            continue
        job = remap(old)
        job["id"] = (
            f"pd{revision}-tp{tp}-{job['bandwidth_label']}-{job['method'].lower()}-s0"
        )
        job["worker_argv"][0] = sys.executable
        job["worker_argv"].append("--record-timeline")
        job["memory_estimate_bytes"] = 2 * 2**30
        job["baseline_output"] = old["output"]
        if "--limit" in job["worker_argv"] or job["expected_requests"] != 12031:
            raise ValueError("An experiment was truncated")
        jobs.append(job)
    if len(jobs) != 10 or sum(j["reference"] for j in jobs) != 1:
        raise ValueError("Expected ten conditions with one reference")
    write(design / "jobs.json", jobs)
    resources = dict(
        planned_cpu_pool=sorted(os.sched_getaffinity(0))[:12],
        logical_cpus_per_job=1,
        minimum_host_available_bytes=2 * 2**30,
        memory_limit_bytes=8 * 2**30,
        admission_estimates="2 GiB initial estimate per full Mooncake job; live RSS and peak growth reserve determine concurrency; no fixed three-job cap; retain 2 GiB host reserve",
    )
    batch = dict(
        id="all-mooncake",
        phase="1+3+4",
        count=10,
        plan=relative(design / "execution/all-mooncake.plan.json"),
    )
    write(design / "execution/index.json", dict(batches=[batch], total_jobs=10))
    write(
        design / "experiment.json",
        dict(
            name=name,
            hardware=profile.describe(),
            seed=0,
            trace="mooncake",
            input_version="shifted v1",
            selected_phases=[1, 3, 4],
            total_jobs=10,
            resources=resources,
            run_root=relative(run),
            report_dir=relative(report),
            record_timeline=True,
            scheduling="All bandwidths share one independent-job queue within each TP; TP studies run sequentially",
        ),
    )
    (design / "SPECIFICATIONS.md").write_text(
        f"# Mooncake TP{tp} P/D timeline run\n\n"
        f"Run name: `{name}`.\n\n"
        f"llama-7b/FP16, H100 SXM, TP{tp}, PP1, 4P+4D, {8*tp} simulated GPUs.\n"
        "Full shifted-v1 seed-0 input: 12,031 requests, 3,959 returning turns, 4,122,048 output tokens.\n"
        "One Serial inf reference and Serial/PPD/MigrationTax at 450/128/50 GB/s: ten conditions.\n"
        "Every native worker receives `--record-timeline`; per-request, worker-step and explicit MigrationTax transfer CSVs stay in its seed-0 output.\n"
        "Independent bandwidth jobs share one resource-admitted queue; TP1, TP4 and TP8 execute in order.\n"
        "The controller uses up to 12 logical CPUs, one CPU per job, an 8 GiB per-job cap and a 2 GiB host reserve.\n"
        "Hardware, simulation capacity and input are copied from the corresponding previous TP experiment.\n"
    )
    paths = []
    for name in (
        "algorithm",
        "native_methods",
        "migrationtax/reference",
        "TokenSim/TokenSim",
    ):
        paths.extend((ROOT / name).rglob("*.py"))
    paths += [p for p in design.rglob("*") if p.is_file()]
    paths += [ROOT / "Trace/sources.json", Path(__file__)]
    paths += [
        ROOT / f"native_research/{name}.py"
        for name in (
            "prepare_mooncake_link_study",
            "run_mooncake_link_study",
            "run_bandwidth",
            "named_job",
            "aligned_audit",
            "show_jobs",
            "report_bandwidth",
            "report_five_stage_links",
        )
    ]
    locks = {
        relative(p): digest(p)
        for p in sorted(set(paths))
        if "__pycache__" not in p.parts
    }
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        phase="1+3+4",
        batch="all-mooncake",
        hardware_alignment=profile.describe(),
        source_sha256=locks,
        inputs={"mooncake": manifest},
        jobs=jobs,
        resources=resources,
    )
    guards(plan)
    write(ROOT / batch["plan"], plan)
    write(
        design / "validation.json",
        dict(
            status="passed",
            simulations_launched=False,
            requests=12031,
            returning=3959,
            output_tokens=4122048,
            conditions=10,
            input_sha256=digest(target),
            byte_identical_to_previous_tp_run=True,
            record_timeline_on_every_job=True,
            previous_hardware_profile_sha256=digest(original / "hardware.json"),
            current_hardware_profile_sha256=digest(design / "hardware.json"),
        ),
    )
    # validation.json is an output of preparation and does not participate in locks.
    return dict(
        name=run.name,
        design=relative(design),
        run=relative(run),
        report=relative(report),
        plan=batch["plan"],
        tensor_parallel=tp,
        jobs=10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=("01", "02"), default="01")
    args = parser.parse_args()
    registry = (
        ROOT
        / f"native_research/runs/mooncake7b-pd-timeline-s0-{args.revision}-monitor.json"
    )
    if registry.exists():
        raise FileExistsError(registry)
    studies = [prepare(tp, args.revision) for tp in (1, 4, 8)]
    write(
        registry,
        dict(plans=[s["plan"] for s in studies], studies=studies, total_jobs=30),
    )
    print(json.dumps(dict(studies=studies, monitor=relative(registry)), indent=2))


if __name__ == "__main__":
    main()
