"""Freeze five separately scheduled phases without starting native jobs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from native_methods.hardware_profile import load_profile
from .aligned_audit import ROOT, digest, guards, read
from .five_stage_order import batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    design = args.design.resolve()
    destination = design / "execution"
    if destination.exists():
        parser.error("Refusing to overwrite frozen phase plans")
    paths = []
    for name in (
        "algorithm",
        "native_methods",
        "migrationtax/reference",
        "TokenSim/TokenSim",
    ):
        paths.extend((ROOT / name).rglob("*.py"))
    paths += [path for path in design.rglob("*") if path.is_file()]
    paths += [ROOT / "Trace/sources.json"]
    paths += [
        ROOT / f"native_research/{name}.py"
        for name in (
            "run_bandwidth",
            "named_job",
            "aligned_audit",
            "show_jobs",
            "report_bandwidth",
            "design_five_stage_links",
            "build_five_stage_plans",
            "run_five_stage_links",
            "report_five_stage_links",
            "validate_five_stage_links",
            "plot_five_stage_links",
            "show_five_stage_jobs",
            "five_stage_order",
        )
    ]
    locks = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(set(paths))
        if "__pycache__" not in path.parts
    }
    inputs = read(design / "input-manifest.json")
    hardware = load_profile(design / "hardware.json").describe()
    destination.mkdir()
    plans = []
    inventory = read(design / "jobs.json")
    for batch in batches(inventory):
        jobs = [job for job in inventory if job["id"] in batch["job_ids"]]
        for job in jobs:
            job["memory_estimate_bytes"] = (
                2
                if job["trace"] == "mooncake"
                else 5 if job["method"] == "MigrationTax" else 3
            ) * 2**30
        selected = {
            "mooncake_sanity" if batch["phase"] == 0 else job["trace"] for job in jobs
        }
        plan = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            phase=batch["phase"],
            batch=batch["id"],
            hardware_alignment=hardware,
            source_sha256=locks,
            inputs={key: inputs[key] for key in selected},
            jobs=jobs,
            resources=dict(
                planned_cpu_pool=sorted(os.sched_getaffinity(0))[:12],
                logical_cpus_per_job=1,
                minimum_host_available_bytes=2 * 2**30,
                admission_estimates="2 GiB per Mooncake job; 3 GiB per full ServeGen Serial/PPD job; 5 GiB per full ServeGen MigrationTax job; sampled growth reserve and 8 GiB per-job cap",
            ),
        )
        guards(plan)
        path = destination / f'{batch["id"]}.plan.json'
        path.write_text(json.dumps(plan, indent=2) + "\n")
        plans.append(
            dict(
                phase=batch["phase"],
                batch=batch["id"],
                trace=batch["trace"],
                plan=str(path.relative_to(ROOT)),
                count=len(jobs),
                job_ids=batch["job_ids"],
            )
        )
    (destination / "index.json").write_text(
        json.dumps(
            dict(phases=plans, total_jobs=sum(row["count"] for row in plans)), indent=2
        )
        + "\n"
    )
    print(destination / "index.json")


if __name__ == "__main__":
    main()
