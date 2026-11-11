"""Create the frozen ten-condition ServeGen communication-bandwidth matrix."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from native_methods.hardware_profile import load_profile
from native_methods.inputs import load_requests
from native_methods.worker import lock_raw_source

ROOT = Path(__file__).resolve().parents[1]
RUN = "native_research/runs/servegen7b-bw-s0-8000-01"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default="configs/hardware-llama7b-h100-tp1-block64.json"
    )
    parser.add_argument("--input", default="inputs/servegen-s0-first8000.jsonl")
    parser.add_argument("--run-root", default=RUN)
    parser.add_argument("--study-dir", default="configs")
    parser.add_argument(
        "--output", default="native_research/plans/servegen7b-bandwidth-s0-8000.json"
    )
    args = parser.parse_args()
    profile, source = args.profile, args.input
    rows = load_requests(ROOT / source, "servegen")
    if not rows:
        parser.error("The study requires at least one request")
    path = ROOT / args.output
    if path.exists():
        parser.error("Refusing to overwrite an existing frozen plan")
    jobs = []
    for label, bandwidth in (
        ("reference", 1e15),
        ("bw50", 50e9),
        ("bw100", 100e9),
        ("bw200", 200e9),
    ):
        for method in (
            ("Serial",) if label == "reference" else ("Serial", "PPD", "MigrationTax")
        ):
            output = f"{args.run_root}/{label}/{method}/seed-0"
            jobs.append(
                dict(
                    id=f"tokensim-servegen7b-tp{load_profile(ROOT / profile).tensor_parallel}-n{len(rows)}-{label}-{method.lower()}-s0",
                    backend="tokensim",
                    trace="servegen",
                    method=method,
                    seed=0,
                    bandwidth_label=label,
                    bandwidth_bytes_per_s=bandwidth,
                    reference=label == "reference",
                    input=source,
                    input_sha256=digest(ROOT / source),
                    expected_requests=len(rows),
                    expected_output_tokens=sum(r["output_tokens"] for r in rows),
                    output=output,
                    job=output + "-job",
                    memory_estimate_bytes=3 * 1024**3,
                    worker_argv=[
                        sys.executable,
                        "-m",
                        "native_methods.worker",
                        "--backend",
                        "tokensim",
                        "--method",
                        method,
                        "--trace",
                        "servegen",
                        "--seed",
                        "0",
                        "--input",
                        source,
                        "--hardware-profile",
                        profile,
                        "--experiment-config",
                        f"{args.study_dir}/study_{label}.json",
                        "--output",
                        output,
                    ],
                )
            )
    paths = []
    for name in (
        "algorithm",
        "native_methods",
        "migrationtax/reference",
        "TokenSim/TokenSim",
    ):
        paths.extend((ROOT / name).rglob("*.py"))
    paths += [p for p in (ROOT / "configs").rglob("*") if p.is_file()]
    paths += [
        ROOT / "Trace/sources.json",
        ROOT / "native_research/run_bandwidth.py",
        ROOT / "native_research/aligned_audit.py",
        ROOT / "native_research/named_job.py",
        ROOT / "native_research/build_bandwidth_plan.py",
        ROOT / "native_research/prepare_tp8_bandwidth.py",
        ROOT / "native_research/run_overnight.py",
        ROOT / "native_research/report_bandwidth.py",
        ROOT / "native_research/plot_bandwidth.py",
        ROOT / "native_research/show_bandwidth_jobs.py",
    ]
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        hardware_alignment=load_profile(ROOT / profile).describe(),
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted(set(paths))
            if "__pycache__" not in p.parts
        },
        inputs={
            "servegen-s0": dict(
                path=source,
                sha256=digest(ROOT / source),
                requests=len(rows),
                raw_source=lock_raw_source("servegen"),
            )
        },
        resources=dict(
            planned_cpu_pool=sorted(os.sched_getaffinity(0))[:12],
            logical_cpus_per_job=1,
            minimum_host_available_bytes=2 * 1024**3,
            admission_estimates="3 GiB per condition plus live peak growth; 8 GiB hard per-job cap",
        ),
        jobs=jobs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
