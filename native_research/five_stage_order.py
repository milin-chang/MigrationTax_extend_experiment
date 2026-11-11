"""Prioritize all Mooncake jobs while retaining the five logical phases."""

import json
from pathlib import Path

BATCHES = (
    ("00-sanity", 0, "mooncake"),
    ("01-mooncake-fast", 1, "mooncake"),
    ("02-mooncake-pcie6", 3, "mooncake"),
    ("03-mooncake-ib400", 4, "mooncake"),
    ("04-servegen-fast", 2, "servegen"),
    ("05-servegen-pcie6", 3, "servegen"),
    ("06-servegen-ib400", 4, "servegen"),
)


def batches(jobs):
    result = []
    for number, (name, phase, trace) in enumerate(BATCHES):
        selected = [
            job for job in jobs if job["phase"] == phase and job["trace"] == trace
        ]
        result.append(
            dict(
                id=name,
                phase=phase,
                trace=trace,
                count=len(selected),
                job_ids=[job["id"] for job in selected],
                previous_batch=BATCHES[number - 1][0] if number else None,
            )
        )
    flattened = [key for batch in result for key in batch["job_ids"]]
    if (
        len(jobs) != 22
        or len(set(flattened)) != 22
        or set(flattened) != {job["id"] for job in jobs}
    ):
        raise ValueError("Mooncake-first order must cover every job exactly once")
    if [batch["count"] for batch in result] != [2, 4, 3, 3, 4, 3, 3]:
        raise ValueError("Unexpected batch populations")
    return result


def configure(design):
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    jobs = json.loads((design / "jobs.json").read_text())
    order = batches(jobs)
    ranks = {
        key: index for index, batch in enumerate(order) for key in batch["job_ids"]
    }
    jobs.sort(key=lambda job: ranks[job["id"]])
    write(design / "jobs.json", jobs)
    write(
        design / "execution-batches.json",
        dict(
            priority="mooncake_first",
            logical_phases=5,
            execution_batches=order,
            total_jobs=22,
        ),
    )
    for path in (design / "phases").glob("*.json"):
        phase = json.loads(path.read_text())
        phase.pop("depends_on_phases", None)
        phase["execution_batches"] = [
            batch["id"] for batch in order if batch["phase"] == phase["phase"]
        ]
        phase["dependency_rule"] = (
            "Follow execution-batches.json; formal finite-link metrics share their trace-specific reference"
        )
        write(path, phase)
    path = design / "experiment.json"
    experiment = json.loads(path.read_text())
    experiment["execution_priority"] = "mooncake_first"
    experiment["execution_batches"] = "execution-batches.json"
    experiment["first_delivery"] = dict(
        after_batch="03-mooncake-ib400",
        cumulative_jobs=12,
        sanity_jobs=2,
        mooncake_formal_jobs=10,
        contents=[
            "All Mooncake per-request CSVs",
            "Affected share for 450, 128 and 50 GB/s",
            "Mooncake reference and native metrics",
        ],
    )
    experiment["phase2_delivery"] = dict(
        after_batch="04-servegen-fast",
        cumulative_jobs=16,
        includes="All Mooncake results and ServeGen reference plus three methods at 450 GB/s",
    )
    write(path, experiment)
