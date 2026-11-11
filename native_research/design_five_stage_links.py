"""Write a design-only, five-stage TP8 link study with explicit provenance."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import yaml

from native_methods.hardware_profile import load_profile
from native_methods.inputs import load_requests
from native_methods.worker import lock_raw_source
from .five_stage_order import configure

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp8-s0"
PREVIOUS = ROOT / "configs/servegen7b-tp8-8000"
METHODS = ("Serial", "PPD", "MigrationTax")
LINKS = (
    ("nvl4", "nvlink4", 450e9),
    ("pcie6", "pcie6_x16", 128e9),
    ("ib400", "ib_ndr400", 50e9),
    ("inf", "inf_link", 1e15),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def relative(path):
    return str(path.relative_to(ROOT))


def inputs():
    original = ROOT.with_name("Simulator-alignment-review")
    raw_source = original / "Trace/mooncake/conversation_trace.jsonl"
    raw_target = ROOT / "Trace/mooncake/conversation_trace.jsonl"
    catalog = json.loads((ROOT / "Trace/sources.json").read_text())
    pinned = next(row for row in catalog["files"] if row["name"] == "Mooncake")
    if digest(raw_source) != pinned["sha256"]:
        raise ValueError("Mooncake raw source differs from its pinned version")
    if raw_target.exists():
        if digest(raw_target) != pinned["sha256"]:
            raise ValueError("Existing Mooncake raw source differs")
    else:
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_source, raw_target)
    sources = {
        "mooncake": original
        / "research/results/mooncake-shifted-v1/inputs/mooncake-s0.jsonl",
        "servegen": original / "research/results/primary-v1/inputs/servegen-s0.jsonl",
    }
    result = {}
    for trace, source in sources.items():
        target = DESIGN / "inputs" / f"{trace}-s0.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows = load_requests(target, trace)
        result[trace] = dict(
            path=relative(target),
            sha256=digest(target),
            source=relative(source) if source.is_relative_to(ROOT) else str(source),
            source_sha256=digest(source),
            raw_source=lock_raw_source(trace),
            version=(
                "shifted v1"
                if trace == "mooncake"
                else "primary v1, full 39698-request normalized input"
            ),
            requests=len(rows),
            returning=sum(row["turn"] > 0 for row in rows),
            output_tokens=sum(row["output_tokens"] for row in rows),
            arrivals_rescaled=False,
            tokens_clipped=False,
        )
    full = ROOT / result["mooncake"]["path"]
    sanity = DESIGN / "inputs/mooncake-s0-first200.jsonl"
    sanity.write_bytes(b"".join(full.read_bytes().splitlines(keepends=True)[:200]))
    rows = load_requests(sanity, "mooncake")
    result["mooncake_sanity"] = {
        **result["mooncake"],
        "path": relative(sanity),
        "sha256": digest(sanity),
        "selection": "Chronological first 200 rows of the full shifted-v1 seed-0 input; parents retained",
        "requests": len(rows),
        "returning": sum(row["turn"] > 0 for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
    }
    if (
        result["mooncake"]["requests"],
        result["servegen"]["requests"],
        result["mooncake_sanity"]["requests"],
    ) != (12031, 39698, 200):
        raise ValueError("Unexpected trace populations")
    return result


def link_configs():
    descriptions = {}
    for label, link_name, bandwidth in LINKS:
        data = DESIGN / f"data_{label}"
        shutil.copytree(PREVIOUS / "data_reference", data)
        path = data / "topologies/links.yaml"
        catalog = yaml.safe_load(path.read_text())
        if label == "inf":
            catalog["links"][link_name] = dict(
                bandwidth_per_direction_bytes_per_s=1e15,
                latency_us=0.0,
                half_bandwidth_bytes=0,
                source_id="five-stage-ideal-reference",
                grade="D",
            )
        selected = deepcopy(catalog["links"][link_name])
        if float(selected["bandwidth_per_direction_bytes_per_s"]) != bandwidth:
            raise ValueError("Link peak bandwidth differs from requested design")
        catalog["links"]["bandwidth_study_link"] = selected
        path.write_text(yaml.safe_dump(catalog, sort_keys=False))
        path = data / "devices/h100_sxm.yaml"
        device = yaml.safe_load(path.read_text())
        device["interconnect"]["scale_up_link"] = "nvlink4"
        device["interconnect"]["host_link"] = link_name
        if device["device_id"] != "h100_sxm":
            raise ValueError("H100 operator package identity changed")
        path.write_text(yaml.safe_dump(device, sort_keys=False))
        transfer = json.loads((PREVIOUS / "transfer_reference.json").read_text())
        latency_us = float(selected["latency_us"])
        for child in transfer["kv_connector_extra_config"]["connectors"]:
            options = child["kv_connector_extra_config"]
            options["bandwidth_bytes_per_s"] = bandwidth
            # Native P/D transfer topology is always cross_node and multiplies
            # configured protocol latency by 1.5. Compensate to expose exactly
            # the selected link's single-hop latency; embedded Store uses host_link.
            options["fixed_latency_us"] = (
                latency_us / 1.5
                if child["kv_connector"] == "MooncakeConnector"
                else 0.0
            )
        write(DESIGN / f"transfer_{label}.json", transfer)
        write(
            DESIGN / f"study_{label}.json",
            dict(
                label=label,
                bandwidth_bytes_per_s=bandwidth,
                reference=label == "inf",
                time_unit="s",
                bandwidth_unit="decimal bytes/s",
                bandwidth_scope="Full TP KV bundle per replica endpoint; not per rank",
                data_root=relative(data),
                cluster_path=relative(DESIGN / "cluster.json"),
                transfer_path=relative(DESIGN / f"transfer_{label}.json"),
                input_manifest=relative(DESIGN / "input-manifest.json"),
                input_selection="Full Mooncake shifted-v1 seed-0 input or full ServeGen primary-v1 seed-0 input without conversation exclusions; stage 0 alone uses a chronological 200-row Mooncake prefix; see phase job input",
                tp_collective_link="nvlink4",
                tp_collective_bandwidth_bytes_per_s=450e9,
                memory_bandwidth_changed=False,
            ),
        )
        descriptions[label] = dict(
            data_root=relative(data),
            selected_link=link_name,
            bandwidth_bytes_per_s=bandwidth,
            latency_us=latency_us,
            tp_scale_up_link="nvlink4",
            host_link=link_name,
            native_link_metadata=selected,
            kv_service_formula="selected_link.latency_us/1e6 + full_TP_KV_bytes / selected_link.bandwidth_per_direction_bytes_per_s",
            half_bandwidth_ramp_applied_to_kv=False,
            collective_efficiency_applied_to_kv=False,
            note="Native uniform Mooncake KV transport uses peak bandwidth and fixed latency only; link ramp/collective fields remain catalog metadata. TP collectives use fixed nvlink4 independently.",
        )
    return descriptions


def make_job(phase, trace, label, method, manifest):
    selected = manifest["mooncake_sanity" if phase == 0 else trace]
    key = f"p{phase}-{trace}-{label}-{method.lower()}-s0"
    output = f"native_research/runs/five-stage-llama7b-h100-tp8-s0-01/phase-{phase}/{trace}/{label}/{method}/seed-0"
    reference = (
        None
        if phase == 0
        else f"native_research/runs/five-stage-llama7b-h100-tp8-s0-01/phase-{1 if trace == 'mooncake' else 2}/{trace}/inf/Serial/seed-0"
    )
    return dict(
        id=key,
        phase=phase,
        backend="tokensim",
        trace=trace,
        method=method,
        seed=0,
        input=selected["path"],
        input_sha256=selected["sha256"],
        expected_requests=selected["requests"],
        expected_output_tokens=selected["output_tokens"],
        expected_returning_requests=selected["returning"],
        bandwidth_label=label,
        bandwidth_bytes_per_s=dict((a, c) for a, b, c in LINKS)[label],
        reference=label == "inf",
        reference_output=reference,
        output=output,
        job=output + "-job",
        hardware_profile=relative(DESIGN / "hardware.json"),
        experiment_config=relative(DESIGN / f"study_{label}.json"),
        worker_argv=[
            sys.executable,
            "-m",
            "native_methods.worker",
            "--backend",
            "tokensim",
            "--method",
            method,
            "--trace",
            trace,
            "--seed",
            "0",
            "--input",
            selected["path"],
            "--hardware-profile",
            relative(DESIGN / "hardware.json"),
            "--experiment-config",
            relative(DESIGN / f"study_{label}.json"),
            "--output",
            output,
        ],
    )


def main():
    if DESIGN.exists():
        raise FileExistsError("Refusing to overwrite the five-stage design")
    DESIGN.mkdir(parents=True)
    for name in ("hardware.json", "cluster.json"):
        shutil.copy2(PREVIOUS / name, DESIGN / name)
    manifest = inputs()
    write(DESIGN / "input-manifest.json", manifest)
    links = link_configs()
    write(DESIGN / "links.json", links)
    stages = [
        (0, "sanity", [("mooncake", label, "Serial") for label in ("nvl4", "ib400")]),
        (
            1,
            "mooncake-fast",
            [("mooncake", "inf", "Serial")]
            + [("mooncake", "nvl4", method) for method in METHODS],
        ),
        (
            2,
            "servegen-fast",
            [("servegen", "inf", "Serial")]
            + [("servegen", "nvl4", method) for method in METHODS],
        ),
        (
            3,
            "both-pcie6",
            [
                (trace, "pcie6", method)
                for trace in ("mooncake", "servegen")
                for method in METHODS
            ],
        ),
        (
            4,
            "both-ib400",
            [
                (trace, "ib400", method)
                for trace in ("mooncake", "servegen")
                for method in METHODS
            ],
        ),
    ]
    jobs, phases = [], []
    for number, label, members in stages:
        selected = [make_job(number, *member, manifest) for member in members]
        phase = dict(
            phase=number,
            name=label,
            generation_does_not_execute=True,
            depends_on_phases=list(range(number)),
            jobs=selected,
            count=len(selected),
            completion_rule="All process, source, full output-token and ownership audits pass; emit CSVs before proceeding",
            sanity_additional_checks=(
                [
                    "Same 200 source requests and identical HBM/TP configuration",
                    "Native KV service uses bytes/B plus selected-link latency",
                    "Compare paired D-first distributions for requests with positive transferred KV; do not require every queued request to be monotone",
                ]
                if number == 0
                else []
            ),
            delivery_after_phase=number == 2,
        )
        path = DESIGN / "phases" / f"{number:02d}-{label}.json"
        write(path, phase)
        phases.append(dict(phase=number, path=relative(path), jobs=len(selected)))
        jobs.extend(selected)
    write(DESIGN / "jobs.json", jobs)
    write(
        DESIGN / "metrics.json",
        dict(
            returning="source_turn > 0",
            arrival="Original trace arrival before parent/admission waits",
            d_first="first_d_output_sampled_at_s - arrival_s",
            client_ttft="native_prefill_latency_s; exported separately",
            affected="D-first strictly greater than 2 times the same source request in the trace-specific Serial inf reference",
            denominator="All returning requests in the full source input; no cache-cell filtering",
            decode_miss="Inherited prefix not completely resident on the eventually chosen D at original arrival; include partial coverage as miss and export partial and zero coverage separately",
            cache_probe="Original-arrival cache snapshot before parent wait, admission or Store load; never substitute d_admit_hit",
            missing_d_sample_policy="Retain short HF responses completed entirely on P; export missing D samples; primary affected share is a lower/upper interval over all returning turns when missing samples exist; never substitute client or ownership timing",
            pairing="Exact source_id, source_turn, arrival, input hash and output token count match",
            reference_scope="One full Serial inf reference per trace, shared across all three methods and three finite links",
            export=[
                "per-request CSV including D-first, client TTFT, reference latency, affected flags and arrival-time cache coverage",
                "affected share and decode miss by link, trace and method",
                "native latency/TPOT/throughput",
                "native timing match counts and source receipts",
            ],
            acceptance="No target affected percentages; measured results determine ordering",
        ),
    )
    write(
        DESIGN / "experiment.json",
        dict(
            name="five-stage-llama7b-h100-sxm80gb-tp8-s0",
            generation_does_not_execute=True,
            execution_status="native_research/runs/five-stage-llama7b-h100-tp8-s0-01/status.json",
            model="llama-7b",
            gpu_display_name="H100 SXM 80GB",
            gpu_catalog_bytes=85899345920,
            hardware=load_profile(DESIGN / "hardware.json").describe(),
            cluster=relative(DESIGN / "cluster.json"),
            seed=0,
            phases=phases,
            total_jobs=len(jobs),
            sanity_jobs=2,
            formal_jobs=20,
            first_delivery=dict(
                after_phase=2,
                cumulative_jobs=10,
                contents=[
                    "Every completed-run CSV",
                    "Affected-share table for 450 GB/s on both traces",
                    "Sanity comparison and native audit receipts",
                ],
            ),
            resources=dict(
                cpu_pool_max=12,
                logical_cpus_per_job=1,
                memory_limit_bytes=8 * 2**30,
                minimum_host_available_bytes=2 * 2**30,
                concurrency="Dynamic memory admission; no execution launched by this design",
            ),
            execution_note="Create and freeze runtime execution plans from phase job records when execution is requested; design JSON files do not launch the current bandwidth runner",
            current_50_100_200_study="Separate running experiment; native named-link latencies differ, so do not relabel existing 50 GB/s outputs as ib400",
        ),
    )
    configure(DESIGN)
    print(DESIGN)


if __name__ == "__main__":
    main()
