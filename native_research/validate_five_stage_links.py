"""Validate phase populations, TP topology and native KV service without runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from native_methods.hardware_profile import load_profile
from native_methods.inputs import load_requests
from .aligned_audit import ROOT, read
from .five_stage_order import batches


def validate(design):
    from TokenSim.hardware.context import HardwareContext
    from TokenSim.mooncake.config import parse_mooncake_config
    from TokenSim.mooncake.transfer_engine import TransferEngineSimulator, Segment
    from TokenSim.kv_transfer.connectors.block_store.core import BlockStore
    from TokenSim.research.native_hardware import configure_native_hardware
    from TokenSim.research.native_study import load_study, study_cluster

    experiment = read(design / "experiment.json")
    jobs = read(design / "jobs.json")
    manifests = read(design / "input-manifest.json")
    profile = load_profile(design / "hardware.json")
    if (
        profile.model_id != "llama-7b"
        or profile.tensor_parallel != 8
        or profile.total_gpus != 64
    ):
        raise ValueError("The design must use llama-7b on eight H100 TP8 replicas")
    if profile.gpu_id != "h100_sxm" or profile.gpu_memory_bytes != 85899345920:
        raise ValueError("H100 SXM 80GB catalog geometry differs")
    if len(jobs) != 22 or len({row["id"] for row in jobs}) != 22:
        raise ValueError("Expected 22 unique job IDs")
    if [sum(row["phase"] == n for row in jobs) for n in range(5)] != [2, 4, 4, 6, 6]:
        raise ValueError("Phase populations differ")
    inputs = {}
    for key, manifest in manifests.items():
        path = ROOT / manifest["path"]
        if sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Input hash differs: " + key)
        trace = "mooncake" if key.startswith("mooncake") else "servegen"
        rows = load_requests(path, trace)
        if key in ("mooncake", "servegen"):
            source = Path(manifest["source"])
            source = source if source.is_absolute() else ROOT / source
            if path.read_bytes() != source.read_bytes():
                raise ValueError("Formal input was filtered, clipped or resampled")
        largest = max(row["context_tokens"] + row["output_tokens"] - 1 for row in rows)
        if (
            largest + profile.block_tokens - 1
        ) // profile.block_tokens + 2 > profile.kv_blocks:
            raise ValueError("A full-input request cannot fit the TP8 native capacity")
        if (
            len(rows) != manifest["requests"]
            or sum(row["turn"] > 0 for row in rows) != manifest["returning"]
        ):
            raise ValueError("Input population differs: " + key)
        inputs[key] = dict(
            requests=len(rows),
            returning=manifest["returning"],
            sha256=manifest["sha256"],
            max_live_tokens=largest,
            capacity_filter_applied=False,
        )
    formal_references = {row["trace"]: row for row in jobs if row["reference"]}
    if (
        set(formal_references) != {"mooncake", "servegen"}
        or sum(row["reference"] for row in jobs) != 2
    ):
        raise ValueError("Each trace needs exactly one full Serial reference")
    for job in jobs:
        manifest = manifests["mooncake_sanity" if job["phase"] == 0 else job["trace"]]
        if (
            job["input_sha256"] != manifest["sha256"]
            or job["expected_requests"] != manifest["requests"]
        ):
            raise ValueError("Job input differs")
        if (
            job["phase"] > 0
            and job["reference_output"] != formal_references[job["trace"]]["output"]
        ):
            raise ValueError("Reference is not shared by trace")
        if (ROOT / job["output"]).exists() or (ROOT / job["job"]).exists():
            raise ValueError("An execution output already exists")
    services = []
    for label, expected in (
        ("nvl4", 450e9),
        ("pcie6", 128e9),
        ("ib400", 50e9),
        ("inf", 1e15),
    ):
        study = load_study(design / f"study_{label}.json")
        hardware = HardwareContext.load(study["data_root"])
        _, parallel, _ = configure_native_hardware(profile, hardware)
        cluster = study_cluster(study, parallel)
        if (
            cluster.num_workers != 64
            or len(cluster.worker_groups) != 8
            or any(group.num_workers != 8 for group in cluster.worker_groups)
        ):
            raise ValueError("TP8 rank topology differs")
        device = hardware.device("h100_sxm")
        if (
            device.scale_up_link != "nvlink4"
            or device.memory_bandwidth_bytes_per_s.value != 3352e9
        ):
            raise ValueError("TP collective link or HBM bandwidth changed")
        host = hardware.link(device.host_link)
        if host.bandwidth_per_direction_bytes_per_s != expected:
            raise ValueError("Host-link bandwidth differs")
        transfer = read(Path(study["transfer_path"]))
        children = {
            row["kv_connector"]: row["kv_connector_extra_config"]
            for row in transfer["kv_connector_extra_config"]["connectors"]
        }
        direct = parse_mooncake_config(children["MooncakeConnector"])
        store = parse_mooncake_config(children["MooncakeStoreConnector"])
        engine = TransferEngineSimulator(direct)
        store_owner = SimpleNamespace(host_link=host, options=store)
        for size in (0, 2**20, 2**28):
            native = engine.create_transfer(
                source_segment=Segment("P", "vram"),
                target_segment=Segment("D", "vram"),
                blocks=1,
                bytes_=size,
                topology="cross_node",
            )
            expected_p2p = host.latency_us / 1e6 + size / expected if size else 0
            expected_store = host.latency_us / 1e6 + size / expected
            actual_store = BlockStore._transfer_latency(store_owner, native)
            if (
                abs(native.latency - expected_p2p) > 1e-12
                or abs(actual_store - expected_store) > 1e-12
            ):
                raise ValueError(
                    "Native P/D or Store service differs from the selected link"
                )
            services.append(
                dict(
                    link=label,
                    bytes=size,
                    p2p_service_s=native.latency,
                    store_service_s=actual_store,
                    bandwidth_bytes_per_s=expected,
                )
            )
    if experiment["total_jobs"] != 22:
        raise ValueError("Top-level experiment count differs")
    order = batches(jobs)
    if read(design / "execution-batches.json")["execution_batches"] != order:
        raise ValueError("Configured execution order differs from Mooncake-first order")
    return dict(
        status="passed",
        model="llama-7b",
        gpu="H100 SXM 80GB",
        tp=8,
        physical_gpus=64,
        phases=[2, 4, 4, 6, 6],
        total_jobs=22,
        inputs=inputs,
        native_link_service_checks=services,
        simulations_launched=False,
        execution_priority="mooncake_first",
        execution_batch_counts=[batch["count"] for batch in order],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.design)
    (args.design / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "total_jobs",
                    "tp",
                    "physical_gpus",
                    "simulations_launched",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
