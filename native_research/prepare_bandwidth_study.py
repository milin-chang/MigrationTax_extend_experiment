"""Prepare a parent-closed chronological ServeGen subset and four link variants."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.with_name("Simulator-alignment-review")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    source = OLD / "research/results/primary-v1/inputs/servegen-s0.jsonl"
    original = source.read_bytes().splitlines(keepends=True)
    all_rows = [json.loads(line) for line in original]
    blocks = (85899345920 - 4169138176 - 6738149376 * 2) // (64 * 524288)
    safe_tokens = (blocks - math.ceil(0.01 * blocks) - 2) * 64
    excluded = {
        row["conversation"]
        for row in all_rows
        if row["context_tokens"] + row["output_tokens"] - 1 > safe_tokens
    }
    selected = [
        (i, line)
        for i, (line, row) in enumerate(zip(original, all_rows))
        if row["conversation"] not in excluded
    ][:8000]
    payload = [line for _, line in selected]
    subset = ROOT / "inputs/servegen-s0-first8000.jsonl"
    subset.write_bytes(b"".join(payload))
    rows = [json.loads(line) for line in payload]
    write(
        ROOT / "inputs/subset-manifest.json",
        dict(
            source=str(source),
            source_sha256=sha256(source.read_bytes()).hexdigest(),
            subset=subset.name,
            subset_sha256=sha256(subset.read_bytes()).hexdigest(),
            count=len(rows),
            selection="First 8000 requests after excluding entire over-capacity conversations; chronological order; no arrival rescaling or token truncation",
            safe_live_token_limit=safe_tokens,
            excluded_conversations=sorted(excluded),
            excluded_requests_full_trace=sum(
                row["conversation"] in excluded for row in all_rows
            ),
            selected_original_indices=[i for i, _ in selected],
            first_arrival_s=rows[0]["arrival_seconds"],
            last_arrival_s=rows[-1]["arrival_seconds"],
            returning=sum(row["turn"] > 0 for row in rows),
            output_tokens=sum(row["output_tokens"] for row in rows),
            maximum_live_tokens=max(
                row["context_tokens"] + row["output_tokens"] - 1 for row in rows
            ),
        ),
    )
    profile = json.loads(
        (ROOT / "native_methods/profiles/llama3-8b-h100-tp1.json").read_text()
    )
    profile.update(
        model_id="llama-7b",
        model_dtype="fp16",
        kv_dtype="fp16",
        model_parameters=6738149376,
        kv_heads=32,
        block_tokens=64,
    )
    write(ROOT / "configs/hardware-llama7b-h100-tp1-block64.json", profile)
    cluster = dict(
        num_workers=8,
        networks={"node0": "ib_ndr400"},
        worker_groups=[
            dict(role=role, hardware="h100_sxm", num_workers=4, network="node0")
            for role in ("prefill", "decode")
        ],
    )
    write(ROOT / "configs/cluster_4p4d_h100.json", cluster)
    for label, bandwidth in (
        ("bw50", 50e9),
        ("bw100", 100e9),
        ("bw200", 200e9),
        ("reference", 1e15),
    ):
        root = ROOT / "configs" / ("data_" + label)
        shutil.copytree(ROOT / "TokenSim/data", root, dirs_exist_ok=True)
        path = root / "devices/h100_sxm.yaml"
        device = yaml.safe_load(path.read_text())
        memory = deepcopy(device["memory"])
        device["interconnect"]["scale_up_link"] = "bandwidth_study_link"
        device["interconnect"]["host_link"] = "bandwidth_study_link"
        path.write_text(yaml.safe_dump(device, sort_keys=False))
        assert yaml.safe_load(path.read_text())["memory"] == memory
        links_path = root / "topologies/links.yaml"
        links = yaml.safe_load(links_path.read_text())
        # The file's link mapping is nested under "links".
        links["links"]["bandwidth_study_link"] = dict(
            display_name="Controlled communication bandwidth",
            source_id="user-bandwidth-sweep",
            grade="D",
            bandwidth_per_direction_bytes_per_s=bandwidth,
            latency_us=0.0,
            half_bandwidth_bytes=0,
        )
        links_path.write_text(yaml.safe_dump(links, sort_keys=False))
        transfer = dict(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config=dict(
                connectors=[
                    dict(
                        kv_connector="MooncakeConnector",
                        kv_connector_extra_config=dict(
                            protocol="rdma",
                            bandwidth_bytes_per_s=bandwidth,
                            fixed_latency_us=0,
                        ),
                    ),
                    dict(
                        kv_connector="MooncakeStoreConnector",
                        kv_connector_extra_config=dict(
                            mode="embedded",
                            protocol="rdma",
                            local_buffer_size_gb=1,
                            memory_capacity_blocks=10000000,
                            enable_offload=False,
                            replica_num=1,
                            admission_policy="always",
                            eviction_policy="lru",
                            save_policy="mooncake",
                            load_async=False,
                            transfer_overlap=False,
                            bandwidth_bytes_per_s=bandwidth,
                            fixed_latency_us=0,
                        ),
                    ),
                ]
            ),
        )
        write(ROOT / f"configs/transfer_{label}.json", transfer)
        write(
            ROOT / f"configs/study_{label}.json",
            dict(
                label=label,
                bandwidth_bytes_per_s=bandwidth,
                data_root=str(root.relative_to(ROOT)),
                cluster_path="configs/cluster_4p4d_h100.json",
                transfer_path=f"configs/transfer_{label}.json",
                memory_bandwidth_changed=False,
                time_unit="s",
                bandwidth_unit="decimal bytes/s",
                reference=label == "reference",
            ),
        )
    print((ROOT / "inputs/subset-manifest.json").read_text())


if __name__ == "__main__":
    main()
