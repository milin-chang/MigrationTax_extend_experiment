"""Prepare an unchanged 200-request prefix and isolated TP8 link catalogs."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import yaml

from native_methods.inputs import load_requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/servegen7b-tp8-200"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    if CONFIG.exists():
        raise FileExistsError("Refusing to overwrite prepared TP8 catalogs")
    source = ROOT / "inputs/servegen-s0-first8000.jsonl"
    subset = ROOT / "inputs/servegen-s0-first200-tp8.jsonl"
    if subset.exists():
        raise FileExistsError("Refusing to overwrite the TP8 subset")
    subset.write_bytes(b"".join(source.read_bytes().splitlines(keepends=True)[:200]))
    rows = load_requests(subset, "servegen")
    assert len(rows) == 200
    write(
        ROOT / "inputs/servegen-s0-first200-tp8.manifest.json",
        dict(
            source=str(source.relative_to(ROOT)),
            source_sha256=sha256(source.read_bytes()).hexdigest(),
            source_manifest="inputs/subset-manifest.json",
            subset=str(subset.relative_to(ROOT)),
            subset_sha256=sha256(subset.read_bytes()).hexdigest(),
            selection="First 200 chronological rows of the existing capacity-filtered 8000-row ServeGen seed-0 subset; no additional exclusions, token clipping or arrival rescaling; explicit parents preserved",
            requests=len(rows),
            returning=sum(row["turn"] > 0 for row in rows),
            conversations=len({row["conversation"] for row in rows}),
            output_tokens=sum(row["output_tokens"] for row in rows),
            first_arrival_s=rows[0]["arrival_seconds"],
            last_arrival_s=rows[-1]["arrival_seconds"],
            max_live_tokens=max(
                row["context_tokens"] + row["output_tokens"] - 1 for row in rows
            ),
        ),
    )
    profile = json.loads(
        (ROOT / "configs/hardware-llama7b-h100-tp1-block64.json").read_text()
    )
    profile.update(gpus_per_instance=8, tensor_parallel=8)
    write(CONFIG / "hardware.json", profile)
    networks, groups = {}, []
    for role in ("prefill", "decode"):
        for index in range(4):
            node = f"{role}-{index}"
            networks[node] = "ethernet800Gb"
            groups.append(
                dict(
                    role=role,
                    hardware="h100_sxm",
                    num_workers=8,
                    network=node,
                    name=node,
                )
            )
    write(
        CONFIG / "cluster.json",
        dict(num_workers=64, networks=networks, worker_groups=groups),
    )
    for label, bandwidth in (
        ("reference", 1e15),
        ("bw50", 50e9),
        ("bw100", 100e9),
        ("bw200", 200e9),
    ):
        catalog = CONFIG / ("data_" + label)
        shutil.copytree(ROOT / "configs" / ("data_" + label), catalog)
        path = catalog / "devices/h100_sxm.yaml"
        device = yaml.safe_load(path.read_text())
        before = deepcopy(device)
        device["interconnect"]["scale_up_link"] = "nvlink4"
        device["interconnect"]["host_link"] = "bandwidth_study_link"
        path.write_text(yaml.safe_dump(device, sort_keys=False))
        assert device["memory"] == before["memory"]
        links = yaml.safe_load((catalog / "topologies/links.yaml").read_text())["links"]
        assert float(links["nvlink4"]["bandwidth_per_direction_bytes_per_s"]) == 450e9
        assert (
            float(links["bandwidth_study_link"]["bandwidth_per_direction_bytes_per_s"])
            == bandwidth
        )
        shutil.copyfile(
            ROOT / f"configs/transfer_{label}.json", CONFIG / f"transfer_{label}.json"
        )
        write(
            CONFIG / f"study_{label}.json",
            dict(
                label=label,
                bandwidth_bytes_per_s=bandwidth,
                bandwidth_scope="Aggregate full-TP KV bundle per replica endpoint; not per GPU rank",
                tp_collective_link="nvlink4",
                tp_collective_bandwidth_bytes_per_s=450e9,
                data_root=str(catalog.relative_to(ROOT)),
                cluster_path=str((CONFIG / "cluster.json").relative_to(ROOT)),
                transfer_path=str(
                    (CONFIG / f"transfer_{label}.json").relative_to(ROOT)
                ),
                input_manifest="inputs/servegen-s0-first200-tp8.manifest.json",
                input_selection="同一份 ServeGen seed 0 容量篩選子集的前 200 筆；未再排除請求、截短 token 或重縮到達時間；保留前輪相依。",
                memory_bandwidth_changed=False,
                time_unit="s",
                bandwidth_unit="decimal bytes/s",
                reference=label == "reference",
            ),
        )
    print(CONFIG)


if __name__ == "__main__":
    main()
