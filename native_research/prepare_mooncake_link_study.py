"""Prepare independent TP2/TP4 Mooncake studies without launching simulations."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

from native_methods.hardware_profile import load_profile
from native_methods.inputs import load_requests
from .aligned_audit import ROOT, guards, read

BASE = ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp8-s0"
BASE_RUN = "native_research/runs/five-stage-llama7b-h100-tp8-s0-01"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def relative(path):
    return str(path.relative_to(ROOT))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(tp):
    design = ROOT / f"config/five-stage-llama7b-h100-sxm80gb-tp{tp}-s0"
    run = f"native_research/runs/five-stage-llama7b-h100-tp{tp}-s0-01"
    report = f"native_research/reports/five-stage-llama7b-h100-tp{tp}-s0-01"
    if any(path.exists() for path in (design, ROOT / run, ROOT / report)):
        raise FileExistsError("Refusing to replace an existing prepared or started study")

    def remap(value):
        if isinstance(value, str):
            return value.replace(relative(BASE), relative(design)).replace(BASE_RUN, run)
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    design.mkdir(parents=True)
    hardware = read(BASE / "hardware.json")
    hardware.update(gpus_per_instance=tp, tensor_parallel=tp)
    write(design / "hardware.json", hardware)
    profile = load_profile(design / "hardware.json")
    assert profile.tensor_parallel == tp and profile.total_gpus == 8 * tp
    cluster = read(BASE / "cluster.json")
    cluster["num_workers"] = 8 * tp
    for group in cluster["worker_groups"]:
        group["num_workers"] = tp
    write(design / "cluster.json", cluster)
    shutil.copy2(BASE / "metrics.json", design / "metrics.json")
    write(design / "links.json", remap(read(BASE / "links.json")))
    for label in ("nvl4", "pcie6", "ib400", "inf"):
        shutil.copytree(BASE / f"data_{label}", design / f"data_{label}")
        shutil.copy2(BASE / f"transfer_{label}.json", design / f"transfer_{label}.json")
        study = remap(read(BASE / f"study_{label}.json"))
        study.update(bandwidth_scope="Full model KV bundle per TP group endpoint; not per rank",
            input_selection="Full Mooncake shifted-v1 seed-0 input, byte-identical to TP1/TP8; no filtering, clipping or arrival rescaling",
            tp_collective_applicable=True)
        write(design / f"study_{label}.json", study)
    source_input = BASE / "inputs/mooncake-s0.jsonl"
    target_input = design / "inputs/mooncake-s0.jsonl"
    target_input.parent.mkdir()
    shutil.copy2(source_input, target_input)
    rows = load_requests(target_input, "mooncake")
    assert len(rows) == 12031 and sum(r["turn"] > 0 for r in rows) == 3959
    assert sum(r["output_tokens"] for r in rows) == 4122048
    assert digest(target_input) == digest(source_input)
    largest = max(r["context_tokens"] + r["output_tokens"] - 1 for r in rows)
    required = (largest + profile.block_tokens - 1) // profile.block_tokens + 2
    assert required <= profile.kv_blocks
    manifest = deepcopy(read(BASE / "input-manifest.json")["mooncake"])
    manifest.update(path=relative(target_input), copied_from=relative(source_input), sha256=digest(target_input))
    write(design / "input-manifest.json", {"mooncake": manifest})
    jobs = []
    for old in read(BASE / "jobs.json"):
        if old["trace"] != "mooncake" or old["phase"] not in (1, 3, 4):
            continue
        job = remap(old)
        job["id"] = f"tp{tp}-" + job["id"]
        job["worker_argv"][0] = sys.executable
        job["memory_estimate_bytes"] = 2 * 2**30
        assert not (ROOT / job["output"]).exists() and not (ROOT / job["job"]).exists()
        jobs.append(job)
    assert len(jobs) == len({job["id"] for job in jobs}) == 10
    assert sum(job["reference"] for job in jobs) == 1
    assert [sum(job["phase"] == n for job in jobs) for n in (1, 3, 4)] == [4, 3, 3]
    write(design / "jobs.json", jobs)
    resources = dict(planned_cpu_pool=sorted(os.sched_getaffinity(0))[:12],
        logical_cpus_per_job=1, minimum_host_available_bytes=2 * 2**30,
        memory_limit_bytes=8 * 2**30,
        admission_estimates="2 GiB per Mooncake job with sampled growth reserve; one active TP study at a time; 1 CPU/job, 8 GiB limit, retain 2 GiB host memory")
    batches = []
    for phase, name in ((1, "01-mooncake-fast"), (3, "03-mooncake-pcie6"), (4, "04-mooncake-ib400")):
        write(design / f"phases/{name}.json", dict(phase=phase, jobs=[j for j in jobs if j["phase"] == phase]))
        batches.append(dict(id=name, phase=phase, count=sum(j["phase"] == phase for j in jobs),
                            plan=relative(design / "execution" / (name + ".plan.json"))))
    write(design / "experiment.json", dict(name=Path(run).name, hardware=profile.describe(),
        seed=0, trace="mooncake", input_version="shifted v1", selected_phases=[1, 3, 4],
        total_jobs=10, sanity_jobs=0, servegen_jobs=0, resources=resources,
        comparison="Fixed 4P+4D model groups, varying physical GPU total with TP; not equal-GPU-budget experiments",
        reference="One full Serial inf baseline per TP, shared by its nine finite-link conditions",
        run_root=run, report_dir=report, execute_benchmark_for_estimation=False))
    from TokenSim.hardware.context import HardwareContext
    from TokenSim.research.native_hardware import configure_native_hardware
    from TokenSim.research.native_study import load_study, study_cluster
    for label, bandwidth in (("nvl4", 450e9), ("pcie6", 128e9), ("ib400", 50e9), ("inf", 1e15)):
        study = load_study(design / f"study_{label}.json")
        context = HardwareContext.load(study["data_root"])
        _, parallel, _ = configure_native_hardware(profile, context)
        actual_cluster = study_cluster(study, parallel)
        assert actual_cluster.num_workers == 8 * tp
        assert len(actual_cluster.worker_groups) == 8
        assert all(group.num_workers == tp for group in actual_cluster.worker_groups)
        assert context.device("h100_sxm").memory_bandwidth_bytes_per_s.value == 3352e9
        assert study["bandwidth_bytes_per_s"] == bandwidth
    write(design / "validation.json", dict(status="passed", simulations_launched=False,
        total_jobs=10, phase_counts={"1": 4, "3": 3, "4": 3}, tensor_parallel=tp, total_gpus=8 * tp,
        requests=len(rows), returning=3959, output_tokens=4122048, input_sha256=digest(target_input),
        byte_identical_to_tp8=True, max_live_tokens=largest, required_blocks_with_margin=required,
        capacity_blocks_per_tp_group=profile.kv_blocks, capacity_tokens_per_tp_group=profile.kv_blocks * profile.block_tokens,
        capacity_filter_applied=False, unchanged_memory_bandwidth_bytes_per_s=3352e9, microbenchmark_executed=False))
    (design / "SPECIFICATIONS.md").write_text(f"""# TokenSim／Mooncake：TP{tp} 連線實驗

| 項目 | 規格 |
|---|---|
| 模型／精度 | llama-7b／FP16，32 層 |
| GPU | H100 SXM 80GB；catalog 80 GiB／GPU |
| P／D | 4P＋4D，每群組 {tp} GPU；共 {8 * tp} 張模擬 GPU |
| 平行方式 | TP{tp}／PP1 |
| Trace | Mooncake shifted v1、seed 0、全量 12,031 筆；與 TP1／TP8 輸入逐 byte 相同 |
| 輸出／returning | 4,122,048 output tokens；3,959 returning turns |
| KV block／容量 | 64 tokens；{profile.kv_blocks:,} blocks＝{profile.kv_blocks * profile.block_tokens:,} tokens／完整 TP 群組 |
| Batching | paged-attn；max_num_seqs＝64；max_num_batched_tokens＝8,192 |
| 通訊連線 | nvl4＝450 GB/s；pcie6＝128 GB/s；ib400＝50 GB/s；inf＝1e15 bytes/s |
| HBM／群組內連線 | HBM 固定 3,352 GB/s；TP collective 使用固定 NVLink |
| KV 頻寬計量 | 完整 TP 群組端點，不額外乘上 TP rank 數 |
| 計時 | 原生 FP16 operator analytical fallback＋原生 TP collective／connector 計時 |
| CPU／記憶體 | 最多 12 邏輯 CPU；每工作 1 CPU、上限 8 GiB；動態並行並保留至少 2 GiB host 空間 |

| 階段 | 條件 | 組數 |
|---:|---|---:|
| 1 | Serial inf reference＋nvl4 下 Serial／PPD／MigrationTax | 4 |
| 3 | pcie6 下 Serial／PPD／MigrationTax | 3 |
| 4 | ib400 下 Serial／PPD／MigrationTax | 3 |

共 10 組；不執行 sanity、ServeGen 或估時微型模擬。每個 TP 使用自己的 Serial inf 參考。
TP2 與 TP4 分成兩批獨立結果；TP4 等 TP2 成功完成後自動開始，避免兩個控制器同時預留記憶體。

啟動指令：

```bash
python -m native_research.run_mooncake_link_study \\
  --design {relative(design)} \\
  --run-root {run} \\
  --report-dir {report}
```

TP4 排隊時另加 `--after-run native_research/runs/five-stage-llama7b-h100-tp2-s0-01`。
結果自動輸出 Markdown 與逐請求 CSV；原始 log、程序收據、audit 與 source-snapshot 分開保留。
既有計畫與輸出不覆寫。各 TP 的 GPU 總數不同，屬不同資源量的部署比較。
""")
    paths = []
    for name in ("algorithm", "native_methods", "migrationtax/reference", "TokenSim/TokenSim"):
        paths.extend((ROOT / name).rglob("*.py"))
    paths += [p for p in design.rglob("*") if p.is_file()]
    paths += [ROOT / "Trace/sources.json", Path(__file__)]
    paths += [ROOT / f"native_research/{name}.py" for name in
        ("run_mooncake_link_study", "run_bandwidth", "named_job", "aligned_audit", "show_jobs", "report_bandwidth", "report_five_stage_links")]
    locks = {relative(path): digest(path) for path in sorted(set(paths)) if "__pycache__" not in path.parts}
    for batch in batches:
        plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase=batch["phase"],
            batch=batch["id"], hardware_alignment=profile.describe(), source_sha256=locks,
            inputs={"mooncake": manifest}, jobs=[j for j in jobs if j["phase"] == batch["phase"]], resources=resources)
        guards(plan)
        write(ROOT / batch["plan"], plan)
    write(design / "execution/index.json", dict(batches=batches, total_jobs=10))
    return dict(design=relative(design), run_root=run, report_dir=report,
                tensor_parallel=tp, total_gpus=8 * tp, jobs=10, validation="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.tp)))


if __name__ == "__main__":
    main()
