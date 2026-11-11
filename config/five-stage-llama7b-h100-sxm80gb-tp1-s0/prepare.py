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
from native_research.aligned_audit import guards

ROOT = Path('/home/user/uwuowoawa/1_lab_nkcu/Simulator-tokensim-bandwidth')
OLD = ROOT / 'config/five-stage-llama7b-h100-sxm80gb-tp8-s0'
DESIGN = ROOT / 'config/five-stage-llama7b-h100-sxm80gb-tp1-s0'
OLD_RUN = 'native_research/runs/five-stage-llama7b-h100-tp8-s0-01'
RUN = 'native_research/runs/five-stage-llama7b-h100-tp1-s0-01'
REPORT = 'native_research/reports/five-stage-llama7b-h100-tp1-s0-01'

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def read(path):
    return json.loads(path.read_text())

def relative(path):
    return str(path.relative_to(ROOT))

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def remap(value):
    if isinstance(value, str):
        return value.replace(relative(OLD), relative(DESIGN)).replace(OLD_RUN, RUN)
    if isinstance(value, list):
        return [remap(item) for item in value]
    if isinstance(value, dict):
        return {key: remap(item) for key, item in value.items()}
    return value

if DESIGN.exists() or (ROOT / RUN).exists() or (ROOT / REPORT).exists():
    raise FileExistsError('Refusing to replace a prepared or started TP1 study')
DESIGN.mkdir(parents=True)
shutil.copy2(__file__, DESIGN / 'prepare.py')
hardware = read(OLD / 'hardware.json')
hardware.update(gpus_per_instance=1, tensor_parallel=1)
write(DESIGN / 'hardware.json', hardware)
profile = load_profile(DESIGN / 'hardware.json')
assert profile.tensor_parallel == 1 and profile.total_gpus == 8
cluster = read(OLD / 'cluster.json')
cluster['num_workers'] = 8
for group in cluster['worker_groups']:
    group['num_workers'] = 1
write(DESIGN / 'cluster.json', cluster)
shutil.copy2(OLD / 'metrics.json', DESIGN / 'metrics.json')
write(DESIGN / 'links.json', remap(read(OLD / 'links.json')))
for label in ['nvl4', 'pcie6', 'ib400', 'inf']:
    shutil.copytree(OLD / f'data_{label}', DESIGN / f'data_{label}')
    shutil.copy2(OLD / f'transfer_{label}.json', DESIGN / f'transfer_{label}.json')
    study = remap(read(OLD / f'study_{label}.json'))
    study.update(bandwidth_scope='Full model KV per TP1 worker endpoint',
        input_selection='Full Mooncake shifted-v1 seed-0 input, byte-identical to TP8; no clipping, filtering or arrival rescaling',
        tp_collective_applicable=False)
    write(DESIGN / f'study_{label}.json', study)
source_input = OLD / 'inputs/mooncake-s0.jsonl'
target_input = DESIGN / 'inputs/mooncake-s0.jsonl'
target_input.parent.mkdir()
shutil.copy2(source_input, target_input)
rows = load_requests(target_input, 'mooncake')
assert len(rows) == 12031 and sum(r['turn'] > 0 for r in rows) == 3959
assert sum(r['output_tokens'] for r in rows) == 4122048
assert digest(source_input) == digest(target_input)
maximum_live = max(r['context_tokens'] + r['output_tokens'] - 1 for r in rows)
required_blocks = max((r['context_tokens'] + r['output_tokens'] - 1 + profile.block_tokens - 1) // profile.block_tokens + 2 for r in rows)
assert required_blocks <= profile.kv_blocks
manifest = deepcopy(read(OLD / 'input-manifest.json')['mooncake'])
manifest.update(path=relative(target_input), copied_from=relative(source_input), sha256=digest(target_input))
write(DESIGN / 'input-manifest.json', {'mooncake': manifest})
jobs = []
for old_job in read(OLD / 'jobs.json'):
    if old_job['trace'] != 'mooncake' or old_job['phase'] not in (1, 3, 4):
        continue
    job = remap(old_job)
    job['id'] = 'tp1-' + job['id']
    job['worker_argv'][0] = sys.executable
    job['memory_estimate_bytes'] = 2 * 2**30
    assert not (ROOT / job['output']).exists() and not (ROOT / job['job']).exists()
    jobs.append(job)
assert len(jobs) == 10 and sum(job['reference'] for job in jobs) == 1
assert [sum(job['phase'] == phase for job in jobs) for phase in (1, 3, 4)] == [4, 3, 3]
write(DESIGN / 'jobs.json', jobs)
resources = dict(planned_cpu_pool=sorted(os.sched_getaffinity(0))[:12],
                 logical_cpus_per_job=1, minimum_host_available_bytes=2 * 2**30,
                 memory_limit_bytes=8 * 2**30,
                 admission_estimates='2 GiB per Mooncake condition, dynamic growth reserve; one CPU per job; 8 GiB cap; retain 2 GiB host reserve')
batches = []
for phase, name in [(1, '01-mooncake-fast'), (3, '03-mooncake-pcie6'), (4, '04-mooncake-ib400')]:
    selected = [job for job in jobs if job['phase'] == phase]
    write(DESIGN / f'phases/{name}.json', dict(phase=phase, jobs=selected))
    batches.append(dict(id=name, phase=phase, count=len(selected),
        plan=relative(DESIGN / 'execution' / (name + '.plan.json'))))
write(DESIGN / 'experiment.json', dict(
    name='five-stage-llama7b-h100-tp1-s0-01', hardware=profile.describe(), seed=0,
    trace='mooncake', input_version='shifted v1', selected_phases=[1, 3, 4],
    total_jobs=10, sanity_jobs=0, servegen_jobs=0, resources=resources,
    comparison='TP1: 8 physical GPUs; historical TP8: 64 physical GPUs; different GPU budget',
    reference='One full Serial inf baseline shared by all nine finite-link methods',
    run_root=RUN, report_dir=REPORT, execute_benchmark_for_estimation=False,
))
# This is static configuration validation, not a diagnostic simulation.
from TokenSim.hardware.context import HardwareContext
from TokenSim.research.native_hardware import configure_native_hardware
from TokenSim.research.native_study import load_study, study_cluster
for label, bandwidth in [('nvl4',450e9),('pcie6',128e9),('ib400',50e9),('inf',1e15)]:
    study=load_study(DESIGN / f'study_{label}.json')
    context=HardwareContext.load(study['data_root'])
    _, parallel, _ = configure_native_hardware(profile, context)
    actual_cluster=study_cluster(study,parallel)
    assert actual_cluster.num_workers == 8
    assert len(actual_cluster.worker_groups) == 8
    assert all(group.num_workers == 1 for group in actual_cluster.worker_groups)
    assert context.device('h100_sxm').memory_bandwidth_bytes_per_s.value == 3352e9
    assert study['bandwidth_bytes_per_s'] == bandwidth
write(DESIGN / 'validation.json', dict(status='passed', simulations_launched=False,
    total_jobs=10, phase_counts={'1':4,'3':3,'4':3}, tensor_parallel=1, total_gpus=8,
    requests=len(rows), returning=3959, output_tokens=4122048,
    input_sha256=digest(target_input), byte_identical_to_tp8=True,
    max_live_tokens=maximum_live, required_blocks_with_margin=required_blocks,
    capacity_blocks_per_worker=profile.kv_blocks,
    capacity_tokens_per_worker=profile.kv_blocks*profile.block_tokens,
    capacity_filter_applied=False, unchanged_memory_bandwidth_bytes_per_s=3352e9,
    microbenchmark_executed=False,
))
(DESIGN / 'SPECIFICATIONS.md').write_text('''# TokenSim／Mooncake：TP1 連線實驗

沿用 five-stage-llama7b-h100-tp8-s0-01 的完整 Mooncake shifted v1 seed 0，只執行階段 1、3、4；不執行 sanity、ServeGen 或估時微型模擬。

| 項目 | 設定 |
|---|---|
| 模型／精度 | llama-7b／FP16，32 層 |
| GPU | H100 SXM 80GB，catalog 80 GiB |
| P／D | 4P＋4D，每個 worker 1 張，共 8 張 GPU |
| TP／PP | TP1／PP1 |
| 輸入 | 12,031 requests、3,959 returning turns、4,122,048 output tokens；與 TP8 輸入逐 byte 相同 |
| KV block／batching | 64 tokens；paged-attn；max_num_seqs＝64；max_num_batched_tokens＝8,192 |
| 單 worker KV 容量 | 2,034 blocks＝130,176 tokens |
| 最大請求 live tokens | 126,526；全部請求可容納，不需刪減 |
| 連線 | nvl4＝450 GB/s；pcie6＝128 GB/s；ib400＝50 GB/s；inf＝1e15 bytes/s |
| 通訊設定 | 沿用 TP8 各連線的 Store／P→D 頻寬與固定延遲；不改 HBM |
| HBM | 固定 3,352 GB/s |
| 原生計時 | FP16 operator analytical fallback；TP1 無跨 rank 的 TP collective |
| CPU／記憶體 | 最多 12 邏輯 CPU；每工作 1 CPU、上限 8 GiB；動態並行並保留至少 2 GiB host 空間 |

| 階段 | 條件 | 組數 |
|---:|---|---:|
| 1 | Serial inf reference＋nvl4 下 Serial／PPD／MigrationTax | 4 |
| 3 | pcie6 下 Serial／PPD／MigrationTax | 3 |
| 4 | ib400 下 Serial／PPD／MigrationTax | 3 |

依階段順序執行，階段內由現有 resource-admitted controller 動態安排並行。共 10 組，不是 16 組。TP1 共 8 GPU，原 TP8 共 64 GPU；兩者為不同 GPU 預算的部署比較。

完成後自動輸出 results.md、results.json、native-metrics.csv、comparison.csv、csv-index.json 與十組逐請求 CSV；保留原始 audit、job receipts 與 source-snapshot。

啟動：

```bash
python -m native_research.run_mooncake_link_study \\
  --design config/five-stage-llama7b-h100-sxm80gb-tp1-s0 \\
  --run-root native_research/runs/five-stage-llama7b-h100-tp1-s0-01 \\
  --report-dir native_research/reports/five-stage-llama7b-h100-tp1-s0-01
```

既有計畫、工作 ID 與輸出不覆寫。原 TP8 結果不修改。
''')
paths = []
for name in ['algorithm','native_methods','migrationtax/reference','TokenSim/TokenSim']:
    paths.extend((ROOT / name).rglob('*.py'))
paths += [path for path in DESIGN.rglob('*') if path.is_file()]
paths += [ROOT/'Trace/sources.json', ROOT/'native_research/run_mooncake_link_study.py']
paths += [ROOT/f'native_research/{name}.py' for name in
          ['run_bandwidth','named_job','aligned_audit','show_jobs','report_bandwidth','report_five_stage_links']]
locks={relative(path):digest(path) for path in sorted(set(paths)) if '__pycache__' not in path.parts}
for batch in batches:
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(), phase=batch['phase'],
        batch=batch['id'], hardware_alignment=profile.describe(), source_sha256=locks,
        inputs={'mooncake':manifest}, jobs=[job for job in jobs if job['phase']==batch['phase']],
        resources=resources)
    guards(plan)
    write(ROOT / batch['plan'], plan)
write(DESIGN / 'execution/index.json',dict(batches=batches,total_jobs=10))
print(json.dumps({'design':relative(DESIGN),'run_root':RUN,'jobs':len(jobs),
                  'requests_per_condition':len(rows),'total_gpus':profile.total_gpus,
                  'validation':'passed','microbenchmark_executed':False}))
