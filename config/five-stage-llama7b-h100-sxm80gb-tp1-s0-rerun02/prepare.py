"""Clone the completed TP1 study into a separate, checksum-locked repeat."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

from native_methods.hardware_profile import load_profile
from native_research.aligned_audit import guards

ROOT = Path('/home/user/uwuowoawa/1_lab_nkcu/Simulator-tokensim-bandwidth')
OLD = ROOT / 'config/five-stage-llama7b-h100-sxm80gb-tp1-s0'
DESIGN = ROOT / 'config/five-stage-llama7b-h100-sxm80gb-tp1-s0-rerun02'
BASE_RUN = 'native_research/runs/five-stage-llama7b-h100-tp1-s0-01'
RUN = 'native_research/runs/five-stage-llama7b-h100-tp1-s0-02'
BASE_REPORT = 'native_research/reports/five-stage-llama7b-h100-tp1-s0-01'
REPORT = 'native_research/reports/five-stage-llama7b-h100-tp1-s0-02'
AFTER = 'native_research/runs/five-stage-llama7b-h100-tp4-s0-01'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def relative(path):
    return str(path.relative_to(ROOT))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

if DESIGN.exists() or (ROOT / RUN).exists() or (ROOT / REPORT).exists():
    raise FileExistsError('Refusing to replace a prepared or started repeat')
original_jobs = read(OLD / 'jobs.json')
ids = {job['id']:job['id'].replace('tp1-', 'tp1-rerun02-', 1) for job in original_jobs}

def remap(value):
    if isinstance(value, str):
        return ids.get(value, value.replace(relative(OLD), relative(DESIGN)).replace(BASE_RUN, RUN).replace(BASE_REPORT, REPORT))
    if isinstance(value, list):
        return [remap(item) for item in value]
    if isinstance(value, dict):
        return {key:remap(item) for key,item in value.items()}
    return value

shutil.copytree(OLD, DESIGN, ignore=shutil.ignore_patterns('execution', 'prepare.py', '__pycache__'))
for path in DESIGN.rglob('*.json'):
    value = read(path)
    updated = remap(value)
    if updated != value:
        write(path, updated)
shutil.copy2(__file__, DESIGN / 'prepare.py')
for path in DESIGN.glob('*.md'):
    path.write_text(remap(path.read_text()))
profile = load_profile(DESIGN / 'hardware.json')
assert profile.tensor_parallel == 1 and profile.total_gpus == 8
assert digest(OLD / 'hardware.json') == digest(DESIGN / 'hardware.json')
assert digest(OLD / 'cluster.json') == digest(DESIGN / 'cluster.json')
assert digest(OLD / 'inputs/mooncake-s0.jsonl') == digest(DESIGN / 'inputs/mooncake-s0.jsonl')
for label in ('inf', 'nvl4', 'pcie6', 'ib400'):
    assert digest(OLD / f'transfer_{label}.json') == digest(DESIGN / f'transfer_{label}.json')
    for path in (OLD / f'data_{label}').rglob('*'):
        if path.is_file():
            assert digest(path) == digest(DESIGN / path.relative_to(OLD))
    old_study, study = read(OLD / f'study_{label}.json'), read(DESIGN / f'study_{label}.json')
    assert remap(old_study) == study
jobs = read(DESIGN / 'jobs.json')
assert len(jobs) == 10 and len({job['id'] for job in jobs}) == 10
assert [sum(job['phase'] == n for job in jobs) for n in (1, 3, 4)] == [4, 3, 3]
code_checks = []
for old_job, job in zip(original_jobs, jobs):
    job['worker_argv'][0] = sys.executable
    assert job['seed'] == 0 and job['expected_requests'] == 12031
    assert job['method'] == old_job['method'] and job['input_sha256'] == old_job['input_sha256']
    assert not (ROOT / job['output']).exists() and not (ROOT / job['job']).exists()
    condition = read(ROOT / old_job['output'] / 'condition.json')
    unchanged = []
    for name, expected in condition['imported_source_sha256'].items():
        if name.endswith('.py'):
            assert digest(ROOT / name) == expected, name
            unchanged.append(name)
    code_checks.append(dict(baseline=old_job['output'], common_imported_python_files=len(unchanged), all_unchanged=True))
write(DESIGN / 'jobs.json', jobs)
for path in (DESIGN / 'phases').glob('*.json'):
    phase = read(path)
    phase['jobs'] = [job for job in jobs if job['phase'] == phase['phase']]
    write(path, phase)
experiment = read(DESIGN / 'experiment.json')
experiment.update(name=Path(RUN).name, run_root=RUN, report_dir=REPORT, rerun_of=BASE_RUN,
    after_run=AFTER, repeat_controls='Same seed, input bytes, physical hardware, catalogs, links, native adapter, admission and timing source; new output paths and job IDs only')
write(DESIGN / 'experiment.json', experiment)
validation = read(DESIGN / 'validation.json')
validation.update(rerun_of=BASE_RUN, baseline_configuration_identical=True,
    identical_imported_python_checks=code_checks, simulations_launched=False,
    orchestration_note='Current coordinator includes queue support and dynamic report labels; native worker timing and admission sources match baseline snapshots')
write(DESIGN / 'validation.json', validation)
with (DESIGN / 'SPECIFICATIONS.md').open('a') as stream:
    stream.write('\n## 本次重跑\n\n原 TP1 的 seed、Trace、模型、GPU、KV 容量、頻寬、延遲與實際載入的 Python 來源均已逐檔核對相同。只更換工作 ID、設定與輸出路徑；本次等待 TP4 完成後才啟動。\n\n')
    stream.write(f'原始對照：`{BASE_RUN}`；新紀錄：`{RUN}`；新報告：`{REPORT}`。\n\n啟動時加上 `--after-run {AFTER}`。\n')
paths = []
for name in ('algorithm', 'native_methods', 'migrationtax/reference', 'TokenSim/TokenSim'):
    paths.extend((ROOT / name).rglob('*.py'))
paths += [p for p in DESIGN.rglob('*') if p.is_file()]
paths += [ROOT / 'Trace/sources.json']
paths += [ROOT / f'native_research/{name}.py' for name in
    ('run_mooncake_link_study', 'run_bandwidth', 'named_job', 'aligned_audit', 'show_jobs', 'report_bandwidth', 'report_five_stage_links')]
locks = {relative(path):digest(path) for path in sorted(set(paths)) if '__pycache__' not in path.parts}
old_index = read(OLD / 'execution/index.json')
batches = remap(old_index['batches'])
for batch in batches:
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase=batch['phase'], batch=batch['id'],
        hardware_alignment=profile.describe(), source_sha256=locks,
        inputs=read(DESIGN / 'input-manifest.json'), jobs=[j for j in jobs if j['phase'] == batch['phase']],
        resources=experiment['resources'])
    guards(plan)
    write(ROOT / batch['plan'], plan)
write(DESIGN / 'execution/index.json', dict(batches=batches, total_jobs=10))
print(json.dumps(dict(design=relative(DESIGN), run_root=RUN, report_dir=REPORT, jobs=10,
    identical_baseline_native_sources=True, validation='passed', wait_for=AFTER)))
