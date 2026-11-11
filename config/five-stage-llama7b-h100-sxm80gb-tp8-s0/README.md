# Five-stage H100 link study

The complete specification is in [SPECIFICATIONS.md](SPECIFICATIONS.md).
The immutable JSON configurations define 22 jobs in five phases with counts 2, 4, 4, 6 and 6.
The model is llama-7b FP16 on four prefill and four decode H100 SXM 80GB TP8 replicas, totaling 64 simulated GPUs.
The configuration generator does not launch jobs.
Execution is recorded separately in native_research/runs/five-stage-llama7b-h100-tp8-s0-01.

## Validation and plan freezing

```bash
python -m native_research.validate_five_stage_links \
  --design config/five-stage-llama7b-h100-sxm80gb-tp8-s0
python -m native_research.build_five_stage_plans \
  --design config/five-stage-llama7b-h100-sxm80gb-tp8-s0
```

The validation checks the selected H100 catalog, 64-rank topology, exact trace parents and native P/D and Store transfer service.
Plan generation refuses to overwrite existing frozen execution plans.

## Sequential execution

```bash
python -m native_research.run_five_stage_links \
  --design config/five-stage-llama7b-h100-sxm80gb-tp8-s0 \
  --run-root native_research/runs/five-stage-llama7b-h100-tp8-s0-01 \
  --report-dir native_research/reports/five-stage-llama7b-h100-tp8-s0-01
python3 native_research/show_five_stage_jobs.py
```

Execution uses the existing Python 3.12 native dependency environment.
Each phase uses the existing resource-admitted controller and audits every completed native workload.
Stage 0 must pass its paired sanity check before the following stages begin.
The execution order completes all 12 Mooncake jobs, including sanity, before any ServeGen formal job.
Mooncake completion creates the first delivery snapshot with all 12 completed-run CSV files.
ServeGen fast-link completion creates a second snapshot with 16 completed-run CSV files.
The final report contains 22 CSV files, including the two sanity runs and two full references.
