# ServeGen 200-request TP8 bandwidth experiment

This experiment replays the chronological 200-row prefix of the existing capacity-filtered ServeGen seed-0 input.
All ten conditions preserve the same source bytes, arrival timestamps and explicit previous-turn dependencies.
The input contains 120 conversations, 80 returning turns and 137,909 output tokens.

| Setting | Value |
|---|---|
| Model and precision | `llama-7b`, FP16 weights and KV, 32 layers |
| Hardware | H100 SXM, 80 GiB catalog capacity per rank |
| Replicas | Four prefill and four decode replicas, eight GPU ranks per replica |
| Parallelism | TP8, PP1; eight independently scheduled replicas |
| Physical inventory | 64 simulated GPUs |
| KV blocks | 64 tokens per block, 19,084 blocks per rank |
| KV token capacity | 1,221,376 tokens per TP replica; ranks store shards of the same tokens |
| Batching | Paged attention, 64 sequences and 8,192 tokens per batch |
| HBM bandwidth | Unchanged at 3,352 GB/s per GPU |
| TP communication | Fixed `nvlink4`, 450 GB/s per direction before native collective efficiency and launch costs |
| KV and Store communication | 50, 100 or 200 decimal GB/s per replica endpoint |
| Reference | Serial with `1e15 bytes/s` KV and Store service; TP communication remains fixed |
| Methods | Serial, adapted PPD, MigrationTax |
| Resources | One CPU per job, 8 GiB per-job limit, host reserve of at least 2 GiB |

The existing symmetric TP implementation transfers the complete KV bundle over one aggregate replica endpoint.
The swept bandwidth is therefore not multiplied by the eight GPU ranks.
The group topology uses eight named network domains with eight ranks each; it does not describe a single 64-GPU NVLink node.
The legacy network label is `ethernet800Gb`; explicit KV connector and embedded host-link settings control the swept services.
The Store remains a symbolic shared pool of 10,000,000 aggregate KV blocks with a 1 GiB staging buffer per active replica endpoint.
This capacity is not host process RAM.

The fixed H100 vLLM package has no matching FP16 operator measurements.
The native analytical fallback supplies compute timing, and the native collective model supplies TP communication timing.
The historical unused `splitwise_*` hardware-profile fields do not supply timing for this TokenSim experiment.
Prompt lengths beyond the model catalog's 4,096-token context are timing extrapolations.

## Preparation and execution

The commands require the existing Python 3.12 native dependency environment.
Preparation refuses to overwrite existing configurations or input files.
Plan generation refuses to overwrite an existing frozen plan.
The runner refuses to reuse an output directory with execution receipts.

```bash
python -m native_research.prepare_tp8_bandwidth
python -m native_research.build_bandwidth_plan \
  --profile configs/servegen7b-tp8-200/hardware.json \
  --input inputs/servegen-s0-first200-tp8.jsonl \
  --study-dir configs/servegen7b-tp8-200 \
  --run-root native_research/runs/servegen7b-tp8-bw-s0-200-01 \
  --output native_research/plans/servegen7b-tp8-bandwidth-s0-200.json
python -m native_research.run_overnight \
  --plan native_research/plans/servegen7b-tp8-bandwidth-s0-200.json \
  --run-root native_research/runs/servegen7b-tp8-bw-s0-200-01 \
  --report-dir native_research/reports/servegen7b-tp8-bw-s0-200-01
```

The runner refreshes Markdown and CSV reports as conditions finish and generates the plot after all ten conditions pass the audit.
The resource controller retains the existing dynamic memory admission policy.

```bash
python3 native_research/show_bandwidth_jobs.py \
  --plan native_research/plans/servegen7b-tp8-bandwidth-s0-200.json
python -m native_research.report_bandwidth \
  --run-root native_research/runs/servegen7b-tp8-bw-s0-200-01 \
  --output-dir native_research/reports/servegen7b-tp8-bw-s0-200-01
```

## Result definitions

Client TTFT starts at the original arrival and includes previous-turn and admission waits.
D-first uses the first output sampled by the selected decode worker.
The lifecycle ends after complete writeback.
Average TPOT is the arithmetic mean of per-request native decode time divided by output count minus one.
Throughput uses time zero through final writeback, including idle arrival gaps.
Affected share pairs each returning request with the same request in the Serial high-bandwidth reference and uses a strict two-times threshold.
Its denominator is 80 returning turns, so each affected request changes the percentage by 1.25 points.
The high-bandwidth reference retains native scheduling and queues; it is not a per-request zero-wait baseline.

The TP8 200-request results and TP1 8,000-request results differ in both deployment and workload scope.
Their aggregate latencies cannot isolate the causal effect of TP alone.
The earlier raw records and frozen source copies are retained.
