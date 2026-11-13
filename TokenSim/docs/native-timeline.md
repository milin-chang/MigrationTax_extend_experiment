# Native P/D timeline diagnostics

The research adapter can stream per-request and per-worker-step comma-separated values (CSV) into a new run directory.
Enable collection with `--record-timeline` on `python -m native_methods.worker --backend tokensim`.
The Python entry point accepts `record_timeline=True` on `TokenSim.research.native_runtime.execute`.
Collection is disabled by default.
This option supports ordinary independent paged-attention workers, including the symmetric tensor parallel (TP) handover groups.

```bash
python -m native_methods.worker \
  --backend tokensim \
  --method MigrationTax \
  --trace mooncake \
  --seed 0 \
  --input config/five-stage-llama7b-h100-sxm80gb-tp1-s0/inputs/mooncake-s0.jsonl \
  --limit 200 \
  --hardware-profile config/five-stage-llama7b-h100-sxm80gb-tp1-s0/hardware.json \
  --experiment-config config/five-stage-llama7b-h100-sxm80gb-tp1-s0/study_nvl4.json \
  --record-timeline \
  --output native_research/runs/timeline-example-01/mooncake/nvl4/MigrationTax/seed-0
```

Run this command from the experiment repository root in its configured Python environment.
Choose a new output directory for each run.
The worker rejects an existing output directory.
Existing frozen execution plans require regeneration to include this option and updated source checksums.

## Files and timestamps

| File | Granularity |
|---|---|
| `request-timeline.csv` | One completed, rejected, or unfinished request |
| `worker-steps.csv` | One scheduling step on a P or D model-group leader |
| `kv-transfers.csv` | MigrationTax explicit layer, base, and incremental transfers |
| `timeline-schema.json` | Columns, units, coverage, completion status, and row counts |

All timestamps use seconds on the simulation clock.
An empty cell means the event was unobserved or did not apply.
A numerical zero means an observed zero, including events at time zero.
Check both the run completion receipt and `timeline-schema.json` before using the CSV as a completed experiment.
Partial diagnostic files are retained when a run fails.

## Request endpoints

| Column | Meaning |
|---|---|
| `arrival_s` | Original trace arrival |
| `parent_ready_s` | Dependency satisfied, including parent writeback |
| `p_budget_acquired_s`, `d_budget_acquired_s` | Successful P/D budget acquisition times |
| `submitted_s` | Submission to the native engine after budget acquisition |
| `p_queue_enter_s`, `d_queue_enter_s` | First ADD message enqueued to the corresponding worker |
| `p_scheduler_enter_s`, `d_scheduler_enter_s` | First insertion into the scheduler when that worker consumes the ADD message |
| `p_prefill_start_s` | First P prompt-compute call |
| `p_last_prefill_compute_end_s` | Last P prompt-compute call end |
| `p_prefill_end_s` | Worker-step completion that finishes the prompt, including an FF chunk handoff barrier |
| `p_first_compute_start_s`, `p_last_compute_end_s` | P compute interval, including HF P-side decode |
| `d_first_compute_start_s` | First compute call on D |
| `d_first_decode_step_end_s` | First D step classified as decode at scheduling time |
| `first_d_output_at_s`, `first_d_output_phase` | First observed output on D and its phase |
| `first_client_output_at_s` | First observed output on P or D |
| `completed_at_s` | Request completion including writeback |

D-local prefill can emit the first output before the first D decode step.
Its P-prefill columns remain empty.
HF responses with one or two output tokens can finish all client output on P and have no D output or decode step.
Destination KV page reservation does not imply D queue insertion.
An ADD message can wait behind the worker's current batch before reaching its scheduler.
The queue and scheduler columns preserve both timestamps.
Every completed request has a D budget acquisition timestamp.
P participants in the serial, HF, and FF paths have a P budget acquisition timestamp.
D-local prefill may bypass the P budget and leave that field empty.
MigrationTax's dedicated admission path records the successful acquisitions with the same observation callback as the ordinary path.
An abandoned D-hit acquisition that is released after cache eviction is excluded from the successful-acquisition columns.
Request endpoints are observed independently of the existing cumulative TTFT field.

## Worker steps and transfers

`request_ids` and `request_phases` identify the shared batch.
`batch_size` counts requests and `batch_tokens` counts scheduled query tokens.
TP groups produce one leader row instead of one duplicate row per physical rank.
Request-level batch durations must not be added once per request to estimate device occupancy.

`load_wait_s` records the native blocking connector-load latency from `prepare_step`.
`load_end_s` is the corresponding scheduled completion, calculated as start plus latency.
MigrationTax FF layer reads bypass this ordinary blocking load and appear in `kv-transfers.csv`.
That transfer file has only a header for methods without explicit MigrationTax events.
It does not claim to list every native connector transaction.

`compute_service_s` sums native compute calls, including native TP collective costs.
`compute_span_s` retains gaps between calls in an FF pipeline.
`precompute_wait_s`, `between_compute_wait_s`, and `postcompute_wait_s` expose surrounding waits without assigning all of them to a single transfer path.
Step completion precedes synchronous writeback waits, while request completion includes writeback.
Transfer queue, service start, and observed ready times are recorded separately.
Compute and transfers can overlap, so their cumulative durations do not directly sum to time to first token (TTFT).

## Validation

The observers wrap existing calls and yields without adding simulation events.
CSV rows are written incrementally; completed step histories are not accumulated in memory.
Native result JSON and per-request output must match collection-disabled runs, except for simulator wall-clock execution time.
Diagnostic I/O can increase the wall-clock time required to execute an experiment.

```bash
cd TokenSim
python -m unittest tests.test_native_timeline
```
