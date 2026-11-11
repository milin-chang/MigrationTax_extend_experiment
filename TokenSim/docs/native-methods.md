# Native research methods

The bundle-level `native_methods.worker` enters `TokenSim.research.native_runtime.execute`.
This research entry creates a real `LLMEngine`, paged attention schedulers, native block managers, and Mooncake transfer and store connectors.
The analytical `algorithm.evaluate()` functions remain reference implementations and do not supply this entry's completion times.

## Configuration and time

The current experiment uses four prefill workers and four decode workers, each with one H200 and the native catalog `llama-3-8b` model.
The group settings use 16-token blocks, at most 16 sequences per worker, and a 4,096-token batch budget.
The native operator backend uses its GPU roofline fallback.
It does not use the common MigrationTax PF/ST formulas.
Contexts beyond the catalog training range retain their source lengths and represent timing extrapolation.

The store has a finite 512 GiB pool, 256 GiB of staging space per worker, and a 4 TiB shared host budget.
These are explicit experiment settings.
A missing store prefix can require native recomputation.
The result does not assume an unbounded store.
The experiment records the effective hardware, model, connector, and simulation configuration in `effective-config.json`.

Original source arrival times are retained.
A request waits for its explicit predecessor's complete native response and writeback before submission.
This wait contributes to request latency.
The original source turn and the native router's conversation ordinal are recorded separately.

## Admission and cache ownership

`NativeAdmission` derives conservative block budgets from each native GPU allocator's capacity minus its watermark.
A request obtains these budgets before native destination prefix pinning.
This prevents queued requests from pinning every decode page before a complete request can run.
The original native allocator still performs physical allocation, prefix sharing, eviction, and release.
Budget tokens are a capacity gate and do not supply simulated time.

Placement uses ready prefix information, persistent decode affinity, occupied budgets, and native queue sizes.
A decode-local request does not wait for a prefill budget.
A cache hit is checked again after a capacity wait because native completion and eviction can change it.
These placement and admission choices are experiment extensions and can change results from the earlier 7B diagnostic configuration.

## Method execution

Serial executes native prefill on P, transfers the required context, and samples the response on D.
The adapted PPD method sends a D-hit request to the actual D engine for native prefill and decode.
D-local prefill produces the first output through the native fused sampling callback.
D-miss PPD uses the Serial path.

The opt-in MigrationTax driver is installed only for that method.
HF retains native P execution through `k=min(2,O)`, protects the base pages and partial tail, and transfers base and incremental context before D ownership.
FF composes one-layer native operator backends with actual Mooncake transfer events.
Every layer send for a chunk must drain before the next native chunk can modify its pages.
Disk-tier layer fetch, hybrid KV, mixture-of-experts models, parallelism greater than one, and attention/FFN execution are not supported by this driver.
The driver remains a candidate pending its execution checks and complete workloads.

The core hooks fall through to their existing behavior when the driver is absent.
Changes to the single-request writeback finalizer also require ordinary Serial and PPD regression checks.
The research runtime accepts a method name before that method has completed acceptance.
The bundle capability matrix records the current validation boundary.

## Output semantics

`requests.jsonl` contains source request results.
`native-route-decisions.jsonl` records actual routing and cache observations.
The native exporter writes its unmodified result structure under `native/`.
`native-end-state.json` and `native-admission-budget.json` record queue, page, pin, and budget drainage.
MigrationTax additionally writes its transfer, layer, sample, synchronization, and ownership events.

`native_prefill_latency_s` is the native request's time from original arrival to its first sampled output.
`native_total_time_s` includes completion-clock writeback and should not be interpreted as the last-output response time.
`native_decode_time_sum_s` preserves the native request counter's meaning.
The source request count is separate from the number of worker compute calls, batches, and transfer jobs.

A request with all output tokens but an unfinished handoff or writeback is incomplete.
The runtime uses actual full completion callbacks and requires the completed count to equal the input count before reporting success.
Request-local identity retirement is a memory optimization candidate.
It runs after completion listeners release native ownership, clears only request-local planning metadata, and retains scalar metrics.
Its full-run equivalence must be checked before accepting results from the revised version.

## Evidence and reruns

The bundle entry and resource-limited commands are documented in `../../native_research/README.md`.
Each formal condition saves its input checksum, environment, source snapshot, configuration, native result, and job receipt.
The initial successful Serial Mooncake baseline is `native_research/runs/full-token-serial-mooncake-s0-v3/` in the bundle.
It completed 12,031 source requests and 4,122,048 outputs.
That baseline precedes the MigrationTax hooks and identity retirement candidate.
Current acceptance and failures are recorded in the bundle's `native_research/STATUS.md`.
