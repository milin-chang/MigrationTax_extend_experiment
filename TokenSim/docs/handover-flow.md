# Experimental handover flow

This local fork adds opt-in components for multi-turn shared key/value (KV) cache experiments.
The base commit is `8ec1450c30655fad3c06ab18f39ef77f124e0812`.
The component mode retains the native completion clock and its original endpoint.
The complete scenario mode below adds an arrival-ordered cache policy and generates the initial output on the decode worker.

## Complete scenario mode

```python
SimulationConfig(
    batching="paged-attn",
    block_size=1,
    max_num_seqs=1,
    handover_flow=True,
    handover_clock="arrival",
    handover_cost_model="pf_st",
    handover_fixed_overhead_s=0.001,
)
```

`handover_flow` enables the component handoff and routing options automatically.
The store connector uses `save_policy="turn_end"` so that prompt write-back does not add an unrequested write to the prefill critical path.
The decode worker writes all materialized context KV after producing the last output.
Each turn transfers the full materialized context once, including an unchanged prefix already present in the store.
The final sampled token has not been forwarded through the model, so materialized KV covers `input_tokens + output_tokens - 1` positions.
Completion and source release wait for write-back visibility.

Arrival mode requires one supplied identity per token and a `handover_context_hash_ids` sequence for all materialized input/output positions.
One-token blocks avoid rounding partial prefixes or generated output boundaries.
Prefill, decode and shared-store objects use separate versions.
Device caches apply whole-conversation least-recently-used eviction at trace arrivals.
The prefill holder retains the requested prompt and the decode holder retains its materialized context.
The store is logically unbounded in the study configuration.
The arrival decision captures reusable versions, pins them across queue intervals and waits on actual producer readiness events.
An arrival can therefore record a planned resident prefix while execution waits for that version to finish.
No request reads future KV or copies an expected cell label from a trace.
Actual native allocation must still fit; a stalled or failed allocation is a failed run.
Cancellation removes unproduced versions and releases snapshot references.

The source specifies usable KV capacity, so this mode sets the native additional allocation watermark to zero.
Lowest device ID resolves ties, and exact resident conversation counts rank eligible devices.
These tie rules and serial service are declared experimental choices.
Prefill and decode service both follow tickets assigned at trace arrival, so a later request with a short prefill cannot fill the decode cache ahead of an earlier long request.
The decode worker reserves the complete materialized turn's slots at admission to prevent within-turn preemption.
An evicted version retains only the prefix ranges that captured readers still need.
Logical arrival decisions and real service timestamps are exported separately.
The scenario's omitted port and landing-buffer limits remain omitted.

The prefill worker completes prompt construction without sampling an output.
The decode worker waits for peer-to-peer transfer and executes the initial decode query.
Its output event defines time to first token, including the configured fixed per-turn overhead once.
A one-output-token request follows the same path.
The raw prefill completion endpoint is exported separately and is not called the unmodified engine's first-token result.

`handover_cost_model="native"` keeps the native performance backend.
`handover_cost_model="pf_st"` uses the supplied arithmetic constants and requires serial service.
Its prefill price is `PF(new tokens)` with cumulative differences across chunks, and its decode price is `ST(current context)`.
This is the supplied analytical abstraction for cached attention and does not claim measured kernel accuracy.
The study compares exact architectural KV bytes of 524288 per token separately from the source's rounded 524000-byte approximation.

## Component mode from the earlier audit

## Configuration

The Python engine API accepts these `SimulationConfig` fields:

```python
SimulationConfig(
    batching="paged-attn",
    preserve_p2p_kv=True,
    handover_routing=True,
)
```

Both fields default to `False`.
The command-line interface does not expose these experimental fields.
These options require dedicated prefill and decode workers, uniform paged caches, and tensor, pipeline and data parallel sizes of one.
Attention/feed-forward disaggregated execution is unsupported.

`preserve_p2p_kv` preserves the prefill worker's blocks until the request completes or is cancelled.
The decode worker receives the context once through the peer-to-peer transfer and avoids fetching the same prompt again from the shared store.
Transferred input blocks become reusable after the modeled load completes.
If a destination prefix disappears while the transfer is queued, admission aborts with a state error instead of claiming untransferred blocks exist.
Reserving destination hits across this queue interval remains unimplemented.
The transfer byte count subtracts the destination's reusable prefix blocks.
The simulator transfers metadata and models elapsed time; it does not copy tensor payloads.

`handover_routing` selects a prefill and decode worker once per arriving request.
Each request must supply `chat_id`, contiguous zero-based `turn`, and explicit `prefix_tokens`.
Nonzero inherited prefixes also require block hash identities.
The Qwen JSON Lines loader forwards the optional `prefix_tokens` field.
The router prefers prefill workers holding the entire inherited prefix and retains the conversation's previous decode worker.
Available workers are ranked by the number of known conversations whose initial block remains resident, then by worker ID.
This resident-count proxy and the ID tie rule are documented experimental choices.
They have not been established as equivalent to whole-conversation least-recently-used eviction.
Partial prefix coverage remains observable and is not automatically classified as a full hit.
Profile reset clears observations while retaining routes and affinity.

## Shared store updates

The store connector accepts `save_policy="prefill_delta"`.
This policy publishes previously absent complete prompt blocks once the prompt finishes, including when admission fetched an existing prefix from the store.
It does not save the whole prompt at every decode step.
It does not publish newly generated output KV at turn completion.
The native `mooncake` policy retains its previous load-step save behavior.
The scheduler also records actual local/store/compute provenance for uniform prefill admission.
This provenance correction intentionally changes the reported source counts without changing the default scheduling calculation.
These source counts describe available cached input coverage; a full prompt hit still executes the final query position needed for logits, so the counts alone do not measure executed query work.

## Acceptance limits

The small tests exercise local reuse, store retrieval, independent store-disabled recomputation, decode hits and misses, cancellation, pinned capacity pressure, and eight-worker routing.
The six diagnostic states derive from live cache observations.
The recomputation cases explicitly disable store admission and are separate from the main unbounded-store scenario.
Mixed or partial prefix coverage produces an unclassified observation.

The cache still advances on native computation and transfer completion.
The three-request clock diagnostic demonstrates that changing transfer bandwidth can change a later arrival's cache state.
The source's fixed-state A/B assumption therefore fails in this mode.

Native prefill samples an output token before peer-to-peer handoff.
A request for one output token completes on the prefill worker and has no decode handoff.
`native_timing_observation` preserves native time to first token and reports the source endpoint as unsupported.
It does not fabricate a common endpoint by adding a decode step.

These component-mode limitations are retained as regression tests.
They do not apply as claims about the complete scenario mode above.
Formal trace A/B runs remain gated by the complete mode's tests and per-request invariants.
Earlier native trace diagnostics retain their original scope and are not results for the complete mode.

## Verification

```bash
python3 -m unittest \
  tests.test_handover_store_policy \
  tests.test_handover_p2p \
  tests.test_handover_routing \
  tests.test_handover_states \
  tests.test_handover_clock \
  tests.test_handover_visibility \
  tests.test_handover_flow \
  tests.test_handover_arrival \
  tests.test_handover_arrival_states \
  tests.test_handover_cost \
  -v
```

The study runner fixes the container to two logical CPUs and 8 GiB of memory, with a bounded wall-clock timeout.
The study directory stores commands, logs, resource measurements and machine-readable acceptance observations separately from this repository.
