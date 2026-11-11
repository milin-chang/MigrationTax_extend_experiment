"""Completed-context store writes using native transfer plans and ready times."""

from types import SimpleNamespace
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.kv_transfer import KVConnectorMetadata
from TokenSim.mooncake import pool_keys_for_request
from TokenSim.errors import ConfigurationError


def completed_hashes(req):
    count = req.num_computed_tokens // req.block_size
    hashes = getattr(req, "handover_context_hash_ids", None)
    if hashes is None:
        hashes = list(req.hash_ids or [])
        if req.prefill_len % req.block_size == 0:
            hashes.extend(req.output_hash_ids or [])
    if len(hashes) < count:
        raise ConfigurationError(
            "complete handover write-back requires computed-context hash identities"
        )
    return hashes[:count]


def save_completed_context(worker, req, start_delay):
    req.handover_store_writeback_bytes = 0
    hashes = completed_hashes(req)
    manager = worker.scheduler.block_manager
    keys = build_prefix_keys(
        hashes,
        model=manager.kv_cache_manager.model,
        cache_salt=req.cache_salt,
        reuse_group=req.reuse_group,
    )
    manager.kv_cache_manager.register_blocks(
        manager.block_table.get_blocks(req.id)[: len(keys)], keys
    )
    state = getattr(worker.engine.router, "arrival_state", None)
    if state is not None:
        state.publish(worker, req)
    latency = 0.0
    for connector in flatten_connectors(worker.connector):
        if connector.name != "MooncakeStoreConnector":
            continue
        if connector.options.admission_policy == "never":
            continue
        source = SimpleNamespace(
            id=req.id,
            hash_ids=hashes,
            input_cache_keys=[],
            prefill_len=len(hashes) * req.block_size,
            block_size=req.block_size,
            cache_salt=req.cache_salt,
            reuse_group=req.reuse_group,
        )
        pool_keys = pool_keys_for_request(
            source,
            model_name=connector.model_name,
            rank_info=connector.rank_info,
            store_namespace=connector.store_namespace,
            cache_prefix=connector.options.cache_prefix,
            tenant_id=connector.options.tenant_id,
            group_id=connector.options.group_id,
            pcp_rank=connector.options.pcp_rank,
            dcp_rank=connector.options.dcp_rank,
        )
        plan = connector._build_save_plan(
            req, pool_keys, computed_tokens=req.num_computed_tokens
        )
        if plan is None:
            continue
        req.handover_store_writeback_bytes += plan.bytes
        connector.set_simulation_time(worker.env.now + start_delay + latency)
        connector.bind_connector_metadata(
            KVConnectorMetadata(connector_name=connector.name, saves=[plan])
        )
        latency += connector.wait_for_save()
    return latency


def prepare_writebacks(worker, requests):
    worker.handover_pending_completions = []
    if not worker.engine.config.handover_flow or worker.role != "decode":
        return 0.0
    latency = 0.0
    for req in requests:
        if not getattr(req, "handover_completion_pending", False):
            continue
        req.handover_writeback_started_at = worker.env.now + latency
        latency += save_completed_context(worker, req, latency)
        worker.handover_pending_completions.append(req)
    return latency


def finalize_writebacks(worker):
    for req in getattr(worker, "handover_pending_completions", []):
        finalize_completed_context(worker, req)
    worker.handover_pending_completions = []


def finalize_completed_context(worker, req):
    """Finalize one writeback without sharing a worker's batch completion list."""
    worker.scheduler.free(req)
    req.handover_writeback_completed_at = worker.env.now
    state = getattr(worker.engine.router, "arrival_state", None)
    if state is not None:
        state.store_ready(req)
    req.completed_at = worker.env.now
    req.total_time += worker.env.now - req.handover_output_finished_at
    req.handover_completion_pending = False
    timing = req.complete_timing()
    if req.record_timing:
        worker.engine.timing_recorder.record_request(timing)
    if req.completion_event is not None and not req.completion_event.triggered:
        req.completion_event.succeed(timing)
    worker.engine.record_request_completion("decode", req)
