"""Opt-in completed-transfer admission helpers for uniform paged KV."""

import math
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.llm_request import RequestStatus


def describe_received_context(plan, requests, sender):
    """Describe the actual suffix blocks, preserving delta transfer identity."""
    plan.extra["received_context_tokens"] = {
        req.id: req.num_computed_tokens for req in requests
    }
    plan.extra["missing_block_ranges"] = {}
    plan.keys = []
    for req in requests:
        end = math.ceil(req.num_computed_tokens / sender.block_size)
        start = end - plan.request_block_counts[req.id]
        plan.extra["missing_block_ranges"][req.id] = [start, end]
        blocks = sender.scheduler.block_manager.block_table.get_blocks(req.id)
        for i in range(start, end):
            block = blocks[i]
            key = block.block_hash
            plan.keys.append(
                str(key)
                if key is not None
                else f"req:{req.id}:block:{block.block_number}"
            )


def admit_incoming_context(scheduler, req):
    migration = getattr(scheduler, "native_migration", None)
    if req.id in scheduler.incoming_context and migration is not None:
        if migration.admit_destination(scheduler, req):
            return True
    incoming = scheduler.incoming_context.pop(req.id, None)
    if incoming is None or req.needs_recompute:
        return False
    received, required_local = incoming
    if not 0 <= received <= req.context_len or (
        received == req.context_len and not getattr(req, "handover_flow", False)
    ):
        raise ConfigurationError("incoming KV must leave a real decode query")
    if (
        scheduler.block_manager.kv_cache_manager.plan_reuse(req).hit_tokens
        < required_local
    ):
        raise SimulationStateError(
            "destination prefix changed while transfer queued; KV reservation is required"
        )
    scheduler.block_manager.allocate(req)
    from TokenSim.placement.handover_clock import release_snapshot

    release_snapshot(
        req,
        getattr(scheduler.block_manager.kv_cache_manager, "handover_worker_id", None),
    )
    if getattr(req, "handover_flow", False):
        from TokenSim.llm.handover_reservation import release_destination

        release_destination(req)
    req.start_context_build(received)
    req.status = RequestStatus.RUNNING
    return True


def receive_transferred_context(worker, message):
    for plan in message.metadata.loads:
        for rid, tokens in plan.extra.get("received_context_tokens", {}).items():
            start = plan.extra["missing_block_ranges"][rid][0]
            worker.scheduler.incoming_context[rid] = (tokens, start * worker.block_size)
    for req in message.requests or []:
        if req.id in worker.scheduler.incoming_context:
            req.handoff_received_at = worker.env.now


def commit_received_context(worker, requests):
    for req in requests:
        if getattr(req, "handoff_received_at", None) is not None:
            if getattr(req, "handoff_ready_at", None) is None:
                worker.scheduler.block_manager.commit_input_cache(req)
                req.handoff_ready_at = worker.env.now


def release_pinned_prefill(engine, req, manager, sender):
    from TokenSim.llm.engine.messages import Task, WorkerStatus

    if req.completion_event is None:
        raise SimulationStateError("pinned P-to-D requests require a completion event")
    yield req.completion_event
    sender.pinned_handoffs.discard(req.id)
    if manager.block_table.get_blocks(req.id):
        manager.release_request_blocks(req)
    if sender.status == WorkerStatus.WAITING and sender.scheduler.waiting:
        engine.send_task(sender, Task.ADD, [])
