"""Optional prefill completion without sampling an output on the P worker."""


def advance_flow(worker, req, latency, batch):
    """Return True when the handover flow handled this worker step."""
    if not worker.engine.config.handover_flow:
        return False
    migration = getattr(worker.engine, "native_migration", None)
    if migration is not None and migration.advance(worker, req, latency, batch):
        return True
    if worker.role == "prefill" and req.is_prefill:
        req.num_computed_tokens += req.step_tokens
        req.scheduled_tokens = req.scheduled_query_tokens = (
            req.scheduled_placeholder_tokens
        ) = 0
        req.prefill_chunks += 1
        req.prefill_service_time += latency
        if req.num_computed_tokens >= req.prefill_len:
            req.handover_prefill_complete = True
            req.handover_prefill_finished_at = worker.env.now
            worker.scheduler.block_manager.commit_input_cache(req)
            state = getattr(worker.engine.router, "arrival_state", None)
            if state is not None:
                state.publish(worker, req)
        return True
    if worker.role == "decode" and req.generation_idx == 0:
        req.handover_first_decode_service_s = latency
    event, record = req.completion_event, req.record_timing
    emitted = 1 if req.is_context_build else req.step_tokens
    final = req.generation_idx + emitted >= req.decode_len
    if final:
        req.completion_event = None
        req.record_timing = False
    before = req.generation_idx
    req.advance(worker.env, latency, batch, worker.engine.timing_recorder)
    req.completion_event, req.record_timing = event, record
    if migration is not None:
        migration.observe_sample(worker, req, before)
    if final and req.is_done:
        req.handover_output_finished_at = worker.env.now
        req.handover_completion_pending = True
    return True


def complete_worker_step(worker, requests, latency):
    from TokenSim.llm.handover_writeback import prepare_writebacks

    worker.step(requests, latency)
    if not worker.engine.config.handover_flow:
        worker.scheduler.update(requests)
        worker.connector.set_simulation_time(worker.env.now)
        return worker.connector.wait_for_save()
    worker.connector.set_simulation_time(worker.env.now)
    native_save = worker.connector.wait_for_save()
    writeback = prepare_writebacks(worker, requests)
    worker.scheduler.update(
        [r for r in requests if not getattr(r, "handover_completion_pending", False)]
    )
    return native_save + writeback


def wait_for_capacity(worker, running, load_latency):
    if running or load_latency or not worker.scheduler.waiting:
        return False
    return worker.engine.config.handover_flow or (
        worker.engine.config.preserve_p2p_kv
        and worker.role == "prefill"
        and getattr(worker, "pinned_handoffs", None)
    )
