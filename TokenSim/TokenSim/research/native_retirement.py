"""Drop completed-request identity lists after native release callbacks drain.

The native finite cache/store retain their own keys and physical blocks. Length,
timing, source provenance, counters and method plans remain available to the
native reporter. Rebinding never mutates a shared cache-key list in place.
"""


def retire_identity(engine, req, counters):
    # Completion event listeners include native destination-prefix and pinned-P
    # release processes. This zero-time yield follows all of those listeners.
    yield engine.env.timeout(0)
    if not req.is_done or getattr(req, "handover_completion_pending", False):
        raise RuntimeError(
            "Identity retirement before native response/writeback completion"
        )
    if getattr(req, "handover_d_reservation", None) is not None:
        raise RuntimeError(
            "Identity retirement before native destination prefix release"
        )
    if any(
        worker.scheduler.block_manager.block_table.get_num_blocks(req.id)
        for worker in engine.workers
    ):
        raise RuntimeError(
            "Identity retirement while a native block table still owns the request"
        )
    migration = getattr(req, "native_migration", None)
    if migration is not None:
        if migration.base_snapshot is not None or migration.store_leases:
            raise RuntimeError("Identity retirement with outstanding migration leases")
        if migration.base_event is not None and not migration.base_event.triggered:
            raise RuntimeError("Identity retirement before migration base completion")
        # Ownership gave req and its migration state the same key list. Rebind
        # both references only after every native owner has released the pages.
        migration.destination_input_keys = []
        migration.store_plans = []
        migration.fetch_events = None
        migration.base_event = None
    # Placement probes every P manager. Native free() forgets the chosen
    # worker's plan, while unselected managers can otherwise retain its entire
    # prefix chain forever. These are request-local plan caches, not resident
    # KV entries; the native API does not touch block ownership or LRU here.
    for worker in engine.workers:
        manager = worker.scheduler.block_manager.kv_cache_manager
        counters["completed_routing_plan_keys_retired"] += len(
            manager._input_keys_cache.get(req.id, ())
        )
        manager.forget_request(req.id)
    req.hash_ids = None
    req.output_hash_ids = None
    req.handover_context_hash_ids = None
    req.input_cache_keys = []
    req.trace_hash_ids = None
    counters["completed_request_identities_retired"] += 1
