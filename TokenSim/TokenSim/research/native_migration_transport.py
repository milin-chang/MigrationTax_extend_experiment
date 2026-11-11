"""Native Mooncake transfers and physical leases for MigrationTax events."""

from math import ceil, inf

import simpy

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer.connectors.base import (
    flatten_connectors,
    transfer_block_bytes,
)
from TokenSim.kv_transfer.connectors.metadata import ConnectorTransferPlan
from TokenSim.latency.base import roofline_latency_us
from TokenSim.llm.handover_writeback import save_completed_context
from TokenSim.mooncake import Segment


def mooncake(worker):
    return next(
        c for c in flatten_connectors(worker.connector) if c.name == "MooncakeConnector"
    )


def scoped_completed_writeback(worker, req):
    """Issue native synchronous writes without replacing an active D step.

    The native store, per-request bookkeeping, statistics and completion delay
    retain their updates. Only the temporary connector metadata and clock are
    restored, before any SimPy yield can expose them to another request.
    Async store pending-state handling belongs to the native worker loop and
    is not supported by this out-of-band short-response issue path.
    """
    saved = []
    for connector in flatten_connectors(worker.connector):
        if connector.name != "MooncakeStoreConnector":
            continue
        if connector.options.load_async or connector.options.transfer_overlap:
            raise ConfigurationError(
                "HF background writeback requires synchronous native store plans"
            )
        saved.append((connector, connector._metadata, connector.simulation_time))
    try:
        return save_completed_context(worker, req, 0.0)
    finally:
        for connector, metadata, clock in saved:
            connector.bind_connector_metadata(metadata)
            connector.set_simulation_time(clock)


def reserve_destination(driver, req):
    state = req.native_migration
    if state.destination_reserved:
        return
    manager = state.d.scheduler.block_manager
    source_blocks = list(req._physical_token_blocks)
    source_admission = req.admission_state()
    state.received_tokens = manager.kv_cache_manager.plan_reuse(req).hit_tokens
    logical = req._logical_tokens
    req._logical_tokens = req.prefill_len
    try:
        manager.allocate(req)
    finally:
        req._logical_tokens = logical
    state.destination_input_keys = list(req.input_cache_keys)
    req.restore_admission_state(source_admission)
    # Allocation records the real D pages. The active native P request retains
    # its physical view until the ownership barrier installs the D view.
    manager._sync_request_blocks(req, source_blocks)
    state.destination_reserved = True
    driver.log(
        "D_pages_reserved", req, blocks=manager.block_table.get_num_blocks(req.id)
    )


def restore_destination(req):
    manager = req.native_migration.d.scheduler.block_manager
    manager._sync_request_blocks(req, manager.block_table.get_blocks(req.id))


def pin_source(req, tokens):
    manager = req.native_migration.p.scheduler.block_manager
    blocks = manager.block_table.get_blocks(req.id)[: ceil(tokens / req.block_size)]
    if len(blocks) != ceil(tokens / req.block_size):
        raise SimulationStateError("Migration source has insufficient native pages")
    for block in blocks:
        manager.gpu_allocator.acquire(block)
    return manager, blocks


def release_source(snapshot):
    manager, blocks = snapshot
    for block in blocks:
        manager.gpu_allocator.free(block)


def clone_hf_tail(driver, req):
    """Copy a physical tail before P overwrites a base-transfer snapshot."""
    state = req.native_migration
    if state.k <= 1 or req.prefill_len % req.block_size == 0:
        return 0.0
    worker = state.p
    manager = worker.scheduler.block_manager
    blocks = manager.block_table.get_blocks(req.id)
    index = req.prefill_len // req.block_size
    old = blocks[index]
    new = manager.gpu_allocator.allocate()
    new.block_hash, new.is_full, new.cached = old.block_hash, old.is_full, False
    blocks[index] = new
    manager.block_table.set_blocks(req.id, blocks)
    manager._sync_request_blocks(req, blocks)
    manager.gpu_allocator.free(old)
    bytes_ = int(
        (req.prefill_len % req.block_size) * worker.cache_config.size_per_token
    )
    duration = roofline_latency_us(worker.device, bytes_ * 2) * 1e-6
    driver.log("HF_tail_COW_started", req, bytes=bytes_, duration_s=duration)
    return duration


class NativeTransfers:
    """FIFO endpoint queues whose service comes from native Mooncake jobs.

    Queueing and layer payloads are explicit adapter extensions. Native
    TransferEngineSimulator computes protocol, fixed latency and bandwidth.
    """

    def __init__(self, driver):
        self.driver = driver
        self.env = driver.env
        self.queues = {}
        self.store_leases = {}

    def move(self, req, blocks, kind, *, layer=None, connector=None):
        state = req.native_migration
        source, target = state.p, state.d
        store_read = kind == "FF_store_layer"
        connector = connector or mooncake(source)
        source_id = "store" if store_read else source.id
        target_id = source.id if store_read else target.id
        size = blocks * transfer_block_bytes(source.cache_config)
        if layer is not None:
            if size % self.driver.layers:
                raise ConfigurationError(
                    "Native KV bytes must divide into layer planes"
                )
            size //= self.driver.layers
        queue = self.queues.setdefault(
            (source_id, target_id), simpy.Resource(self.env, 1)
        )
        issued = self.env.now
        with queue.request() as slot:
            yield slot
            started = self.env.now
            transfer = connector.transfer_engine.create_transfer(
                source_segment=Segment(
                    str(source_id), "dram" if store_read else "vram"
                ),
                target_segment=Segment(str(target_id), "vram"),
                blocks=blocks,
                bytes_=size,
                kind="store" if store_read else "p2p",
                topology="cross_node",
            )
            # Store paths also account for the native host/network constraints.
            duration = (
                (
                    connector._transfer_latency(transfer)
                    if store_read
                    else transfer.latency
                )
                if size
                else 0.0
            )
            plan = ConnectorTransferPlan(
                request_ids=[req.id],
                request_block_counts={req.id: blocks},
                source_worker_id=None if store_read else source.id,
                target_worker_id=target_id,
                blocks=blocks,
                bytes=size,
                latency=duration,
                kind="load" if store_read else "save",
                tier="memory" if store_read else "p2p",
                extra={
                    "connector": connector.name,
                    "layer": layer,
                    "method_event": kind,
                    "tp_shards": getattr(source.cache_config, "handover_tp_size", 1),
                },
            )
            job = connector.transfer_engine.submit(transfer, now=started, async_=False)
            connector.stats.record_transfer(
                blocks=blocks, bytes_=size, latency=duration, kind=plan.kind
            )
            connector.mooncake_stats.record_transfer(
                bytes_=size,
                latency=duration,
                kind="store_load" if store_read else "p2p",
                blocking_latency=duration,
            )
            self.driver.log(
                kind + "_started",
                req,
                layer=layer,
                blocks=blocks,
                bytes=size,
                queue_s=started - issued,
                service_s=duration,
                source=source_id,
                target=target_id,
                native_transfer=type(transfer).__name__,
                tp_shards=getattr(source.cache_config, "handover_tp_size", 1),
                per_rank_bytes=size
                // getattr(source.cache_config, "handover_tp_size", 1),
                completion_scope="all_tp_shards",
            )
            try:
                yield self.env.timeout(duration)
            finally:
                # Preserve native live-job ownership without retaining every
                # finished layer job for the entire public workload.
                connector.transfer_engine.jobs.remove(job)
            self.driver.log(
                kind + "_ready", req, layer=layer, blocks=blocks, bytes=size
            )
            return plan

    def pin_store(self, connector, plan):
        if plan.tier != "memory":
            raise ConfigurationError(
                "MigrationTax layer reads currently require DRAM store objects"
            )
        keys = connector._pool_keys_from_plan(plan)
        store = connector.service.store
        # Native get validates ready objects, applies the native touch and pins
        # their actual eviction clock. Counted leases restore it after all reads.
        objects = [store.objects[key.to_string()] for key in keys]
        for obj in objects:
            key = id(obj)
            if key not in self.store_leases:
                self.store_leases[key] = [obj, obj.in_use_until, 0]
            self.store_leases[key][2] += 1
        connector.service.store.get(
            keys,
            plan.tier,
            now=self.env.now,
            initial_delay=inf,
            stats=connector.mooncake_stats,
        )
        connector.mooncake_stats.record_pool_keys(plan.keys)
        return objects

    def unpin_store(self, objects):
        for obj in objects:
            lease = self.store_leases[id(obj)]
            lease[2] -= 1
            if lease[2] == 0:
                obj.in_use_until = max(lease[1], self.env.now)
                del self.store_leases[id(obj)]
