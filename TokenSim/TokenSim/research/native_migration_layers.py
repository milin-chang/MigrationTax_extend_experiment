"""Native scheduler cohorts and operator-composed MigrationTax layer events."""

from copy import copy
from dataclasses import replace
from math import ceil

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.research.native_migration_transport import reserve_destination


def is_pipeline(req):
    return (
        getattr(getattr(req, "native_plan", None), "branch", None) == "layer_pipeline"
    )


def install_cohorts(worker):
    """Preserve native batch selection within FF or ordinary execution cohorts."""
    scheduler = worker.scheduler
    original = scheduler.schedule_output

    def schedule_output(*args, **kwargs):
        names = ("running", "waiting", "loaded_waiting")
        heads = [r for name in names for r in getattr(scheduler, name)]
        if not heads:
            return original(*args, **kwargs)
        selected = is_pipeline(min(heads, key=lambda r: (r.arrival_time, r.id)))
        excluded, positions = {}, {}
        for name in names:
            rows = getattr(scheduler, name)
            positions[name] = {r.id: i for i, r in enumerate(rows)}
            excluded[name] = [r for r in rows if is_pipeline(r) != selected]
            setattr(scheduler, name, [r for r in rows if is_pipeline(r) == selected])
        try:
            return original(*args, **kwargs)
        finally:
            for name in names:
                restored = getattr(scheduler, name) + excluded[name]
                restored.sort(key=lambda r: positions[name].get(r.id, -1))
                setattr(scheduler, name, restored)

    scheduler.schedule_output = schedule_output


def install_store_interception(worker, driver):
    """Keep native lookup/allocation; consume FF load plans as layer reads."""
    for connector in flatten_connectors(worker.connector):
        if connector.name != "MooncakeStoreConnector":
            continue
        original = connector.start_load_kv
        storage = connector._storage

        def start_load(connector=connector, storage=storage, original=original):
            metadata = storage._metadata
            ordinary = []
            for plan in metadata.loads:
                ids = plan.request_ids
                req = driver.requests.get(ids[0]) if len(ids) == 1 else None
                if req is None or not is_pipeline(req):
                    ordinary.append(plan)
                    continue
                if plan.tier != "memory":
                    raise ConfigurationError(
                        "FF layer pipeline requires native DRAM store hits"
                    )
                req.native_migration.store_plans.append((connector, plan))
                driver.log(
                    "FF_native_store_plan_deferred",
                    req,
                    bytes=plan.bytes,
                    blocks=plan.blocks,
                )
            storage._metadata = replace(metadata, loads=ordinary)
            try:
                return original()
            finally:
                storage._metadata = metadata

        connector.start_load_kv = start_load


class NativeLayerPipeline:
    def __init__(self, driver):
        self.driver, self.env = driver, driver.env
        self.children = {}
        self.chunk_index = 0

    def backends(self, worker):
        if worker.id in self.children:
            return self.children[worker.id]
        backend = worker.latency_backend
        if not isinstance(backend, OperatorTableLatencyBackend) or (
            backend.model.components
            or backend.model.is_moe
            or backend._layer_backends
            or backend.native_mtp
            or (backend.tp != 1 and not worker.engine.config.handover_tp_groups)
            or backend.layers_local != self.driver.layers
        ):
            raise ConfigurationError(
                "Migration layer timing requires uniform dense native operators and validated TP groups"
            )
        children = []
        for layer in range(self.driver.layers):
            child = copy(backend)
            child.layers_local = child.dense_layers_local = 1
            child.moe_layers_local = 0
            child.is_last_stage = False
            child.include_pp_transfer = False
            child.include_embedding_scale = (
                backend.include_embedding_scale and layer == 0
            )
            child.include_step_overhead = backend.include_step_overhead and layer == 0
            children.append(child)
        self.children[worker.id] = children
        return children

    def fetch(self, req, layer):
        state = req.native_migration
        for connector, plan in state.store_plans:
            yield from self.driver.transfers.move(
                req, plan.blocks, "FF_store_layer", layer=layer, connector=connector
            )
        state.fetched.add(layer)
        self.driver.log("FF_prefix_layer_ready", req, layer=layer)

    def initialize_fetch(self, req):
        state = req.native_migration
        for connector, plan in state.store_plans:
            state.store_leases.append(self.driver.transfers.pin_store(connector, plan))
        state.fetch_events = [
            self.env.process(self.fetch(req, layer))
            for layer in range(self.driver.layers)
        ]

    def send(self, req, layer, after, chunk):
        state = req.native_migration
        start = state.received_tokens // req.block_size
        blocks = max(0, ceil(after / req.block_size) - start)
        yield from self.driver.transfers.move(req, blocks, "FF_send_layer", layer=layer)
        state.received[layer] = max(state.received_tokens, after)
        self.driver.log(
            "FF_destination_layer_ready",
            req,
            layer=layer,
            tokens=state.received[layer],
            computed_P_tokens=after,
            chunk=chunk,
        )

    def compute(self, worker, requests):
        chunk = self.chunk_index
        self.chunk_index += 1
        after = {r.id: r.num_computed_tokens + r.step_tokens for r in requests}
        before = {r.id: r.num_computed_tokens for r in requests}
        for req in requests:
            reserve_destination(self.driver, req)
            if req.native_migration.fetch_events is None:
                self.initialize_fetch(req)
        started, kernel_s = self.env.now, 0.0
        sends = []
        self.driver.log(
            "FF_native_chunk_started",
            worker=worker.id,
            chunk=chunk,
            requests=[r.id for r in requests],
            before=before,
            after=after,
        )
        for layer, backend in enumerate(self.backends(worker)):
            yield self.env.all_of(
                [r.native_migration.fetch_events[layer] for r in requests]
            )
            duration = backend.estimate_step_latency(requests)
            kernel_s += duration
            yield from worker.engine.run_complete_compute(worker.id, duration)
            self.driver.counters["native_layer_operator_calls"] += 1
            for req in requests:
                self.driver.log(
                    "FF_native_layer_computed",
                    req,
                    layer=layer,
                    chunk=chunk,
                    tokens=after[req.id],
                    native_compute_s=duration,
                )
                sends.append(
                    self.env.process(self.send(req, layer, after[req.id], chunk))
                )
        # Physical pages can be modified by the following native chunk only
        # after every layer send for this chunk has completed.
        yield self.env.all_of(sends)
        for req in requests:
            state = req.native_migration
            if set(state.received) != set(range(self.driver.layers)) or any(
                tokens < after[req.id] for tokens in state.received.values()
            ):
                raise ConfigurationError("Incomplete destination layer-plane barrier")
            state.received_tokens = min(state.received.values())
            for objects in state.store_leases:
                self.driver.transfers.unpin_store(objects)
            state.store_leases.clear()
            state.chunks += 1
            self.driver.log(
                "FF_native_chunk_committed",
                req,
                chunk=chunk,
                tokens=after[req.id],
                native_compute_s=kernel_s,
                elapsed_s=self.env.now - started,
            )
        return kernel_s
