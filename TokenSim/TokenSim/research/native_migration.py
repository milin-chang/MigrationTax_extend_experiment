"""Opt-in MigrationTax events through native TokenSim workers and paged KV.

Call install(engine, admission, output, counters), then on_route after the
runtime has built an untimed native_plan. Call finish after the event loop.
Serial and PPD do not install this driver and retain their existing behavior.
"""

import gzip
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace

from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.llm.engine.messages import Task
from TokenSim.llm.handover_reservation import release_destination, wake_for_capacity
from TokenSim.llm.handover_writeback import (
    completed_hashes,
    finalize_completed_context,
)
from TokenSim.llm.llm_request import RequestStatus
from TokenSim.research.native_migration_admission import install_admission
from TokenSim.research.native_migration_layers import (
    NativeLayerPipeline,
    install_cohorts,
    install_store_interception,
    is_pipeline,
)
from TokenSim.research.native_migration_transport import (
    NativeTransfers,
    clone_hf_tail,
    pin_source,
    release_source,
    reserve_destination,
    restore_destination,
    scoped_completed_writeback,
)


def install(engine, admission, output, counters):
    if getattr(engine, "native_migration", None) is not None:
        raise ConfigurationError("A MigrationTax driver is already installed")
    driver = NativeMigration(engine, admission, output, counters)
    engine.native_migration = driver
    install_admission(admission)
    for worker in engine.request_workers:
        if not engine._is_schedulable_worker(worker):
            continue
        worker.scheduler.native_migration = driver
        if worker.role == "prefill":
            install_cohorts(worker)
        install_store_interception(worker, driver)
        manager = worker.scheduler.block_manager
        original = manager.can_allocate

        def can_allocate(req, original=original, manager=manager):
            state = getattr(req, "native_migration", None)
            if state is not None and state.d.scheduler.block_manager is manager:
                if state.destination_reserved and state.owned_by_d:
                    return True
            return original(req)

        manager.can_allocate = can_allocate
        if worker.role == "prefill":
            original_count = manager._admission_block_count

            def count(req, plan, original_count=original_count):
                state = getattr(req, "native_migration", None)
                if state is not None and state.branch == "prefill_side_handoff":
                    return max(
                        plan.total_logical_blocks,
                        ceil((req.prefill_len + state.k - 1) / req.block_size),
                    )
                return original_count(req, plan)

            manager._admission_block_count = count
    engine.dispatch_prefill_to_decode = driver.dispatch
    return driver


class NativeMigration:
    def __init__(self, engine, admission, output, counters):
        if not engine.config.handover_flow or not engine.config.preserve_p2p_kv:
            raise ConfigurationError(
                "MigrationTax requires native completion-clock handover"
            )
        if (
            engine.config.handover_clock != "completion"
            or engine.af_executor is not None
        ):
            raise ConfigurationError(
                "MigrationTax currently requires ordinary completion-clock workers"
            )
        self.engine, self.env, self.admission = engine, engine.env, admission
        for worker in engine.request_workers:
            for connector in flatten_connectors(worker.connector):
                if connector.name == "MooncakeStoreConnector" and (
                    connector.options.load_async or connector.options.transfer_overlap
                ):
                    raise ConfigurationError(
                        "MigrationTax currently requires synchronous native store plans"
                    )
        self.counters, self.layers = counters, engine.model_spec.num_layers
        self.requests = {}
        self.original_dispatch = engine.dispatch_prefill_to_decode
        self.output = Path(output)
        self.stream = gzip.open(self.output / "native-migration-events.jsonl.gz", "xt")
        self.transfers = NativeTransfers(self)
        self.pipeline = NativeLayerPipeline(self)
        self.closed = False

    def log(self, kind, req=None, **values):
        row = dict(time_s=self.env.now, kind=kind, **values)
        if req is not None:
            row["source_id"] = str(req.id)
            row["branch"] = req.native_plan.branch
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.counters["migration_event:" + kind] += 1

    def on_route(self, req, worker, observation):
        if (
            req.native_plan.method != "MigrationTax"
            or req.native_plan.layers != self.layers
        ):
            raise ConfigurationError(
                "MigrationTax routing requires the native model layer count"
            )
        _, p, d = self.engine.router.routes[req.id]
        branch = req.native_plan.branch
        req.native_migration = SimpleNamespace(
            branch=branch,
            p=p,
            d=d,
            k=req.native_plan.p_outputs_before_handoff,
            destination_reserved=False,
            destination_input_keys=[],
            received_tokens=0,
            received={},
            fetched=set(),
            chunks=0,
            fetch_events=None,
            store_plans=[],
            store_leases=[],
            base_event=None,
            base_snapshot=None,
            tail_copy_s=0.0,
            owned_by_d=False,
            dispatch_started=False,
        )
        self.requests[req.id] = req
        self.env.process(self.retire(req))
        req.handover_store_limit = req.prefix_tokens
        observation["native_method_branch"] = branch
        observation["native_compute_worker"] = (
            d.id if branch == "decode_local_prefill" else p.id
        )
        self.log(
            "method_route",
            req,
            P=p.id,
            D=d.id,
            prefix_tokens=req.prefix_tokens,
            p_local_tokens=observation["p_local_tokens"],
            d_local_tokens=observation["d_local_tokens"],
        )
        if branch == "decode_local_prefill":
            self.admission.release_p(req)
            return d
        return p

    def ready_to_dispatch(self, worker, req):
        state = getattr(req, "native_migration", None)
        if state is None or state.branch != "prefill_side_handoff":
            return True
        return req.generation_idx >= state.k

    def is_pipeline_batch(self, worker, requests):
        if worker.role != "prefill" or not any(is_pipeline(r) for r in requests):
            return False
        if not all(is_pipeline(r) for r in requests):
            raise SimulationStateError(
                "Native FF and ordinary compute cohorts were mixed"
            )
        return True

    def retire(self, req):
        yield req.completion_event
        yield self.env.timeout(0)
        self.requests.pop(req.id, None)

    def advance(self, worker, req, latency, batch):
        state = getattr(req, "native_migration", None)
        if (
            state is None
            or state.branch != "prefill_side_handoff"
            or worker is not state.p
        ):
            return False
        before = req.generation_idx
        event, recording = req.completion_event, req.record_timing
        req.completion_event, req.record_timing = None, False
        req.advance(worker.env, latency, batch, worker.engine.timing_recorder)
        req.completion_event, req.record_timing = event, recording
        self.observe_sample(worker, req, before)
        if req.generation_idx > 0 and state.base_event is None:
            req.handover_prefill_finished_at = self.env.now
            worker.scheduler.block_manager.commit_input_cache(req)
            state.base_snapshot = pin_source(req, req.prefill_len)
            state.tail_copy_s = clone_hf_tail(self, req)
            state.base_event = self.env.process(self.base_handoff(req))
        if req.is_done:
            req.handover_output_finished_at = self.env.now
            req.handover_completion_pending = True
            # Native scheduler.update removes response-complete requests from
            # running, even while their physical pages await writeback. Start
            # the short-response handoff here so O=1/2 cannot lose dispatch.
            self.dispatch(worker, [req])
        return True

    def observe_sample(self, worker, req, before):
        if req.generation_idx != before:
            self.log(
                "native_output_sampled",
                req,
                worker=worker.id,
                role=worker.role,
                before=before,
                generated=req.generation_idx,
                computed_tokens=req.num_computed_tokens,
            )

    def before_compute(self, worker, requests):
        for req in requests:
            state = getattr(req, "native_migration", None)
            if state is not None and worker is state.p and state.tail_copy_s:
                duration, state.tail_copy_s = state.tail_copy_s, 0.0
                yield self.env.timeout(duration)
                self.log("HF_tail_COW_ready", req)

    def base_handoff(self, req):
        state = req.native_migration
        reserve_destination(self, req)
        blocks = max(
            0,
            ceil(req.prefill_len / req.block_size)
            - state.received_tokens // req.block_size,
        )
        yield from self.transfers.move(req, blocks, "HF_base")
        release_source(state.base_snapshot)
        state.base_snapshot = None
        self.log("HF_base_complete", req, tokens=req.prefill_len)

    def dispatch(self, worker, requests):
        ordinary = []
        for req in requests:
            state = getattr(req, "native_migration", None)
            if state is None:
                ordinary.append(req)
                continue
            if state.dispatch_started:
                raise SimulationStateError("Duplicate native MigrationTax dispatch")
            state.dispatch_started = True
            if state.branch == "prefill_side_handoff":
                self.env.process(self.finish_hf(req))
            elif state.branch == "layer_pipeline":
                # The native chunk already drained every layer send. Ownership
                # and D submission are synchronous; there is no event to wait.
                self.finish_ff(req)
            else:
                raise SimulationStateError("A D-local request reached P dispatch")
        if ordinary:
            self.original_dispatch(worker, ordinary)

    def finish_hf(self, req):
        state = req.native_migration
        if (
            req.generation_idx != state.k
            or req.num_computed_tokens != req.prefill_len + state.k - 1
        ):
            raise SimulationStateError(
                "HF native output or KV progress differs from its plan"
            )
        yield state.base_event
        if state.k > 1:
            end = req.prefill_len + state.k - 1
            blocks = ceil(end / req.block_size) - req.prefill_len // req.block_size
            yield from self.transfers.move(req, blocks, "HF_incremental")
        self.log(
            "HF_incremental_complete",
            req,
            valid_tokens=state.k - 1,
            computed_tokens=req.num_computed_tokens,
        )
        self.ownership(req)
        if req.is_done:
            self.env.process(self.completed_on_p(req))
        else:
            self.submit_decode(req)

    def finish_ff(self, req):
        state = req.native_migration
        if len(state.received) != self.layers or any(
            n != req.prefill_len for n in state.received.values()
        ):
            raise SimulationStateError("FF handoff before every native layer is ready")
        if req.generation_idx != 0 or req.num_computed_tokens != req.prefill_len:
            raise SimulationStateError(
                "FF P must compute input without sampling a client output"
            )
        self.ownership(req)
        self.submit_decode(req)

    def ownership(self, req):
        state = req.native_migration
        manager = state.p.scheduler.block_manager
        keys = build_prefix_keys(
            completed_hashes(req),
            model=manager.kv_cache_manager.model,
            cache_salt=req.cache_salt,
            reuse_group=req.reuse_group,
        )
        manager.kv_cache_manager.register_blocks(
            manager.block_table.get_blocks(req.id), keys
        )
        manager.release_request_blocks(req)
        state.p.connector.on_request_released(req)
        restore_destination(req)
        req.input_cache_keys = state.destination_input_keys
        req.input_cache_committed = False
        release_destination(req)
        state.owned_by_d = True
        state.store_plans.clear()
        state.fetch_events = None
        self.admission.release_p(req)
        wake_for_capacity(state.p)
        self.log(
            "ownership_D",
            req,
            generated=req.generation_idx,
            computed_tokens=req.num_computed_tokens,
            destination_blocks=state.d.scheduler.block_manager.block_table.get_num_blocks(
                req.id
            ),
        )

    def submit_decode(self, req):
        state = req.native_migration
        req.status = RequestStatus.WAITING_FOR_KV
        req.handoff_received_at = self.env.now
        state.d.scheduler.incoming_context[req.id] = (req.num_computed_tokens, 0)
        self.engine.send_task(state.d, Task.ADD, [req])

    def admit_destination(self, scheduler, req):
        state = getattr(req, "native_migration", None)
        if state is None or not state.destination_reserved or not state.owned_by_d:
            return False
        if scheduler is not state.d.scheduler:
            return False
        received, _ = scheduler.incoming_context.pop(req.id)
        restore_destination(req)
        req.start_context_build(received)
        req.status = RequestStatus.RUNNING
        self.log("native_D_admitted", req, received_tokens=received)
        return True

    def completed_on_p(self, req):
        worker = req.native_migration.d
        req.handover_writeback_started_at = self.env.now
        self.log("HF_completed_response_writeback_started", req)
        duration = scoped_completed_writeback(worker, req)
        self.log("HF_completed_response_writeback_issued", req, duration_s=duration)
        yield self.env.timeout(duration)
        finalize_completed_context(worker, req)
        self.log("HF_completed_response_writeback_ready", req)
        wake_for_capacity(worker)

    def finish(self):
        if not self.closed:
            self.stream.close()
            self.closed = True
        result = dict(
            timing="Native TokenSim operator composition and Mooncake TransferEngineSimulator",
            layers=self.layers,
            native_kv_geometry=dict(
                block_size=self.engine.config.block_size,
                token_bytes=int(
                    self.engine.request_workers[0].cache_config.size_per_token
                )
                * self.engine.parallel_config.tensor_parallel_size,
                token_bytes_per_rank=int(
                    self.engine.request_workers[0].cache_config.size_per_token
                ),
                tp_shards=self.engine.parallel_config.tensor_parallel_size,
            ),
            hf_k="min(2,O), from the untimed native plan",
            store_leases_remaining=len(self.transfers.store_leases),
            ff_chunk_barrier="All native layer sends finish before the next chunk; no cross-chunk overlap",
            batching_policy="Separate FF/ordinary cohorts only on P; D retains native mixed context/decode batches",
            method_semantics_version="native-migration-candidate-v2-native-D-batching",
            ff_native_layer_timing="One-layer native operator backends; embedding and step overhead occur once per native chunk; P emits no logits",
            admission="Native full-lifetime block budgets with two spare COW/boundary pages; conservative full-response P budget",
            native_metrics="Source Request times use real native callbacks; FF native compute service excludes layer waits, which remain in source elapsed time",
            unsupported=[
                "Disk-tier layer fetch",
                "Hybrid KV",
                "MoE",
                "Pipeline parallelism greater than one",
                "Asymmetric TP rank progress or uneven KV head sharding",
                "AF execution",
                "Async or overlapping native store plans",
            ],
        )
        (self.output / "native-migration-evidence.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result
