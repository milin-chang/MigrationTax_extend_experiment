"""Run native LLMEngine with completion-clock KV and native operator timing."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import simpy
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.hardware import HardwareContext
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.reporting import RunReport, export_result
from algorithm.native import build_plan
from algorithm.model import Turn, CacheState
from native_methods.identities import Span, block_hashes, input_spans
from native_methods.hardware_profile import load_profile
from TokenSim.research.native_hardware import (
    configure_native_hardware,
    validate_native_caches,
)
from TokenSim.llm.handover_parallel import tp_group_summary
from TokenSim.research.native_admission import NativeAdmission
from TokenSim.research.native_retirement import retire_identity
from TokenSim.research.native_migration import install as install_migration
from TokenSim.research.native_study import (
    load_study,
    study_cluster,
    study_connector,
    install_observers,
    observe_arrival,
    request_observation,
)


def execute(
    rows,
    method,
    seed,
    output,
    *,
    probe=None,
    hardware_profile=None,
    experiment_config=None,
    record_timeline=False,
):
    if method not in ("Serial", "PPD", "MigrationTax"):
        raise ValueError("Unknown native method: " + str(method))
    if type(record_timeline) is not bool:
        raise ValueError("record_timeline must be boolean")
    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    env = simpy.Environment()
    deployment = load_profile(hardware_profile)
    study = load_study(experiment_config)
    hardware = HardwareContext.load(study["data_root"] if study else root / "data")
    model, parallel, cluster = configure_native_hardware(deployment, hardware)
    if study:
        cluster = study_cluster(study, parallel)
    connector = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_connector_extra_config": {
                        "bandwidth_bytes_per_s": 100e9,
                        "fixed_latency_us": 0,
                    },
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_connector_extra_config": {
                        "mode": "standalone-store",
                        "save_policy": "turn_end",
                        "global_segment_size": 2**39,
                        "local_buffer_size": 2**38,
                        "host_available_bytes": 2**42,
                        "admission_policy": "always",
                        "eviction_policy": "lru",
                    },
                },
            ]
        },
    )
    if study:
        connector = study_connector(study)
    config = SimulationConfig(
        batching="paged-attn",
        block_size=deployment.block_tokens,
        gpu_memory_utilization=1.0,
        max_num_seqs=64 if study else 16,
        max_num_batched_tokens=8192 if study else 4096,
        preserve_p2p_kv=True,
        handover_flow=True,
        handover_tp_groups=deployment.tensor_parallel > 1,
        handover_clock="completion",
        handover_cost_model="native",
        latency_backend_type="operator_table" if study else "analytical",
        operator_backend="vllm" if study else None,
        random_seed=seed,
    )
    engine = LLMEngine(
        env=env,
        hardware=hardware,
        model_spec=model,
        cluster_config=cluster,
        parallel_config=parallel,
        kv_transfer_config=connector,
        config=config,
        workload_config=WorkloadConfig(distribution="burst"),
    )
    admission = NativeAdmission(engine)
    validate_native_caches(engine, deployment)
    counters = Counter()
    migration = (
        install_migration(engine, admission, output, counters)
        if method == "MigrationTax"
        else None
    )
    if study:
        install_observers(engine, migration)
    if probe is not None:
        probe(engine, admission, env)
    spans = input_spans(rows)
    requests = {}
    completed = []
    rejected = []
    failed_sources = set()
    ordinal = Counter()
    by_id = {r["source_id"]: r for r in rows}
    original_select = engine.router.select

    def select(req):
        worker = original_select(req)
        obs = engine.router.observations[-1]
        k = req.prefix_tokens
        state = ("H" if k and obs["p_local_tokens"] >= k else "F") + (
            "H" if k and obs["d_local_tokens"] >= k else "F"
        )
        row = by_id[str(req.id)]
        req.native_plan = build_plan(
            method,
            Turn(
                req.prefill_len,
                row["new_tokens"],
                CacheState(state),
                req.decode_len,
                row["turn"] > 0,
                str(req.id),
            ),
            layers=engine.model_spec.num_layers,
        )
        counters["untimed_plans"] += 1
        if migration is not None:
            return migration.on_route(req, worker, obs)
        if req.native_plan.branch == "decode_local_prefill":
            target = engine.router.routes[req.id][2]
            obs["native_compute_worker"] = target.id
            obs["native_method_branch"] = req.native_plan.branch
            counters["D_local_prefill_submissions"] += 1
            admission.release_p(req)
            return target
        obs["native_compute_worker"] = worker.id
        obs["native_method_branch"] = req.native_plan.branch
        return worker

    engine.router.select = select
    engine.router.describe = lambda: dict(
        type="HandoverRouter",
        clock="completion",
        decisions=len(engine.router.observations),
        routing="ready prefix holders / persistent decode affinity",
    )
    original_compute = engine.run_complete_compute

    def compute(*args, **kwargs):
        counters["LLMEngine.run_complete_compute"] += 1
        worker_id = args[0] if args else kwargs["worker_id"]
        counters["native_worker_compute:" + str(worker_id)] += 1
        return original_compute(*args, **kwargs)

    engine.run_complete_compute = compute
    log = (output / "requests.jsonl").open("x")
    timeline = None
    if record_timeline:
        from TokenSim.research.native_timeline import NativeTimeline

        timeline = NativeTimeline(engine, output, method)
    timeline_status = "failed"
    started = time.monotonic()
    effective = dict(
        method=method,
        seed=seed,
        hardware_alignment=deployment.describe(),
        tensor_parallel_handover=tp_group_summary(engine),
        model=asdict(model),
        simulation=asdict(config),
        cluster=asdict(cluster),
        transfer=asdict(connector),
        timing_backend=(
            "H100 vLLM measured operator tables with native interpolation/extrapolation/fallback"
            if study
            else "TokenSim native operator analytical fallback; not algorithm PF/ST"
        ),
        context_extrapolation="Source lengths retained beyond model training context; timing extrapolation only",
        predecessor_rule="Original arrival retained; native submission waits for actual source parent completion",
        native_turn_rule="Router ordinal per conversation; original source_turn and exact parent retained",
        store_assumption=(
            "Provided embedded DRAM store, 10,000,000 blocks, 1 GiB buffer/worker; no offload"
            if study
            else "Finite 512 GiB pool, 256 GiB staging/worker and 4 TiB shared host budget; experimental allocation"
        ),
        ppd_lowering="D-hit plan submits original request directly to its native D engine; native prefill samples first output and subsequent native decode completes the response",
        migration_lowering=(
            "Native P sampling through k=min(2,O), base/tail KV sync and D ownership; FF native per-layer operators and Mooncake transfer events, all-layer chunk barrier"
            if migration is not None
            else "not installed"
        ),
        model_choice=f"Shared {deployment.model_id}/{deployment.model_dtype}, H100 SXM, TP{deployment.tensor_parallel}, PP1, 4P+4D; timing-only beyond catalog context range",
        admission=admission.describe(),
        memory_policy="Retire completed request-local identity lists after full writeback and native release callbacks; preserve cache/store ownership and native metric scalars",
        bandwidth_study=study,
        request_timeline_enabled=record_timeline,
        hardware_memory_bandwidth_bytes_per_s=hardware.device(
            deployment.gpu_id
        ).memory_bandwidth_bytes_per_s.value,
        communication_link=(
            asdict(hardware.link("bandwidth_study_link")) if study else None
        ),
    )
    (output / "effective-config.json").write_text(
        json.dumps(effective, indent=2, default=str) + "\n"
    )

    def arrival(row):
        yield env.timeout(row["arrival_seconds"])
        cid = str(row["conversation"])
        rid = row["source_id"]
        nt = 0
        req = Request(
            id=int(rid),
            prefill_len=row["context_tokens"],
            decode_len=row["output_tokens"],
            block_size=deployment.block_tokens,
            hash_ids=block_hashes(spans[rid], deployment.block_tokens),
            chat_id=cid,
            turn=nt,
            prefix_tokens=row["context_tokens"] - row["new_tokens"],
        )
        req.handover_context_hash_ids = block_hashes(
            spans[rid] + [Span("output:" + rid, 0, row["output_tokens"] - 1)],
            deployment.block_tokens,
        )
        req.arrive(env)
        requests[rid] = req
        if timeline is not None:
            timeline.arrival(req, row)
        if study:
            observe_arrival(engine, admission, req)
        parent = row["previous_source_id"]
        if parent is not None:
            if parent not in requests:
                raise RuntimeError("Explicit parent has not arrived")
            pr = requests[parent]
            if pr.completion_event is not None and not pr.completion_event.triggered:
                yield pr.completion_event
            if parent in failed_sources:
                failed_sources.add(rid)
                rejected.append(
                    dict(source_id=rid, status="ancestor_failed", parent=parent)
                )
                req.completion_event.succeed(None)
                if timeline is not None:
                    timeline.completed(req, "ancestor_failed", "Explicit parent failed")
                return
        if timeline is not None:
            timeline.ready(req)
        try:
            engine.validate_request_capacity([req])
        except Exception as e:
            failed_sources.add(rid)
            rejected.append(
                dict(
                    source_id=rid,
                    source_turn=row["turn"],
                    arrival_s=row["arrival_seconds"],
                    status="rejected",
                    reason=str(e),
                )
            )
            if not req.completion_event.triggered:
                req.completion_event.succeed(None)
            if timeline is not None:
                timeline.completed(req, "rejected", str(e))
            return
        yield from admission.acquire(req, method)
        if timeline is not None:
            timeline.submitted(req)
        nt = ordinal[cid]
        ordinal[cid] += 1
        req.turn = nt
        engine.add_requests([req])
        counters["LLMEngine.add_requests"] += 1
        yield req.completion_event
        # The native failure path can wake completion listeners to stop the
        # engine. Such a wake is not a successful response or safe retirement.
        engine.raise_if_failed()
        admission.release(req)
        completed.append(req)
        if timeline is not None:
            timeline.completed(req)
        log.write(
            json.dumps(
                dict(
                    source_id=rid,
                    source_turn=row["turn"],
                    native_turn=nt,
                    previous_source_id=parent,
                    arrival_s=row["arrival_seconds"],
                    completed_at=req.completed_at,
                    status="completed",
                    plan_branch=req.native_plan.branch,
                    generated=req.generation_idx,
                    native_prefill_latency_s=req.prefill_latency,
                    native_total_time_s=req.total_time,
                    native_decode_time_sum_s=req.decode_time_sum,
                    native_budget_wait_s=req.native_budget_wait_s,
                    **(request_observation(engine, req) if study else {}),
                )
            )
            + "\n"
        )
        log.flush()
        env.process(retire_identity(engine, req, counters))
        if len(completed) % 100 == 0:
            (output / "progress.json").write_text(
                json.dumps(
                    dict(
                        completed=len(completed),
                        rejected=len(rejected),
                        simulated_time_s=env.now,
                        counters=dict(counters),
                    )
                )
                + "\n"
            )

    for row in rows:
        env.process(arrival(row))
    try:
        env.run()
        engine.raise_if_failed()
        if migration is not None:
            migration.finish()
        reject_ids = {x["source_id"] for x in rejected}
        complete_ids = {str(r.id) for r in completed}
        unfinished = [
            r.id
            for r in requests.values()
            if str(r.id) not in complete_ids and str(r.id) not in reject_ids
        ]
        offered = (
            (len(rows) - 1) / (rows[-1]["arrival_seconds"] - rows[0]["arrival_seconds"])
            if len(rows) > 1
            and rows[-1]["arrival_seconds"] > rows[0]["arrival_seconds"]
            else 0.0
        )
        end_state = []
        for worker in engine.workers:
            scheduler = worker.scheduler
            end_state.append(
                dict(
                    id=worker.id,
                    role=worker.role,
                    status=str(worker.status),
                    gpu_status=scheduler.block_manager.get_gpu_status(),
                    waiting=[r.id for r in scheduler.waiting],
                    running=[
                        dict(
                            id=r.id,
                            computed=r.num_computed_tokens,
                            generated=r.generation_idx,
                            context=r.context_len,
                            status=str(r.status),
                        )
                        for r in scheduler.running
                    ],
                    loading=list(scheduler.loading),
                    incoming=list(scheduler.incoming_context),
                    pinned=list(getattr(worker, "pinned_handoffs", [])),
                    stats=scheduler.stats,
                )
            )
        (output / "native-end-state.json").write_text(
            json.dumps(end_state, indent=2, default=str) + "\n"
        )
        (output / "native-admission-budget.json").write_text(
            json.dumps(admission.describe(), indent=2) + "\n"
        )
        with (output / "native-route-decisions.jsonl").open("x") as routes:
            for observation in engine.router.observations:
                routes.write(json.dumps(observation, default=str) + "\n")
        export_result(
            run=RunReport(output / "native", offered, config.batching),
            requests=completed,
            cluster=cluster,
            request_count=len(completed),
            g_time=engine.timing_recorder.timing,
            engine=engine,
            prefill_lens=[r.prefill_len for r in completed],
            decode_lens=[r.generation_idx for r in completed],
            notdone=unfinished,
            duration=env.now,
            simulator_wall_time=time.monotonic() - started,
        )
        result = dict(
            status=(
                "completed"
                if len(completed) == len(rows) and not rejected and not unfinished
                else "incomplete"
            ),
            input_requests=len(rows),
            arrived=len(requests),
            completed=len(completed),
            rejected=rejected,
            unfinished=unfinished,
            native_scheduler_executed=True,
            native_counters=dict(counters),
            simulated_end_s=env.now,
            method=method,
        )
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        timeline_status = result["status"]
        return result
    finally:
        if migration is not None:
            migration.finish()
        log.close()
        if timeline is not None:
            timeline.close(timeline_status)
