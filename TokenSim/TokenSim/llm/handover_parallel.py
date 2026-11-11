"""Opt-in symmetric tensor-parallel replicas for completion-clock handover.

One leader schedules each replica, matching the native ordinary TP engine.
Uniform ranks share a canonical logical page ledger; each physical rank has
its own native capacity and operator geometry. KV I/O moves the complete TP
bundle over the configured replica endpoint link. This aggregate-link model
does not model independent per-rank NICs or asymmetric rank progress.
"""

from collections import defaultdict

from TokenSim.errors import ConfigurationError


def validate_handover_parallel(config, parallel, cluster, model):
    if type(config.handover_tp_groups) is not bool:
        raise ConfigurationError("handover_tp_groups must be boolean")
    if config.handover_tp_groups:
        if not (
            config.handover_flow and config.preserve_p2p_kv and config.handover_routing
        ):
            raise ConfigurationError(
                "TP handover requires completion-flow routing and persistent KV"
            )
        if (
            parallel is None
            or parallel.tensor_parallel_size < 2
            or parallel.pipeline_parallel_size != 1
            or config.dp_lockstep
            or cluster.af.enabled
            or config.batching != "paged-attn"
            or config.handover_clock != "completion"
        ):
            raise ConfigurationError(
                "TP handover requires TP>=2, PP1, independent replicas and ordinary completion-clock paged workers"
            )
        tp = parallel.tensor_parallel_size
        if (
            model.components
            or model.is_moe
            or model.attention_geometries
            or model.num_key_value_heads % tp
            or model.num_attention_heads % tp
            or model.intermediate_size % tp
            or model.dtype not in ("fp16", "bf16")
            or model.kv_cache_dtype not in ("fp16", "bf16")
        ):
            raise ConfigurationError(
                "TP handover requires uniform dense, evenly sharded FP16/BF16 model and KV geometry"
            )
        return
    if (config.preserve_p2p_kv or config.handover_routing) and (
        config.batching != "paged-attn"
        or cluster.af.enabled
        or (parallel is not None and parallel.world_size != 1)
    ):
        raise ConfigurationError(
            "preserve_p2p_kv requires ordinary paged workers with TP=PP=DP=1; symmetric TP replicas require handover_tp_groups=True"
        )


def configure_transfer_geometry(worker):
    if worker.engine.config.handover_tp_groups:
        worker.cache_config.handover_tp_size = (
            worker.parallel_config.tensor_parallel_size
        )


def bind_tp_groups(engine):
    if not engine.config.handover_tp_groups:
        return
    from TokenSim.kv_transfer.connectors.base import flatten_connectors

    groups = defaultdict(list)
    for worker in engine.request_workers:
        groups[worker.dp_rank].append(worker)
    engine.handover_tp_replicas = {}
    for replica, workers in sorted(groups.items()):
        workers.sort(key=lambda worker: worker.tp_rank)
        leader = workers[0]
        tp = engine.parallel_config.tensor_parallel_size
        if len(workers) != tp or [w.tp_rank for w in workers] != list(range(tp)):
            raise ConfigurationError("Incomplete TP replica rank inventory")
        if len({w.role for w in workers}) != 1 or leader.role not in (
            "prefill",
            "decode",
        ):
            raise ConfigurationError(
                "A TP replica must contain one complete P or D role"
            )
        geometry = lambda w: (
            w.cache_config.num_gpu_blocks,
            w.cache_config.size_per_token,
            w.cache_config.model_param_size,
            w.hardware,
            w.network,
        )
        if any(geometry(w) != geometry(leader) for w in workers):
            raise ConfigurationError(
                "Asymmetric TP rank memory or device geometry is unsupported"
            )
        for worker in workers:
            if any(
                c.name
                not in (
                    "MultiConnector",
                    "MooncakeConnector",
                    "MooncakeStoreConnector",
                    "P2PConnector",
                    "NoopConnector",
                )
                for c in flatten_connectors(worker.connector)
            ):
                raise ConfigurationError(
                    "TP handover needs connectors with aggregate KV payload accounting"
                )
            worker.tp_group_leader_id = leader.id
            worker.tp_group_worker_ids = tuple(w.id for w in workers)
            worker.scheduler.block_manager = leader.scheduler.block_manager
        engine.handover_tp_replicas[replica] = tuple(w.id for w in workers)


def tp_group_summary(engine):
    groups = getattr(engine, "handover_tp_replicas", {})
    return dict(
        enabled=engine.config.handover_tp_groups,
        representation="One native scheduling leader and canonical symmetric KV ledger per TP replica",
        io_model="Full TP KV bundle over one configured replica endpoint link; completion waits for the complete bundle",
        physical_gpus=len(engine.workers),
        replica_ranks={str(key): list(value) for key, value in groups.items()},
    )
