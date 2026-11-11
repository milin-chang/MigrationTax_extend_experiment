"""Construct physical TP replicas and validate their native catalog budgets."""

from TokenSim.config.config import ClusterConfig, WorkerGroupConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError


def configure_native_hardware(deployment, hardware):
    model = hardware.model(deployment.model_id)
    device = hardware.device(deployment.gpu_id)
    expected = dict(
        num_layers=deployment.layers,
        hidden_size=deployment.hidden_size,
        num_attention_heads=deployment.attention_heads,
        num_key_value_heads=deployment.kv_heads,
        head_dim=deployment.head_dim,
        dtype=deployment.model_dtype,
        kv_cache_dtype=deployment.kv_dtype,
    )
    if (
        any(getattr(model, key) != value for key, value in expected.items())
        or model.total_params() != deployment.model_parameters
    ):
        raise ConfigurationError(
            "Native model catalog disagrees with the shared hardware profile"
        )
    if (
        device.memory_capacity_bytes.value != deployment.gpu_memory_bytes
        or device.memory_reserved_bytes.value != deployment.runtime_reserved_bytes
    ):
        raise ConfigurationError(
            "Native GPU catalog disagrees with the shared hardware profile"
        )
    tp = deployment.tensor_parallel
    parallel = ParallelConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        data_parallel_size=(
            deployment.prefill_instances + deployment.decode_instances if tp > 1 else 1
        ),
    )
    if tp == 1:
        networks = {"net": "ethernet800Gb"}
        groups = [
            WorkerGroupConfig(
                role=role, hardware=deployment.gpu_id, num_workers=count, network="net"
            )
            for role, count in (
                ("prefill", deployment.prefill_instances),
                ("decode", deployment.decode_instances),
            )
        ]
    else:
        networks, groups = {}, []
        for role, count in (
            ("prefill", deployment.prefill_instances),
            ("decode", deployment.decode_instances),
        ):
            for index in range(count):
                node = f"{role}-{index}"
                networks[node] = "ethernet800Gb"
                groups.append(
                    WorkerGroupConfig(
                        role=role,
                        hardware=deployment.gpu_id,
                        num_workers=tp,
                        network=node,
                        name=node,
                    )
                )
    cluster = ClusterConfig(
        num_workers=deployment.total_gpus,
        parallel_config=parallel,
        networks=networks,
        worker_groups=groups,
    )
    return model, parallel, cluster


def validate_native_caches(engine, deployment):
    for worker in engine.request_workers:
        cache = worker.cache_config
        if (cache.num_gpu_blocks, cache.size_per_token, cache.model_param_size) != (
            deployment.kv_blocks,
            deployment.kv_bytes_per_token_per_gpu,
            deployment.weights_bytes_per_gpu,
        ):
            raise ConfigurationError(
                "Native cache capacity disagrees with the shared hardware profile"
            )
