"""Explicit bandwidth-study configuration and read-only endpoint observations."""

from functools import wraps
from hashlib import sha256
import json
from pathlib import Path

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]


def load_study(path):
    if path is None:
        return None
    path = Path(path)
    data = json.loads(path.read_text())
    if data["bandwidth_bytes_per_s"] <= 0:
        raise ConfigurationError("Study bandwidth must be positive")
    for key in ("data_root", "cluster_path", "transfer_path"):
        target = (ROOT / data[key]).resolve()
        if not target.is_relative_to(ROOT) or not target.exists():
            raise ConfigurationError("Invalid study path: " + key)
        data[key] = str(target)
    data["config_sha256"] = sha256(path.read_bytes()).hexdigest()
    return data


def study_cluster(study, parallel):
    raw = json.loads(Path(study["cluster_path"]).read_text())
    raw["worker_groups"] = [
        WorkerGroupConfig(**group) for group in raw["worker_groups"]
    ]
    return ClusterConfig(**raw, parallel_config=parallel)


def study_connector(study):
    raw = json.loads(Path(study["transfer_path"]).read_text())
    return KVTransferConfig.from_mapping(raw)


def install_observers(engine, migration):
    """Observe real output advances without changing any simulated duration."""
    for worker in engine.request_workers:
        original = worker.step

        @wraps(original)
        def step(requests, latency, *, worker=worker, original=original):
            before = {req.id: req.generation_idx for req in requests}
            result = original(requests, latency)
            if worker.role == "decode":
                for req in requests:
                    if req.generation_idx > before[req.id]:
                        if not hasattr(req, "study_first_d_sample_at"):
                            req.study_first_d_sample_at = worker.env.now
                        if not hasattr(req, "study_first_d_visible_at"):
                            req.study_first_d_visible_at = worker.env.now
            return result

        worker.step = step
    if migration is not None:
        original_ownership = migration.ownership

        def ownership(req):
            result = original_ownership(req)
            # HF can have sampled client outputs on P before the real KV
            # ownership barrier. Record D availability at that barrier and
            # retain the distinct first token actually sampled on D.
            if req.generation_idx and not hasattr(req, "study_first_d_visible_at"):
                req.study_first_d_visible_at = engine.env.now
            return result

        migration.ownership = ownership


def observe_arrival(engine, admission, req):
    """Snapshot every destination at the original trace arrival, before waits."""
    req.study_arrival_d_local_tokens = {
        str(worker.id): admission.router._hits(worker, req)
        for worker in admission.router.decode_workers
    }
    req.study_arrival_p_local_tokens = {
        str(worker.id): admission.router._hits(worker, req)
        for worker in admission.router.prefill_workers
    }
    req.study_d_choice_at_arrival = admission.choose_d(req).id


def request_observation(engine, req):
    _, p, d = engine.router.routes[req.id]
    visible = getattr(req, "study_first_d_visible_at", None)
    sample = getattr(req, "study_first_d_sample_at", None)
    if visible is None:
        raise ConfigurationError("Completed request has no first output observed at D")
    return dict(
        first_client_output_at_s=req.arrival_at + req.prefill_latency,
        first_d_output_visible_at_s=visible,
        first_d_output_sampled_at_s=sample,
        d_output_visible_latency_s=visible - req.arrival_at,
        d_output_sampled_latency_s=None if sample is None else sample - req.arrival_at,
        actual_prefill_worker=p.id,
        actual_decode_worker=d.id,
        inherited_prefix_tokens=req.prefix_tokens,
        d_local_tokens_at_original_arrival=req.study_arrival_d_local_tokens[str(d.id)],
        p_local_tokens_at_original_arrival=req.study_arrival_p_local_tokens[str(p.id)],
        decode_miss_at_original_arrival=bool(
            req.prefix_tokens
            and req.study_arrival_d_local_tokens[str(d.id)] < req.prefix_tokens
        ),
        decode_choice_at_original_arrival=req.study_d_choice_at_arrival,
        decode_choice_changed_before_admission=req.study_d_choice_at_arrival != d.id,
        d_endpoint_definition="First native D sample, or completed HF output/KV ownership barrier for P-generated outputs",
    )
