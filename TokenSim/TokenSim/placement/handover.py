"""Arrival-time routing for completion-clock handover experiments.

This policy does not replace the cache eviction clock with a trace-arrival clock.
Lowest worker ID resolves ties; partial prefix coverage remains explicitly visible.
"""

from collections import defaultdict
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.errors import ConfigurationError


class HandoverRouter:
    def __init__(self, prefill_workers, decode_workers, clock):
        self.prefill_workers = list(prefill_workers)
        self.decode_workers = list(decode_workers)
        self.clock = clock
        self.routes = {}
        self.previous = {}
        self.first_keys = {}
        self.observations = []
        self.arrival_state = None
        if not self.prefill_workers or not self.decode_workers:
            raise ConfigurationError("handover routing requires separate P and D pools")

    def resident_count(self, worker):
        manager = worker.scheduler.block_manager.kv_cache_manager
        # Count known conversation prefixes with their first block resident.
        # This is explicit block-cache bookkeeping, not a whole-object LRU model.
        return sum(
            manager._is_gpu_hit(manager._cache.get(key))
            for key in self.first_keys.values()
        )

    def reset_stats(self):
        """Reset profiling observations while retaining routes and affinity."""
        self.observations.clear()

    def _hits(self, worker, req):
        return worker.scheduler.block_manager.kv_cache_manager.plan_reuse(
            req
        ).hit_tokens

    def select(self, req):
        if req.id in self.routes:
            old_req, p, _ = self.routes[req.id]
            if old_req is not req:
                raise ConfigurationError("handover request IDs must be unique")
            return p
        prefix = req.prefix_tokens
        if (
            not req.chat_id
            or req.turn is None
            or type(prefix) is not int
            or not 0 <= prefix <= req.prefill_len
        ):
            raise ConfigurationError(
                "handover routing requires chat_id, turn and valid prefix_tokens"
            )
        prior = self.previous.get(req.chat_id)
        if (prior is None and req.turn != 0) or (prior and req.turn != prior[0] + 1):
            raise ConfigurationError(
                "handover conversations require contiguous, unique turns"
            )
        if req.turn == 0 and prefix:
            raise ConfigurationError("a first turn cannot inherit a prefix")
        if prefix and not req.hash_ids:
            raise ConfigurationError("prefix reuse requires explicit hash identities")
        if self.arrival_state is not None:
            return self.arrival_state.select(req, self, prior)
        hits = {w.id: self._hits(w, req) for w in self.prefill_workers}
        holders = [w for w in self.prefill_workers if prefix and hits[w.id] >= prefix]
        candidates = holders or self.prefill_workers
        p = min(candidates, key=lambda w: (self.resident_count(w), w.id))
        d = next((w for w in self.decode_workers if prior and w.id == prior[1]), None)
        if d is None:
            d = min(self.decode_workers, key=lambda w: (self.resident_count(w), w.id))
        self.routes[req.id] = (req, p, d)
        self.previous[req.chat_id] = (req.turn, d.id)
        if getattr(req, "handover_flow", False):
            from TokenSim.llm.handover_reservation import (
                reserve_destination,
                release_on_completion,
            )

            reserve_destination(req, d)
            d.env.process(release_on_completion(d.env, req))
        if req.hash_ids:
            self.first_keys[req.chat_id] = build_prefix_keys(
                req.hash_ids[:1],
                model=p.cache_config.model,
                cache_salt=req.cache_salt,
                reuse_group=req.reuse_group,
            )[0]
        self.observations.append(
            dict(
                request_id=req.id,
                conv_id=req.chat_id,
                turn=req.turn,
                arrival_decision_s=self.clock(),
                prefill_worker=p.id,
                decode_worker=d.id,
                prefix_tokens=prefix,
                p_local_tokens=hits[p.id],
                d_local_tokens=self._hits(d, req),
                partial_prefix=bool(prefix % req.block_size),
                clock="completion",
            )
        )
        return p

    def batches(self, requests):
        grouped = defaultdict(list)
        workers = {w.id: w for w in self.decode_workers}
        for req in requests:
            if req.id not in self.routes:
                raise ConfigurationError("D placement must be decided at arrival")
            grouped[self.routes[req.id][2].id].append(req)
        return [(workers[wid], batch) for wid, batch in grouped.items()]
