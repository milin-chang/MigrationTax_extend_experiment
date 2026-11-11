"""Arrival-ordered object LRU with real readiness events and pinned snapshots.

Logical placement and eviction advance at trace arrivals, independently of
service speed. A captured version still waits for its real producer; this is
not label replay or permission to use a KV tensor before it exists.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.errors import ConfigurationError, SimulationStateError


@dataclass
class CacheVersion:
    request: object
    tokens: int
    ready: object
    worker: object = None
    resident: bool = True
    readers: int = 0
    blocks: tuple = ()
    pinned: bool = False
    pinned_count: int = 0
    reader_lengths: list = field(default_factory=list)

    def retain(self, tokens):
        self.reader_lengths.append(tokens)
        self.readers = len(self.reader_lengths)

    def release(self, tokens):
        self.reader_lengths.remove(tokens)
        self.readers = len(self.reader_lengths)
        if self.readers < 0:
            raise SimulationStateError("negative handover snapshot references")
        self._unpin_if_unused()

    def evict(self):
        self.resident = False
        self._unpin_if_unused()

    def _unpin_if_unused(self):
        needed = self.tokens if self.resident else max(self.reader_lengths, default=0)
        if self.pinned_count > needed:
            allocator = self.worker.scheduler.block_manager.gpu_allocator
            for block in self.blocks[needed : self.pinned_count]:
                allocator.free(block)
            self.pinned_count = needed
            self.pinned = needed > 0
            from TokenSim.llm.handover_reservation import wake_for_capacity

            wake_for_capacity(self.worker)

    def publish(self):
        if self.worker is not None:
            manager = self.worker.scheduler.block_manager
            self.blocks = tuple(
                manager.block_table.get_blocks(self.request.id)[: self.tokens]
            )
            if len(self.blocks) != self.tokens:
                raise SimulationStateError(
                    "published KV is shorter than the arrival-clock version"
                )
            if self.resident or self.readers:
                needed = self.tokens if self.resident else max(self.reader_lengths)
                for block in self.blocks[:needed]:
                    manager.gpu_allocator.acquire(block)
                self.pinned = True
                self.pinned_count = needed
        if not self.ready.triggered:
            self.ready.succeed(self)


class ArrivalCacheState:
    def __init__(self, engine):
        if not engine.config.handover_flow or engine.config.block_size != 1:
            raise ConfigurationError(
                "arrival-clock flow requires handover_flow and block_size=1 for exact token coverage"
            )
        self.engine = engine
        self.p = list(engine.prefill_workers.workers)
        self.d = list(engine.decode_workers.workers)
        self.caches = {w.id: OrderedDict() for w in self.p + self.d}
        self.store = {}
        self.versions = {}
        self.observations = []
        self.last_prefill = {}
        self.last_decode = {}
        self.previous_context = {}
        if engine.config.max_num_seqs != 1:
            raise ConfigurationError(
                "arrival-clock flow requires serial max_num_seqs=1"
            )
        self.store_enabled = any(
            c.name == "MooncakeStoreConnector" and c.options.admission_policy != "never"
            for c in flatten_connectors(self.p[0].connector)
        )
        for worker in self.p + self.d:
            worker.scheduler.block_manager.kv_cache_manager.handover_worker_id = (
                worker.id
            )

    def _coverage(self, worker, req):
        entry = self.caches[worker.id].get(req.chat_id)
        if entry is None:
            return None, 0
        n = min(req.prefix_tokens, entry.tokens)
        old = getattr(
            entry.request, "handover_context_hash_ids", entry.request.hash_ids
        )
        if old[:n] != req.hash_ids[:n]:
            n = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(old[:n], req.hash_ids[:n]))
                    if a != b
                ),
                n,
            )
        return entry, n

    def _install(self, worker, req, tokens):
        manager = worker.scheduler.block_manager
        capacity = manager.num_total_gpu_blocks - manager.watermark_blocks
        if tokens > capacity:
            raise ConfigurationError(
                "complete pinned turn exceeds the device KV capacity"
            )
        cache = self.caches[worker.id]
        previous = cache.pop(req.chat_id, None)
        if previous is not None:
            previous.evict()
        while sum(x.tokens for x in cache.values()) + tokens > capacity:
            _, old = cache.popitem(last=False)
            old.evict()
        entry = CacheVersion(req, tokens, self.engine.env.event(), worker)
        cache[req.chat_id] = entry
        self.versions[(req.id, worker.role)] = entry

    def select(self, req, router, prior):
        if len(req.hash_ids or []) != req.prefill_len:
            raise ConfigurationError(
                "arrival-clock flow requires one explicit identity per input token"
            )
        full = getattr(req, "handover_context_hash_ids", None)
        if full is None or len(full) != req.prefill_len + req.decode_len - 1:
            raise ConfigurationError(
                "arrival-clock flow requires identities for every materialized context token"
            )
        if req.turn:
            parent = self.previous_context[req.chat_id]
            if (
                req.prefix_tokens > len(parent)
                or parent[: req.prefix_tokens] != req.hash_ids[: req.prefix_tokens]
            ):
                raise ConfigurationError(
                    "arrival-clock conversation ancestry does not match the preceding turn"
                )
        self.previous_context[req.chat_id] = full
        coverage = {w.id: self._coverage(w, req) for w in self.p}
        holders = [
            w
            for w in self.p
            if req.prefix_tokens and coverage[w.id][1] == req.prefix_tokens
        ]
        p = min(holders or self.p, key=lambda w: (len(self.caches[w.id]), w.id))
        d = next((w for w in self.d if prior and w.id == prior[1]), None)
        d = d or min(self.d, key=lambda w: (len(self.caches[w.id]), w.id))
        p_entry, p_tokens = coverage[p.id]
        d_entry, d_tokens = self._coverage(d, req)
        stored = self.store.get(req.chat_id)
        store_tokens = (
            max(0, min(req.prefix_tokens, stored.tokens) - p_tokens) if stored else 0
        )
        held = []
        dependencies = []
        for entry, n in ((p_entry, p_tokens), (d_entry, d_tokens)):
            if n:
                entry.retain(n)
                held.append((entry, n))
                dependencies.append(entry.ready)
        if store_tokens:
            dependencies.append(stored.ready)
        req.handover_snapshots = held
        req.handover_dependencies = dependencies
        if p.id in self.last_prefill:
            dependencies.append(self.last_prefill[p.id])
        req.handover_decode_predecessor = self.last_decode.get(d.id)
        self.last_decode[d.id] = req.completion_event
        req.handover_cache_limits = {p.id: p_tokens, d.id: d_tokens}
        req.handover_store_limit = p_tokens + store_tokens
        req.handover_d_cached_tokens = d_tokens
        router.routes[req.id] = (req, p, d)
        router.previous[req.chat_id] = (req.turn, d.id)
        observation = dict(
            request_id=req.id,
            conv_id=req.chat_id,
            turn=req.turn,
            arrival_decision_s=self.engine.env.now,
            prefill_worker=p.id,
            decode_worker=d.id,
            prefix_tokens=req.prefix_tokens,
            p_local_tokens=p_tokens,
            p_store_tokens=store_tokens,
            d_local_tokens=d_tokens,
            partial_prefix=False,
            clock="arrival",
        )
        router.observations.append(observation)
        self.observations.append(observation)
        self._install(p, req, req.prefill_len)
        self.last_prefill[p.id] = self.versions[(req.id, "prefill")].ready
        self._install(d, req, len(full))
        store_version = CacheVersion(req, len(full), self.engine.env.event())
        self.versions[(req.id, "store")] = store_version
        if self.store_enabled:
            self.store[req.chat_id] = store_version
        self.engine.env.process(self._cleanup_on_completion(req))
        return p

    def _cleanup_on_completion(self, req):
        yield req.completion_event
        for entry, n in getattr(req, "handover_snapshots", [])[:]:
            entry.release(n)
            req.handover_snapshots.remove((entry, n))
        if not req.cancel_requested:
            return
        for phase in ("prefill", "decode", "store"):
            entry = self.versions[(req.id, phase)]
            entry.evict()
            cache = (
                self.caches[entry.worker.id] if entry.worker is not None else self.store
            )
            if cache.get(req.chat_id) is entry:
                del cache[req.chat_id]
            if not entry.ready.triggered:
                entry.ready.fail(
                    SimulationStateError("a required handover producer was cancelled")
                )
                entry.ready.defused = True

    def publish(self, worker, req):
        self.versions[(req.id, worker.role)].publish()

    def store_ready(self, req):
        self.versions[(req.id, "store")].publish()


def setup_arrival_clock(engine):
    if engine.config.handover_flow:
        for worker in engine.request_workers:
            worker.scheduler.block_manager.watermark_blocks = 0
            worker.scheduler.block_manager.handover_reserve_context = (
                worker.role == "decode"
            )
    if engine.config.handover_clock == "arrival":
        engine.router.arrival_state = ArrivalCacheState(engine)


def send_after_dependencies(engine, worker, batch, task):
    if not engine.config.handover_flow:
        engine.send_task(worker, task, batch)
        return
    for req in batch:
        if (
            not getattr(req, "handover_dependencies", None)
            and not engine.config.handover_fixed_overhead_s
        ):
            engine.send_task(worker, task, [req])
            continue
        engine.env.process(_wait_and_send(engine, worker, req, task))


def _wait_and_send(engine, worker, req, task):
    try:
        events = list(getattr(req, "handover_dependencies", []))
        delay = max(
            0, req.arrival_at + engine.config.handover_fixed_overhead_s - engine.env.now
        )
        if delay:
            events.append(engine.env.timeout(delay))
        yield engine.env.all_of(events)
    except SimulationStateError as error:
        engine.record_worker_failure(worker.id, error)
        engine._broadcast_stop()
        return
    if not req.cancel_requested:
        engine.send_task(worker, task, [req])


def release_snapshot(req, worker_id):
    held = getattr(req, "handover_snapshots", [])
    for entry, n in held[:]:
        if entry.worker.id == worker_id:
            entry.release(n)
            held.remove((entry, n))


def limit_reuse(manager, req, plan):
    limits = getattr(req, "handover_cache_limits", {})
    worker_id = getattr(manager, "handover_worker_id", None)
    if worker_id not in limits:
        return plan
    from dataclasses import replace

    tokens = limits[worker_id]
    if plan.hit_tokens < tokens:
        raise SimulationStateError(
            "a pinned arrival-clock prefix disappeared before admission"
        )
    return replace(
        plan,
        hit_blocks=plan.hit_blocks[:tokens],
        hit_tokens=tokens,
        effective_prefill_tokens=req.prefill_len - tokens,
    )


def send_decode_in_arrival_order(engine, worker, batch, metadata):
    from TokenSim.llm.engine.messages import Task

    if engine.config.handover_clock != "arrival":
        engine.send_task(worker, Task.ADD, batch, metadata)
        return
    if len(batch) != 1:
        raise ConfigurationError(
            "arrival-order decode handoff must contain one request"
        )
    engine.env.process(_ordered_decode(engine, worker, batch[0], metadata))


def _ordered_decode(engine, worker, req, metadata):
    from TokenSim.llm.engine.messages import Task

    predecessor = req.handover_decode_predecessor
    if predecessor is not None:
        yield predecessor
    if not req.cancel_requested:
        req.handover_decode_ticket_ready_at = engine.env.now
        engine.send_task(worker, Task.ADD, [req], metadata)
