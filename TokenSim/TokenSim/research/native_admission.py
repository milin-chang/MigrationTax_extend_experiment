"""Conservative live-block budgets before native handover prefix pinning.

Native allocators still allocate, evict, share and COW the real blocks. These
budget tokens only prevent queued prefix readers from pinning every D page
before any complete request can be admitted. No simulated duration is supplied.
"""

from math import ceil
import simpy
from TokenSim.placement.handover import HandoverRouter
from TokenSim.llm.handover_reservation import reserve_destination


class BudgetRouter(HandoverRouter):
    def select(self, req):
        # Fix the placement selected by the live-capacity gate, then execute
        # original native validation, cache lookup, prefix leases and affinity.
        original_p, original_d = self.prefill_workers, self.decode_workers
        self.prefill_workers = [req.native_budget_p]
        self.decode_workers = [req.native_budget_d]
        try:
            return super().select(req)
        finally:
            self.prefill_workers, self.decode_workers = original_p, original_d


class NativeAdmission:
    def __init__(self, engine):
        self.engine, self.env = engine, engine.env
        original = engine.router
        self.router = BudgetRouter(
            original.prefill_workers, original.decode_workers, original.clock
        )
        self.budgets = {}
        self.peak = {}
        for worker in self.router.prefill_workers + self.router.decode_workers:
            manager = worker.scheduler.block_manager
            capacity = manager.num_total_gpu_blocks - manager.watermark_blocks
            self.budgets[worker.id] = simpy.Container(self.env, capacity, capacity)
            self.peak[worker.id] = 0
        engine.router = self.router

    def occupied(self, worker):
        budget = self.budgets[worker.id]
        return budget.capacity - budget.level

    def choose_p(self, req):
        return min(
            self.router.prefill_workers,
            key=lambda worker: (
                -self.router._hits(worker, req) if req.prefix_tokens else 0,
                self.occupied(worker),
                len(worker.scheduler.waiting) + len(worker.scheduler.running),
                worker.id,
            ),
        )

    def choose_d(self, req):
        prior = self.router.previous.get(req.chat_id)
        return next(
            (
                worker
                for worker in self.router.decode_workers
                if prior and worker.id == prior[1]
            ),
            None,
        ) or min(
            self.router.decode_workers,
            key=lambda worker: (
                self.occupied(worker),
                len(worker.scheduler.waiting),
                worker.id,
            ),
        )

    def track_peak(self, worker):
        self.peak[worker.id] = max(self.peak[worker.id], self.occupied(worker))

    def acquired(self, req, p, d, p_blocks, d_blocks, started):
        req.native_budget_p, req.native_budget_d = p, d
        req.native_budget_blocks = (p_blocks, d_blocks)
        req.native_budget_p_released = p_blocks == 0
        req.native_budget_wait_s = self.env.now - started

    def acquire(self, req, method):
        started = self.env.now
        # Two extra pages conservatively cover a retained partial source tail
        # during COW and an input/output boundary page. No cache-hit discount.
        p_blocks = ceil(req.prefill_len / req.block_size) + 2
        d_blocks = ceil((req.prefill_len + req.decode_len - 1) / req.block_size) + 2
        d = self.choose_d(req)
        if d_blocks > self.budgets[d.id].capacity:
            raise ValueError("Source cannot fit native D live-block budget")
        if (
            method == "PPD"
            and req.prefix_tokens
            and self.router._hits(d, req) >= req.prefix_tokens
        ):
            # No P token is held or awaited on the D-hit fast path. A hit can
            # be evicted while waiting for capacity; recheck before native pin.
            with self.budgets[d.id].get(d_blocks) as gate:
                yield gate
            self.track_peak(d)
            if self.router._hits(d, req) >= req.prefix_tokens:
                p = self.choose_p(req)
                self.acquired(req, p, d, 0, d_blocks, started)
                try:
                    reserve_destination(req, d)
                except BaseException:
                    self.budgets[d.id].put(d_blocks)
                    raise
                return
            self.budgets[d.id].put(d_blocks)
            # Release D before attempting P, avoiding reverse-order waits.
        p = self.choose_p(req)
        if p_blocks > self.budgets[p.id].capacity:
            raise ValueError("Source cannot fit native P live-block budget")
        with self.budgets[p.id].get(p_blocks) as gate:
            yield gate
        self.track_peak(p)
        try:
            d = self.choose_d(req)
            if d_blocks > self.budgets[d.id].capacity:
                raise ValueError("Source cannot fit native D live-block budget")
            with self.budgets[d.id].get(d_blocks) as gate:
                yield gate
            self.track_peak(d)
        except BaseException:
            self.budgets[p.id].put(p_blocks)
            raise
        self.acquired(req, p, d, p_blocks, d_blocks, started)

    def release_p(self, req):
        if req.native_budget_p_released:
            return
        self.budgets[req.native_budget_p.id].put(req.native_budget_blocks[0])
        req.native_budget_p_released = True

    def release(self, req):
        self.release_p(req)
        self.budgets[req.native_budget_d.id].put(req.native_budget_blocks[1])

    def describe(self):
        return dict(
            policy="Full-lifetime private block demand before native prefix pinning; original arrival retained",
            source="Native GPU allocator capacity minus native watermark",
            cache_discount=False,
            extra_cow_boundary_pages=2,
            devices={
                str(key): dict(
                    capacity_blocks=value.capacity,
                    occupied_at_end=value.capacity - value.level,
                    peak_budget_blocks=self.peak[key],
                )
                for key, value in self.budgets.items()
            },
            limitation="Conservative gate can delay sources that a sharing-aware global allocator could admit",
        )
