"""Arrival-atomic MigrationTax study over native TokenSim LRU primitives.

Each PhysicalTokenBlock is an extent header, not a GPU token allocation.
The new byte ledger accounts exactly for the extent's integer token length.
Completion-clock execution and tensor allocation are outside this study mode.
"""

import math
import random

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.free_queue import FreeBlockQueue

TOKEN_BYTES = 524_000


class Holder:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.queue = FreeBlockQueue()
        self.used = self.sequence = self.evictions = self.oversize = 0
        self.native_calls = {
            "PhysicalTokenBlock": 0,
            "FreeBlockQueue.append_lru": 0,
            "FreeBlockQueue.remove": 0,
            "FreeBlockQueue.popleft": 0,
        }

    def length(self, key):
        entry = self.entries.get(key)
        return None if entry is None else entry.extent_tokens

    def has(self, key, required):
        size = self.length(key)
        return size is not None and size >= required

    def put(self, key, tokens):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.queue.remove(previous)
            self.native_calls["FreeBlockQueue.remove"] += 1
            self.used -= previous.extent_tokens * TOKEN_BYTES
        report = {
            "replaced_tokens": None if previous is None else previous.extent_tokens,
            "evicted": [],
            "oversize": tokens * TOKEN_BYTES > self.capacity,
        }
        if report["oversize"]:
            self.oversize += 1
            return report
        while self.used + tokens * TOKEN_BYTES > self.capacity:
            victim = self.queue.popleft()
            self.native_calls["FreeBlockQueue.popleft"] += 1
            del self.entries[victim.block_hash]
            self.used -= victim.extent_tokens * TOKEN_BYTES
            report["evicted"].append([victim.block_hash, victim.extent_tokens])
            self.evictions += 1
        self.sequence += 1
        header = PhysicalTokenBlock(Device.GPU, self.sequence, 1)
        self.native_calls["PhysicalTokenBlock"] += 1
        header.block_hash = key
        header.cached = header.is_full = True
        header.last_accessed = self.sequence
        header.extent_tokens = tokens
        self.entries[key] = header
        self.queue.append_lru(header)
        self.native_calls["FreeBlockQueue.append_lru"] += 1
        self.used += tokens * TOKEN_BYTES
        return report

    def audit(self):
        assert self.used == sum(
            e.extent_tokens * TOKEN_BYTES for e in self.entries.values()
        )
        assert self.used <= self.capacity
        assert len(self.queue) == len(self.entries)
        assert all(e.ref_count == 0 for e in self.entries.values())


def prices(context, delta, state, beta=100_000_000_000.0):
    """Price all three methods on one frozen state; times are seconds."""
    pf = (2 * 6.7e9 * delta + 4 * 32 * 4096 * delta * context) / 268.8e12
    st = 0.00291 + 0.000000207 * context
    reference = pf + st
    n1 = 0 if state[0] == "H" else context - delta
    n3 = delta if state[1] == "H" else context
    fetch, send = n1 * TOKEN_BYTES / beta, n3 * TOKEN_BYTES / beta
    serial = fetch + pf + send + st
    ppd = reference if state[1] == "H" else serial
    a, b, c = fetch / 32, pf / 32, send / 32
    pipeline = a + b + c + 31 * max(a, b, c) + st
    ours = pipeline if state == "FF" else reference
    return {
        method: dict(
            ttft_s=duration,
            reference_s=reference,
            affected=duration > 2 * reference,
            store_to_p_tokens=(
                n1
                if method == "Serial"
                or state[1] == "F"
                and not (method == "Ours" and state == "HF")
                else 0
            ),
            p_to_d_tokens=(n3 if method == "Serial" or state[1] == "F" else 0),
        )
        for method, duration in (("Serial", serial), ("PPD", ppd), ("Ours", ours))
    }


class Study:
    """Own eight independent holders, a shared logical store and routing RNG."""

    def __init__(
        self, seed=0, capacity=242_000_000_000, high_bandwidth=100000 * 1024**3
    ):
        if type(seed) is not int or type(capacity) is not int or capacity < 1:
            raise ValueError("seed and positive capacity must be integers")
        if not math.isfinite(high_bandwidth) or high_bandwidth <= 1e11:
            raise ValueError("high bandwidth must be finite and exceed 100 GB/s")
        self.seed, self.capacity, self.high_bandwidth = seed, capacity, high_bandwidth
        self.p = [Holder(capacity) for _ in range(4)]
        self.d = [Holder(capacity) for _ in range(4)]
        self.previous_d = {}
        self.store = {}
        self.rng = random.Random(seed)
        self.row = 0
        self.last_arrival = -1.0
        self.rng_choice_calls = 0

    def step(self, request):
        cid = str(request["conversation"])
        turn, context, delta, output = (
            request["turn"],
            request["context_tokens"],
            request["new_tokens"],
            request["output_tokens"],
        )
        arrival = request["arrival_seconds"]
        if any(type(v) is not int for v in (turn, context, delta, output)):
            raise ValueError("turn and token counts must be integers")
        if turn < 0 or context < 1 or output < 1 or not 0 <= delta <= context:
            raise ValueError("invalid turn or token lengths")
        if turn == 0 and delta != context:
            raise ValueError("first turn cannot inherit a prefix")
        if (
            isinstance(arrival, bool)
            or not math.isfinite(arrival)
            or arrival < 0
            or arrival < self.last_arrival
        ):
            raise ValueError("arrivals must be finite, nonnegative and sorted")
        prefix = context - delta
        draws = []
        loads = {
            "P": [len(h.entries) for h in self.p],
            "D": [len(h.entries) for h in self.d],
        }

        def choose(group, side):
            sizes = loads[side]
            candidates = [i for i, n in enumerate(sizes) if n == min(sizes)]
            selected = self.rng.choice(candidates)
            self.rng_choice_calls += 1
            draws.append(
                dict(
                    side=side,
                    candidates=candidates,
                    selected=selected,
                    call=self.rng_choice_calls,
                )
            )
            return selected

        holders = [i for i, h in enumerate(self.p) if turn > 0 and h.has(cid, prefix)]
        p_index = holders[0] if holders else choose(self.p, "P")
        d_index = self.previous_d.get(cid)
        if d_index is None or not self.d[d_index].has(cid, prefix):
            d_index = choose(self.d, "D")
        ph, dh = self.p[p_index], self.d[d_index]
        p_length, d_length = ph.length(cid), dh.length(cid)
        p_hit, d_hit = ph.has(cid, prefix), dh.has(cid, prefix)
        state = ("H" if p_hit else "F") + ("H" if d_hit else "F")
        cell = state if turn > 0 else "T0"
        cost_state = state if turn > 0 else "FF"
        methods = prices(context, delta, cost_state)
        high = prices(context, delta, cost_state, self.high_bandwidth)
        previous_store = self.store.get(cid)
        writes = {
            "P": ph.put(cid, context),
            "D": dh.put(cid, context + output),
        }
        self.store[cid] = dict(
            tokens=context + output, version=self.row, arrival_seconds=arrival
        )
        self.previous_d[cid] = d_index
        record = dict(
            row_index=self.row,
            conv_id=cid,
            turn=turn,
            arrival_s=arrival,
            C_t=context,
            dC_clip=delta,
            output_tokens=output,
            required_prefix_tokens=prefix,
            P=p_index,
            D=d_index,
            cell=cell,
            p_version_tokens_before=p_length,
            d_version_tokens_before=d_length,
            p_hit=p_hit,
            d_hit=d_hit,
            loads_before=loads,
            rng_choices=draws,
            methods=methods,
            high_bandwidth=high,
            high_bandwidth_bytes_s=self.high_bandwidth,
            writes=writes,
            native_used_bytes=[h.used for h in self.p + self.d],
            footprints={
                "P_tokens": context,
                "D_tokens": context + output,
                "kv_bytes_per_token": TOKEN_BYTES,
                "allocation_tokens": 1,
            },
            store_write=dict(
                tokens=context + output,
                bytes=(context + output) * TOKEN_BYTES,
                visibility="atomic_at_arrival",
                version=self.row,
            ),
            store_prefix_tokens_before=(
                None if previous_store is None else previous_store["tokens"]
            ),
        )
        self.row += 1
        self.last_arrival = arrival
        return record

    def audit(self):
        for holder in self.p + self.d:
            holder.audit()
        return dict(
            capacity_bytes=self.capacity,
            token_bytes=TOKEN_BYTES,
            allocation_tokens=1,
            prefill_devices=4,
            decode_devices=4,
            native_calls=[h.native_calls for h in self.p + self.d],
            used_bytes=[h.used for h in self.p + self.d],
            resident_entries=[len(h.entries) for h in self.p + self.d],
            evictions=[h.evictions for h in self.p + self.d],
            oversize=[h.oversize for h in self.p + self.d],
            store_versions=len(self.store),
            rng_choice_calls=self.rng_choice_calls,
            study_rows=self.row,
            native_scheduler_executed=False,
        )
