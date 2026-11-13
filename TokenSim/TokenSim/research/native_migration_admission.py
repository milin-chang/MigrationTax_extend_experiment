"""Opt-in full-lifetime native block budgets for MigrationTax participants."""

from math import ceil

from TokenSim.llm.handover_reservation import reserve_destination


def install_admission(admission):
    original = admission.acquire

    def acquire(req, method):
        if method != "MigrationTax":
            yield from original(req, method)
            return
        started = admission.env.now
        d_blocks = ceil((req.prefill_len + req.decode_len - 1) / req.block_size) + 2
        # The native uniform allocator reserves complete generated context on
        # both workers. Charging its complete demand prevents hidden P pressure
        # while HF retains a base snapshot and a COW tail.
        p_blocks = d_blocks
        d = admission.choose_d(req)
        if d_blocks > admission.budgets[d.id].capacity:
            raise ValueError("Source cannot fit native D live-block budget")
        if req.prefix_tokens and admission.router._hits(d, req) >= req.prefix_tokens:
            yield admission.budgets[d.id].get(d_blocks)
            admission.track_peak(d)
            if admission.router._hits(d, req) >= req.prefix_tokens:
                admission.observe_budget(req, d)
                p = admission.choose_p(req)
                admission.acquired(req, p, d, 0, d_blocks, started)
                try:
                    reserve_destination(req, d)
                except BaseException:
                    admission.budgets[d.id].put(d_blocks)
                    raise
                return
            admission.budgets[d.id].put(d_blocks)
        p = admission.choose_p(req)
        if p_blocks > admission.budgets[p.id].capacity:
            raise ValueError("Source cannot fit native P live-block budget")
        yield admission.budgets[p.id].get(p_blocks)
        admission.track_peak(p)
        admission.observe_budget(req, p)
        try:
            d = admission.choose_d(req)
            if d_blocks > admission.budgets[d.id].capacity:
                raise ValueError("Source cannot fit native D live-block budget")
            yield admission.budgets[d.id].get(d_blocks)
            admission.track_peak(d)
            admission.observe_budget(req, d)
        except BaseException:
            admission.budgets[p.id].put(p_blocks)
            raise
        admission.acquired(req, p, d, p_blocks, d_blocks, started)

    admission.acquire = acquire
