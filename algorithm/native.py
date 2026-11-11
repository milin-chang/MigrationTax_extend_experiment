"""Untimed method operations for native simulator adapters.

An adapter supplies execution time, physical allocation, routing and events.
The plan expresses logical KV tokens; physical block padding belongs to it.
HF uses at most two P-side outputs, then a KV and ownership barrier.
Only forwarded tokens have KV: output sample k materializes k-1 positions.
"""

from dataclasses import dataclass
from .model import Turn


@dataclass(frozen=True)
class Operation:
    id: str
    kind: str
    device: str
    tokens: int
    after: tuple[str, ...] = ()
    layer: int | None = None
    context_tokens: int = 0


@dataclass(frozen=True)
class ExecutionPlan:
    method: str
    branch: str
    operations: tuple[Operation, ...]
    first_output: str
    completion: str
    layers: int
    p_outputs_before_handoff: int = 0

    def __post_init__(self):
        seen = set()
        for operation in self.operations:
            if operation.id in seen or any(
                parent not in seen for parent in operation.after
            ):
                raise ValueError(
                    "Operations must have unique topologically ordered identities"
                )
            if operation.tokens < 0 or operation.context_tokens < 0:
                raise ValueError("Operation lengths must be nonnegative")
            seen.add(operation.id)
        if self.first_output not in seen or self.completion not in seen:
            raise ValueError("Missing output or completion operation")


def branch_for(method: str, state) -> str:
    """Select a method branch without allocating its operation DAG."""
    from .model import CacheState

    state = CacheState(state)
    if method == "Serial":
        return "serial"
    if method not in ("PPD", "MigrationTax"):
        raise ValueError("Unknown method")
    if state.decode_hit:
        return "decode_local_prefill"
    if method == "PPD":
        return "serial_fallback"
    return "prefill_side_handoff" if state.prefill_hit else "layer_pipeline"


def build_plan(
    method: str, turn: Turn, *, layers: int = 32, handoff_outputs: int = 2
) -> ExecutionPlan:
    if method not in ("Serial", "PPD", "MigrationTax"):
        raise ValueError("Unknown method")
    if type(layers) is not int or layers < 1:
        raise ValueError("layers must be a positive integer")
    if type(handoff_outputs) is not int or handoff_outputs < 1:
        raise ValueError("handoff_outputs must be positive")
    operations = []

    def add(name, kind, device, tokens=0, after=(), layer=None, context=None):
        operations.append(
            Operation(
                name,
                kind,
                device,
                tokens,
                tuple(after),
                layer,
                turn.context_tokens if context is None else context,
            )
        )
        return name

    state, c, q, o = (
        turn.cache_state,
        turn.context_tokens,
        turn.new_tokens,
        turn.output_tokens,
    )
    k = 0
    if method != "Serial" and state.decode_hit:
        branch = "decode_local_prefill"
        prefill = add("prefill", "prefill", "D", q)
        first = add("first_output", "decode_first", "D", 1, (prefill,))
        last = add("remaining", "decode_remaining", "D", o - 1, (first,))
    elif method == "MigrationTax" and state.prefill_hit:
        branch = "prefill_side_handoff"
        prefill = add("prefill", "prefill", "P", q)
        base = add("base_handoff", "kv_transfer", "P->D", c, (prefill,))
        first = add("first_output", "decode_first", "P", 1, (prefill,))
        k = min(handoff_outputs, o)
        p_last = add("p_continue", "decode_remaining", "P", k - 1, (first,))
        sync = add("incremental_sync", "kv_transfer", "P->D", k - 1, (base, p_last))
        owner = add("ownership", "ownership_barrier", "P->D", 0, (sync, p_last))
        last = add(
            "remaining", "decode_remaining", "D", o - k, (owner,), context=c + k - 1
        )
    elif method == "MigrationTax":
        branch = "layer_pipeline"
        previous_compute = previous_fetch = previous_send = None
        for layer in range(layers):
            fetch = add(
                f"fetch_{layer}",
                "kv_transfer",
                "store->P",
                turn.prefix_tokens,
                () if previous_fetch is None else (previous_fetch,),
                layer,
            )
            compute_deps = (
                (fetch,) if previous_compute is None else (fetch, previous_compute)
            )
            compute = add(
                f"prefill_{layer}", "prefill_layer", "P", q, compute_deps, layer
            )
            send_deps = (
                (compute,) if previous_send is None else (compute, previous_send)
            )
            send = add(f"send_{layer}", "kv_transfer", "P->D", c, send_deps, layer)
            previous_fetch, previous_compute, previous_send = fetch, compute, send
        first = add("first_output", "decode_first", "D", 1, (previous_send,))
        last = add("remaining", "decode_remaining", "D", o - 1, (first,))
    else:
        branch = "serial" if method == "Serial" else "serial_fallback"
        fetch = add(
            "fetch",
            "kv_transfer",
            "store->P",
            0 if state.prefill_hit else turn.prefix_tokens,
        )
        prefill = add("prefill", "prefill", "P", q, (fetch,))
        send = add(
            "handoff", "kv_transfer", "P->D", q if state.decode_hit else c, (prefill,)
        )
        first = add("first_output", "decode_first", "D", 1, (send,))
        last = add("remaining", "decode_remaining", "D", o - 1, (first,))
    write = add("writeback", "kv_transfer", "D->store", c + o - 1, (last,))
    finish = add("complete", "publish_and_release", "D", 0, (write,))
    return ExecutionPlan(method, branch, tuple(operations), first, finish, layers, k)
