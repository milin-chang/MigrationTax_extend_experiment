"""Stream native P/D observations without introducing simulation events."""

import csv
import json
from collections import Counter
from contextlib import ExitStack

from TokenSim.errors import ConfigurationError
from TokenSim.research.native_timeline_schema import (
    REQUEST_COLUMNS,
    STEP_COLUMNS,
    TRANSFER_COLUMNS,
    SEMANTICS,
)


class NativeTimeline:
    def __init__(self, engine, output, method):
        if engine.af_executor is not None or engine.config.dp_lockstep:
            raise ConfigurationError(
                "Native timeline requires ordinary independent workers"
            )
        self.engine, self.env, self.output, self.method = (
            engine,
            engine.env,
            output,
            method,
        )
        self.requests, self.steps, self.pending_transfers = {}, {}, {}
        self.counts = Counter()
        self.next_step = 0
        self.closed = False
        self.files = ExitStack()
        columns = {
            "request-timeline.csv": REQUEST_COLUMNS,
            "worker-steps.csv": STEP_COLUMNS,
            "kv-transfers.csv": TRANSFER_COLUMNS,
        }
        if any((output / name).exists() for name in [*columns, "timeline-schema.json"]):
            raise FileExistsError("Timeline output already exists")
        try:
            self.writers = {}
            for name, fields in columns.items():
                stream = self.files.enter_context(
                    (output / name).open("x", newline="", buffering=262144)
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                self.writers[name] = writer
            (output / "timeline-schema.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        status="recording",
                        columns=columns,
                        semantics=SEMANTICS,
                    ),
                    indent=2,
                )
                + "\n"
            )
            from TokenSim.research.native_timeline_hooks import install

            install(self)
            engine.native_timeline = self
        except BaseException:
            self.files.close()
            raise

    def arrival(self, req, row):
        self.requests[req.id] = dict(
            source_id=str(req.id),
            method=self.method,
            previous_source_id=row["previous_source_id"],
            source_turn=row["turn"],
            arrival_s=self.env.now,
        )

    def ready(self, req):
        self.requests[req.id].update(
            parent_ready_s=self.env.now, admission_started_s=self.env.now
        )

    def budget(self, req, role):
        self.requests[req.id][
            f"{'p' if role == 'prefill' else 'd'}_budget_acquired_s"
        ] = self.env.now

    def submitted(self, req):
        self.requests[req.id]["submitted_s"] = self.env.now

    def queued(self, worker, requests, *, scheduler=False):
        for req in requests:
            state = self.requests.get(req.id)
            if state is not None:
                prefix = "p" if worker.role == "prefill" else "d"
                key = f"{prefix}_{'scheduler' if scheduler else 'queue'}_enter_s"
                state.setdefault(key, self.env.now)

    def prepared(self, worker, requests, latency, started):
        if worker.id in self.steps:
            raise RuntimeError("An observed worker step has not been closed")
        phases = {
            str(r.id): "prefill" if r.is_context_build else "decode" for r in requests
        }
        kinds = set(phases.values())
        self.steps[worker.id] = dict(
            step_id=self.next_step,
            worker_id=worker.id,
            role=worker.role,
            tp_group_size=len(getattr(worker, "tp_group_worker_ids", (worker.id,))),
            tp_rank_ids=list(getattr(worker, "tp_group_worker_ids", (worker.id,))),
            phase=(
                next(iter(kinds)) if len(kinds) == 1 else "mixed" if kinds else "none"
            ),
            request_ids=[str(r.id) for r in requests],
            request_phases=phases,
            batch_size=len(requests),
            batch_tokens=sum(r.query_tokens for r in requests),
            scheduled_at_s=started,
            load_start_s=self.env.now,
            load_end_s=self.env.now + latency,
            load_wait_s=latency,
            first_compute_start_s=None,
            last_compute_end_s=None,
            compute_service_s=0.0,
            compute_calls=0,
        )
        self.next_step += 1
        if not requests and not latency:
            self.finish_step(worker, "no_compute")

    def compute_start(self, worker_id):
        step = self.steps.get(worker_id)
        if step is None:
            raise RuntimeError("Observed compute has no prepared worker step")
        if step["first_compute_start_s"] is None:
            step["first_compute_start_s"] = self.env.now
        prefix = "p" if step["role"] == "prefill" else "d"
        for rid in step["request_ids"]:
            state = self.requests[int(rid)]
            state.setdefault(f"{prefix}_first_compute_start_s", self.env.now)
            if prefix == "p" and step["request_phases"][rid] == "prefill":
                state.setdefault("p_prefill_start_s", self.env.now)

    def compute_end(self, worker_id, started):
        step = self.steps[worker_id]
        step["last_compute_end_s"] = self.env.now
        step["compute_service_s"] += self.env.now - started
        step["compute_calls"] += 1
        for rid in step["request_ids"]:
            state = self.requests[int(rid)]
            phase = step["request_phases"][rid]
            if step["role"] == "prefill":
                state["p_last_compute_end_s"] = self.env.now
                if phase == "prefill":
                    state["p_last_prefill_compute_end_s"] = self.env.now
            elif phase == "decode":
                state.setdefault("d_first_decode_step_end_s", self.env.now)
                state.setdefault("d_first_decode_batch_size", step["batch_size"])

    def advanced(self, worker, requests, before):
        step = self.steps[worker.id]
        for req in requests:
            state = self.requests[req.id]
            phase = step["request_phases"][str(req.id)]
            if (
                worker.role == "prefill"
                and phase == "prefill"
                and req.num_computed_tokens >= req.prefill_len
            ):
                state.setdefault("p_prefill_end_s", self.env.now)
            if req.generation_idx > before[req.id]:
                if before[req.id] == 0:
                    state.setdefault("first_client_output_at_s", self.env.now)
                if worker.role == "decode":
                    state.setdefault("first_d_output_at_s", self.env.now)
                    state.setdefault("first_d_output_phase", phase)
        self.finish_step(worker, "completed")

    def finish_step(self, worker, status):
        row = self.steps.pop(worker.id, None)
        if row is None:
            return
        start, end = row["first_compute_start_s"], row["last_compute_end_s"]
        has_compute = start is not None and end is not None
        row.update(
            status=status,
            step_completed_at_s=self.env.now,
            compute_span_s=end - start if has_compute else None,
            precompute_wait_s=None if start is None else start - row["load_end_s"],
            between_compute_wait_s=(
                end - start - row["compute_service_s"] if has_compute else None
            ),
            postcompute_wait_s=None if end is None else self.env.now - end,
        )
        for key in ("tp_rank_ids", "request_ids", "request_phases"):
            row[key] = json.dumps(row[key], separators=(",", ":"))
        self.writers["worker-steps.csv"].writerow(row)
        self.counts["worker_steps"] += 1

    def transfer(self, kind, req, values):
        if req is None:
            return
        if kind.endswith("_started") and "service_s" in values:
            key = (req.id, kind[:-8], values.get("layer"))
            if key in self.pending_transfers:
                raise RuntimeError("Overlapping transfer identity")
            self.pending_transfers[key] = dict(
                source_id=str(req.id),
                plan_branch=req.native_plan.branch,
                kind=key[1],
                layer=values.get("layer"),
                source_endpoint=values["source"],
                target_endpoint=values["target"],
                queued_at_s=self.env.now - values["queue_s"],
                service_started_s=self.env.now,
                queue_wait_s=values["queue_s"],
                service_s=values["service_s"],
                blocks=values["blocks"],
                bytes=values["bytes"],
                tp_shards=values["tp_shards"],
            )
        elif kind.endswith("_ready"):
            key = (req.id, kind[:-6], values.get("layer"))
            row = self.pending_transfers.pop(key, None)
            if row is not None:
                row.update(service_ended_s=self.env.now, status="completed")
                self.writers["kv-transfers.csv"].writerow(row)
                self.counts["kv_transfers"] += 1

    def completed(self, req, status="completed", reason=""):
        state = self.requests.pop(req.id)
        state.update(status=status, reason=reason)
        if status == "completed":
            _, p, d = self.engine.router.routes[req.id]
            state.update(
                plan_branch=req.native_plan.branch,
                prefill_worker=p.id,
                decode_worker=d.id,
                completed_at_s=req.completed_at,
                lifecycle_s=req.total_time,
                output_tokens=req.generation_idx,
                ttft_s=req.prefill_latency,
                parent_wait_s=state["parent_ready_s"] - state["arrival_s"],
                admission_wait_s=req.native_budget_wait_s,
            )
        self.writers["request-timeline.csv"].writerow(state)
        self.counts["requests"] += 1

    def close(self, status):
        if self.closed:
            return
        for worker in self.engine.request_workers:
            self.finish_step(worker, "unfinished")
        for state in self.requests.values():
            self.writers["request-timeline.csv"].writerow(
                dict(state, status="unfinished")
            )
        for row in self.pending_transfers.values():
            self.writers["kv-transfers.csv"].writerow(dict(row, status="unfinished"))
        self.files.close()
        schema = self.output / "timeline-schema.json"
        data = json.loads(schema.read_text())
        data.update(
            status=status,
            counts=dict(self.counts),
            unfinished_requests=len(self.requests),
            unfinished_transfers=len(self.pending_transfers),
        )
        schema.write_text(json.dumps(data, indent=2) + "\n")
        self.requests.clear()
        self.pending_transfers.clear()
        self.closed = True
