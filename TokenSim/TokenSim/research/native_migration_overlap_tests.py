"""Scheduled HF short-response writes during another native D decode step.

Observers wrap real native compute, connector save calls and driver events.
They neither supply latencies nor replace cache decisions or completion events.
"""

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.research.native_runtime import execute
from tests.test_handover_states import evict_idle_cache


def requests(outputs):
    return [
        dict(
            source_id="1",
            previous_source_id=None,
            conversation="A",
            turn=0,
            arrival_seconds=0.0,
            context_tokens=64,
            new_tokens=64,
            output_tokens=1,
            source_order=0,
        ),
        dict(
            source_id="2",
            previous_source_id=None,
            conversation="B",
            turn=0,
            arrival_seconds=1.0,
            context_tokens=64,
            new_tokens=64,
            output_tokens=128,
            source_order=1,
        ),
        dict(
            source_id="3",
            previous_source_id="1",
            conversation="A",
            turn=1,
            arrival_seconds=1.05,
            context_tokens=81,
            new_tokens=17,
            output_tokens=outputs,
            source_order=2,
        ),
    ]


def stores(worker):
    return [
        c
        for c in flatten_connectors(worker.connector)
        if c.name == "MooncakeStoreConnector"
    ]


def pending_state(connector):
    return (
        set(connector._pending_async_loads),
        set(connector._pending_async_puts),
        dict(connector._delayed_releases),
    )


def one_case(path, outputs):
    path.mkdir()
    active, before, observations, saves = {}, [], [], []

    def probe(engine, admission, env):
        original_completion = engine.record_request_completion
        original_compute = engine.run_complete_compute
        driver = engine.native_migration
        original_log = driver.log

        def completed(kind, req):
            original_completion(kind, req)
            if kind == "decode" and req.id == 1:
                # P retains A. Its real D cache entries are evicted before B
                # arrives; both conversations retain their native D0 affinity.
                for worker in engine.workers:
                    if worker.role == "decode":
                        evict_idle_cache(worker)

        def compute(worker_id, latency):
            worker = engine.workers[worker_id]
            active[worker_id] = dict(
                started=env.now,
                duration=latency,
                requests=[r.id for r in worker.scheduler.running],
                progress={r.id: r.generation_idx for r in worker.scheduler.running},
            )
            try:
                return (yield from original_compute(worker_id, latency))
            finally:
                active.pop(worker_id)

        def log(kind, req=None, **values):
            if req is not None and req.id == 3:
                worker = req.native_migration.d
                if kind == "HF_completed_response_writeback_started":
                    current = active.get(worker.id)
                    assert current is not None and 2 in current["requests"], current
                    assert current["progress"][2] > 0, current
                    assert (
                        current["started"]
                        <= env.now
                        < current["started"] + current["duration"]
                    )
                    for connector in stores(worker):
                        before.append(
                            (
                                connector,
                                connector._metadata,
                                connector.simulation_time,
                                pending_state(connector),
                            )
                        )
                    observations.append(
                        dict(
                            native_D_worker=worker.id,
                            native_active_requests=current["requests"],
                            native_active_progress=current["progress"],
                            native_step_started_s=current["started"],
                            native_step_service_s=current["duration"],
                            HF_write_issued_s=env.now,
                        )
                    )
                if kind == "HF_completed_response_writeback_issued":
                    assert before
                    for connector, metadata, clock, pending in before:
                        assert connector._metadata is metadata
                        assert connector.simulation_time == clock
                        assert pending_state(connector) == pending
                    observations[-1]["metadata_identity_restored"] = True
                    observations[-1]["simulation_clock_restored"] = True
                    observations[-1]["native_pending_state_preserved"] = True
            original_log(kind, req, **values)

        for worker in engine.workers:
            for connector in stores(worker):
                original_save = connector.wait_for_save

                def wait_for_save(
                    connector=connector,
                    original_save=original_save,
                    worker_id=worker.id,
                ):
                    saves.append(
                        dict(
                            time_s=env.now,
                            worker=worker_id,
                            request_ids=[
                                rid
                                for p in connector._metadata.saves
                                for rid in p.request_ids
                            ],
                        )
                    )
                    return original_save()

                connector.wait_for_save = wait_for_save
        engine.record_request_completion = completed
        engine.run_complete_compute = compute
        driver.log = log

    result = execute(requests(outputs), "MigrationTax", 0, path, probe=probe)
    assert result["status"] == "completed" and result["completed"] == 3, result
    assert len(observations) == 1, observations
    assert sum(3 in row["request_ids"] for row in saves) == 1, saves
    source = [
        json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()
    ]
    assert {r["source_id"]: r["generated"] for r in source} == {
        "1": 1,
        "2": 128,
        "3": outputs,
    }
    routes = [
        json.loads(line)
        for line in (path / "native-route-decisions.jsonl").read_text().splitlines()
    ]
    long_route = next(r for r in routes if r["request_id"] == 2)
    hf_route = next(r for r in routes if r["request_id"] == 3)
    assert hf_route["p_local_tokens"] >= 64 and hf_route["d_local_tokens"] < 64
    assert long_route["decode_worker"] == hf_route["decode_worker"]
    with gzip.open(path / "native-migration-events.jsonl.gz", "rt") as stream:
        events = [json.loads(line) for line in stream]
    own = [e for e in events if e.get("source_id") == "3"]
    owner = next(e for e in own if e["kind"] == "ownership_D")
    issued = next(
        e for e in own if e["kind"] == "HF_completed_response_writeback_issued"
    )
    done = next(e for e in own if e["kind"] == "HF_completed_response_writeback_ready")
    assert owner["time_s"] <= issued["time_s"] <= done["time_s"]
    samples = [e for e in own if e["kind"] == "native_output_sampled"]
    assert [e["generated"] for e in samples] == list(range(1, outputs + 1))
    assert all(e["role"] == "prefill" for e in samples)
    end = json.loads((path / "native-end-state.json").read_text())
    assert all(
        r["gpu_status"][1] == 0 and not r["running"] and not r["waiting"] for r in end
    )
    budgets = json.loads((path / "native-admission-budget.json").read_text())
    assert all(d["occupied_at_end"] == 0 for d in budgets["devices"].values())
    evidence = json.loads((path / "native-migration-evidence.json").read_text())
    assert evidence["store_leases_remaining"] == 0
    proof = dict(
        outputs=outputs,
        native_overlap=observations,
        hf_save_plan_native_calls=1,
        generated_counts={"A": outputs, "B": 128},
        owners_and_budgets_released=True,
    )
    (path / "overlap-assertions.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--outputs", nargs="+", type=int, choices=[1, 2], default=[1, 2]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    report = [one_case(args.output / f"HF-O{o}", o) for o in args.outputs]
    (args.output / "overlap-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
