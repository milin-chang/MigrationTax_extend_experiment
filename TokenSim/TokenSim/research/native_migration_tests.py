"""Scheduled native MigrationTax fixtures; labels assert real cache outcomes.

The bundle coordinator runs this entry under its single-job resource lock.
It requires native_runtime.execute to install the opt-in MigrationTax driver.
"""

import argparse
import gzip
import json
from math import ceil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from TokenSim.research.native_runtime import execute
from tests.test_handover_states import evict_idle_cache


def requests(output_tokens, new_tokens, parent_context):
    return [
        dict(
            source_id="1",
            previous_source_id=None,
            conversation="fixture",
            turn=0,
            arrival_seconds=0.0,
            context_tokens=parent_context,
            new_tokens=parent_context,
            output_tokens=1,
            source_order=0,
        ),
        dict(
            source_id="2",
            previous_source_id="1",
            conversation="fixture",
            turn=1,
            arrival_seconds=1.0,
            context_tokens=parent_context + new_tokens,
            new_tokens=new_tokens,
            output_tokens=output_tokens,
            source_order=1,
        ),
    ]


def one_case(path, cell, outputs, new_tokens, parent_context):
    path.mkdir()
    flushes = []

    def probe(engine, admission, env):
        original = engine.record_request_completion

        def completed(kind, req):
            original(kind, req)
            if kind != "decode" or req.id != 1:
                return
            for role, missed in (
                ("prefill", cell[0] == "F"),
                ("decode", cell[1] == "F"),
            ):
                if not missed:
                    continue
                for worker in engine.workers:
                    if worker.role == role and engine._is_schedulable_worker(worker):
                        # Existing native test helper calls the real cache's
                        # evict API and rejects attempts to flush pinned pages.
                        evict_idle_cache(worker)
                        flushes.append(
                            dict(time_s=env.now, worker=worker.id, role=role)
                        )

        engine.record_request_completion = completed

    result = execute(
        requests(outputs, new_tokens, parent_context),
        "MigrationTax",
        0,
        path,
        probe=probe,
    )
    assert result["status"] == "completed", result
    assert result["completed"] == 2
    rows = [
        json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()
    ]
    child = next(r for r in rows if r["source_id"] == "2")
    assert child["generated"] == outputs
    routes = [
        json.loads(line)
        for line in (path / "native-route-decisions.jsonl").read_text().splitlines()
    ]
    route = next(r for r in routes if r["request_id"] == 2)
    observed = ("H" if route["p_local_tokens"] >= parent_context else "F") + (
        "H" if route["d_local_tokens"] >= parent_context else "F"
    )
    assert observed == cell, (cell, route, flushes)
    with gzip.open(path / "native-migration-events.jsonl.gz", "rt") as stream:
        events = [json.loads(line) for line in stream]
    own = [e for e in events if e.get("source_id") == "2"]
    samples = [e for e in own if e["kind"] == "native_output_sampled"]
    evidence = json.loads((path / "native-migration-evidence.json").read_text())
    layers = evidence["layers"]
    block_bytes = 16 * evidence["native_kv_geometry"]["token_bytes"]
    assert [e["generated"] for e in samples] == list(range(1, outputs + 1)), samples
    assert samples[0]["role"] == ("prefill" if cell == "HF" else "decode")
    assert abs(samples[0]["time_s"] - 1.0 - child["native_prefill_latency_s"]) < 1e-8
    if cell == "HF":
        k = min(2, outputs)
        owner = next(e for e in own if e["kind"] == "ownership_D")
        base = next(e for e in own if e["kind"] == "HF_base_complete")
        sync = next(e for e in own if e["kind"] == "HF_incremental_complete")
        assert owner["time_s"] >= max(
            base["time_s"], sync["time_s"], samples[k - 1]["time_s"]
        )
        assert sync["valid_tokens"] == k - 1
        base_moves = [e for e in own if e["kind"] == "HF_base_started"]
        incremental = [e for e in own if e["kind"] == "HF_incremental_started"]
        context = parent_context + new_tokens
        assert sum(e["bytes"] for e in base_moves) == ceil(context / 16) * block_bytes
        expected_increment = (
            0
            if k == 1
            else (ceil((context + k - 1) / 16) - context // 16) * block_bytes
        )
        assert sum(e["bytes"] for e in incremental) == expected_increment
        assert sum(e["role"] == "prefill" for e in samples) == k
        assert all(
            e["time_s"] >= owner["time_s"] for e in samples if e["role"] == "decode"
        )
        if outputs <= 2:
            write = next(
                e for e in own if e["kind"] == "HF_completed_response_writeback_started"
            )
            assert write["time_s"] >= owner["time_s"]
        if k > 1 and (parent_context + new_tokens) % 16:
            cow = next(e for e in own if e["kind"] == "HF_tail_COW_ready")
            assert cow["time_s"] <= samples[1]["time_s"]
    if cell == "FF":
        owner = next(e for e in own if e["kind"] == "ownership_D")
        ready = [e for e in own if e["kind"] == "FF_destination_layer_ready"]
        chunks = [e for e in own if e["kind"] == "FF_native_chunk_committed"]
        assert len(chunks) >= max(1, ceil(new_tokens / 4096))
        assert len(ready) == layers * len(chunks)
        assert (
            max(e["time_s"] for e in ready) <= owner["time_s"] <= samples[0]["time_s"]
        )
        assert all(e["role"] == "decode" for e in samples)
        previous = None
        previous_tokens = 0
        transfers = [e for e in own if e["kind"] == "FF_send_layer_started"]
        by_layer = {
            layer: [e for e in transfers if e["layer"] == layer]
            for layer in range(layers)
        }
        for index, committed in enumerate(chunks):
            starts = [
                e
                for e in events
                if e["kind"] == "FF_native_chunk_started"
                and e["chunk"] == committed["chunk"]
            ]
            assert len(starts) == 1
            if previous is not None:
                assert starts[0]["time_s"] >= previous
            previous = committed["time_s"]
            sends = [e for e in ready if e["chunk"] == committed["chunk"]]
            assert {e["layer"] for e in sends} == set(range(layers))
            assert max(e["time_s"] for e in sends) <= committed["time_s"]
            pages = ceil(committed["tokens"] / 16) - previous_tokens // 16
            assert all(len(by_layer[layer]) == len(chunks) for layer in range(layers))
            assert all(
                by_layer[layer][index]["bytes"] == pages * block_bytes // layers
                for layer in range(layers)
            )
            previous_tokens = committed["tokens"]
    end = json.loads((path / "native-end-state.json").read_text())
    assert all(row["gpu_status"][1] == 0 for row in end), end
    assert all(not row["waiting"] and not row["running"] for row in end), end
    budgets = json.loads((path / "native-admission-budget.json").read_text())
    assert all(d["occupied_at_end"] == 0 for d in budgets["devices"].values())
    assert evidence["store_leases_remaining"] == 0
    assertions = dict(
        cell=cell,
        outputs=outputs,
        new_tokens=new_tokens,
        parent_context=parent_context,
        generated=child["generated"],
        native_output_roles=[e["role"] for e in samples],
        real_cache_flushes=flushes,
        owners_released=True,
    )
    (path / "fixture-assertions.json").write_text(
        json.dumps(assertions, indent=2) + "\n"
    )
    return assertions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=["HH", "FH", "HF", "FF"],
        default=["HH", "FH", "HF", "FF"],
    )
    parser.add_argument("--outputs", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--new-tokens", type=int, default=17)
    parser.add_argument("--parent-context", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or args.new_tokens < 0 or args.parent_context < 1:
        parser.error("Use a new output directory and valid input lengths")
    if any(o < 1 for o in args.outputs):
        parser.error("Output counts must be positive")
    args.output.mkdir(parents=True)
    report = []
    for cell in args.cells:
        for outputs in args.outputs:
            report.append(
                one_case(
                    args.output / f"{cell}-O{outputs}",
                    cell,
                    outputs,
                    args.new_tokens,
                    args.parent_context,
                )
            )
    (args.output / "fixture-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
