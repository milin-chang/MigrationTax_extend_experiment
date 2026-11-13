"""Check timeline causality and observational equivalence on native P/D paths."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import simpy

from TokenSim.research.native_runtime import execute
from TokenSim.research.native_migration_admission import install_admission
from tests.test_handover_states import evict_idle_cache
from tests.test_native_bandwidth_study import request

ROOT = Path(__file__).resolve().parents[2]


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class NativeTimelineTests(unittest.TestCase):
    def run_pair(self, rows, method, tp=1, probe=None):
        design = ROOT / f"config/five-stage-llama7b-h100-sxm80gb-tp{tp}-s0"
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                out = Path(directory) / str(enabled)
                out.mkdir()
                summary = execute(
                    rows,
                    method,
                    0,
                    out,
                    probe=probe,
                    hardware_profile=design / "hardware.json",
                    experiment_config=design / "study_nvl4.json",
                    record_timeline=enabled,
                )
                self.assertEqual(summary["status"], "completed")
                native = json.loads(
                    next((out / "native").glob("result_*.json")).read_text()
                )
                native.pop("simulator_wall_time")
                records = sorted(
                    map(json.loads, (out / "requests.jsonl").read_text().splitlines()),
                    key=lambda x: x["source_id"],
                )
                results.append((summary, native, records))
                if enabled:
                    timeline = csv_rows(out / "request-timeline.csv")
                    steps = csv_rows(out / "worker-steps.csv")
                    transfers = csv_rows(out / "kv-transfers.csv")
                    schema = json.loads((out / "timeline-schema.json").read_text())
                    self.assertEqual(schema["status"], "completed")
                    self.assertEqual(schema["unfinished_requests"], 0)
                    self.assertEqual(schema["unfinished_transfers"], 0)
                    self.assertEqual(schema["counts"]["requests"], len(rows))
                    self.check_timeline(records, timeline, steps, tp)
                else:
                    self.assertFalse((out / "request-timeline.csv").exists())
            self.assertEqual(
                results[0], results[1], "Observation changed native results"
            )
        return timeline, steps, transfers

    def check_timeline(self, records, timeline, steps, tp):
        by_id = {r["source_id"]: r for r in records}
        self.assertEqual(len(timeline), len(by_id))
        for row in timeline:
            req = by_id[row["source_id"]]
            self.assertEqual(row["status"], "completed")
            self.assertTrue(
                row["d_budget_acquired_s"],
                "Completed request lacks a D budget timestamp",
            )
            self.assertAlmostEqual(
                float(row["d_budget_acquired_s"]), float(row["submitted_s"]), places=12
            )
            if row["plan_branch"] != "decode_local_prefill":
                self.assertTrue(
                    row["p_budget_acquired_s"],
                    "P participant lacks a P budget timestamp",
                )
            if row["p_budget_acquired_s"]:
                self.assertLessEqual(
                    float(row["admission_started_s"]), float(row["p_budget_acquired_s"])
                )
                self.assertLessEqual(
                    float(row["p_budget_acquired_s"]), float(row["d_budget_acquired_s"])
                )
            self.assertAlmostEqual(
                float(row["ttft_s"]), req["native_prefill_latency_s"], places=12
            )
            self.assertAlmostEqual(
                float(row["first_client_output_at_s"]) - float(row["arrival_s"]),
                float(row["ttft_s"]),
                places=10,
            )
            self.assertAlmostEqual(
                float(row["submitted_s"]) - float(row["parent_ready_s"]),
                float(row["admission_wait_s"]),
                places=10,
            )
            self.assertAlmostEqual(
                float(row["completed_at_s"]), req["completed_at"], places=12
            )
            if req["first_d_output_sampled_at_s"] is None:
                self.assertEqual(row["first_d_output_at_s"], "")
            else:
                self.assertAlmostEqual(
                    float(row["first_d_output_at_s"]),
                    req["first_d_output_sampled_at_s"],
                    places=12,
                )
            if row["d_first_compute_start_s"]:
                self.assertGreaterEqual(
                    float(row["d_first_compute_start_s"]),
                    float(row["d_scheduler_enter_s"]),
                )
                self.assertGreaterEqual(
                    float(row["d_scheduler_enter_s"]), float(row["d_queue_enter_s"])
                )
            if row["plan_branch"] == "decode_local_prefill":
                self.assertEqual(row["p_prefill_start_s"], "")
                self.assertEqual(row["p_prefill_end_s"], "")
                self.assertEqual(row["first_d_output_phase"], "prefill")
                if req["generated"] > 1:
                    self.assertGreater(
                        float(row["d_first_decode_step_end_s"]),
                        float(row["first_d_output_at_s"]),
                    )
        for step in steps:
            ids, phases = json.loads(step["request_ids"]), json.loads(
                step["request_phases"]
            )
            self.assertEqual(int(step["batch_size"]), len(ids))
            self.assertEqual(set(ids), set(phases))
            self.assertEqual(int(step["tp_group_size"]), tp)
            self.assertEqual(len(json.loads(step["tp_rank_ids"])), tp)
            self.assertAlmostEqual(
                float(step["load_end_s"]) - float(step["load_start_s"]),
                float(step["load_wait_s"]),
                places=10,
            )
            if step["first_compute_start_s"]:
                self.assertGreaterEqual(
                    float(step["first_compute_start_s"]) + 1e-12,
                    float(step["load_end_s"]),
                )
                self.assertGreaterEqual(float(step["compute_service_s"]), 0)
                self.assertGreaterEqual(float(step["between_compute_wait_s"]), -1e-10)

    def test_all_methods_tp1_tp8_chunks_and_d_local_prefill(self):
        rows = [
            request(0, None, 0, 16384, 16384, 65, 0),
            request(1, "0", 1, 16512, 128, 4, 0.001),
        ]
        for tp in (1, 8):
            for method in ("Serial", "PPD", "MigrationTax"):
                with self.subTest(tp=tp, method=method):
                    timeline, steps, _ = self.run_pair(rows, method, tp)
                    prompt_steps = [
                        r
                        for r in steps
                        if r["role"] == "prefill"
                        and "0" in json.loads(r["request_ids"])
                    ]
                    self.assertGreaterEqual(len(prompt_steps), 2)
                    if method == "MigrationTax":
                        self.assertTrue(
                            any(int(r["compute_calls"]) == 32 for r in prompt_steps)
                        )

    @staticmethod
    def evict_after_parent(evict_p=False):
        def probe(engine, admission, env):
            original = engine.record_request_completion

            def completed(kind, req):
                original(kind, req)
                if kind == "decode" and req.id == 0:
                    for worker in engine.request_workers:
                        if engine._is_schedulable_worker(worker) and (
                            evict_p or worker.role == "decode"
                        ):
                            evict_idle_cache(worker)

            engine.record_request_completion = completed

        return probe

    def test_hf_short_responses_and_ff_store_reads(self):
        for tp in (1, 8):
            for evict_p in (False, True):
                for outputs in (1, 2, 4):
                    with self.subTest(tp=tp, evict_p=evict_p, outputs=outputs):
                        rows = [
                            request(0, None, 0, 128, 128, 1, 0),
                            request(1, "0", 1, 256, 128, outputs, 1),
                        ]
                        timeline, steps, transfers = self.run_pair(
                            rows, "MigrationTax", tp, self.evict_after_parent(evict_p)
                        )
                        child = next(r for r in timeline if r["source_id"] == "1")
                        self.assertEqual(
                            child["plan_branch"],
                            "layer_pipeline" if evict_p else "prefill_side_handoff",
                        )
                        self.assertTrue(transfers)
                        if not evict_p and outputs <= 2:
                            self.assertEqual(child["d_first_decode_step_end_s"], "")
                            self.assertEqual(child["first_d_output_at_s"], "")
                            self.assertTrue(child["first_client_output_at_s"])
                        if evict_p:
                            self.assertTrue(
                                any(
                                    r["kind"] == "FF_store_layer"
                                    and int(r["bytes"]) > 0
                                    for r in transfers
                                )
                            )
                        for row in transfers:
                            self.assertEqual(row["status"], "completed")
                            self.assertAlmostEqual(
                                float(row["service_ended_s"])
                                - float(row["service_started_s"]),
                                float(row["service_s"]),
                                places=10,
                            )

    def test_shared_batch_and_budget_wait(self):
        def same_group(engine, admission, env):
            admission.choose_p = lambda req: admission.router.prefill_workers[0]
            admission.choose_d = lambda req: admission.router.decode_workers[0]

        rows = [
            dict(request(i, None, 0, 128, 128, 4, 0), conversation=i) for i in range(2)
        ]
        _, steps, _ = self.run_pair(rows, "Serial", probe=same_group)
        self.assertTrue(any(int(r["batch_size"]) == 2 for r in steps))
        rows = [
            dict(request(i, None, 0, 65000, 65000, 4, 0), conversation=i)
            for i in range(2)
        ]
        for method in ("Serial", "MigrationTax"):
            timeline, _, _ = self.run_pair(rows, method, probe=same_group)
            delayed = next(r for r in timeline if float(r["admission_wait_s"]) > 0)
            self.assertGreater(
                float(delayed["p_budget_acquired_s"]),
                float(delayed["admission_started_s"]),
            )
            self.assertGreaterEqual(
                float(delayed["submitted_s"]), float(delayed["d_budget_acquired_s"])
            )

    def test_migration_lost_d_hit_records_only_successful_acquisitions(self):
        env = simpy.Environment()
        p = SimpleNamespace(id=0, role="prefill")
        d = SimpleNamespace(id=1, role="decode")
        req = SimpleNamespace(
            prefill_len=128, decode_len=4, block_size=64, prefix_tokens=64
        )
        events, accepted = [], []
        budgets = {0: simpy.Container(env, 100, 100), 1: simpy.Container(env, 100, 0)}
        admission = SimpleNamespace(
            env=env,
            acquire=lambda req, method: None,
            budgets=budgets,
            choose_d=lambda req: d,
            choose_p=lambda req: p,
            track_peak=lambda worker: None,
            router=SimpleNamespace(_hits=lambda worker, req: 64 if env.now < 1 else 0),
            observe_budget=lambda req, worker: events.append((worker.role, env.now)),
            acquired=lambda *args: accepted.append(
                (args[3], args[4], env.now - args[5])
            ),
        )
        install_admission(admission)

        def free_capacity_after_eviction():
            yield env.timeout(1)
            yield budgets[1].put(100)

        env.process(admission.acquire(req, "MigrationTax"))
        env.process(free_capacity_after_eviction())
        env.run()
        self.assertEqual(events, [("prefill", 1), ("decode", 1)])
        self.assertEqual(accepted, [(5, 5, 1)])
        self.assertEqual((budgets[0].level, budgets[1].level), (95, 95))

    def test_busy_d_worker_message_precedes_scheduler_insertion(self):
        def same_group(engine, admission, env):
            admission.choose_p = lambda req: admission.router.prefill_workers[0]
            admission.choose_d = lambda req: admission.router.decode_workers[0]

        samples = [
            dict(request(i, None, 0, 128, 128, 32, i * 0.003), conversation=i)
            for i in range(8)
        ]
        timeline, _, _ = self.run_pair(samples, "Serial", probe=same_group)
        self.assertTrue(
            any(
                float(r["d_scheduler_enter_s"]) > float(r["d_queue_enter_s"])
                for r in timeline
            )
        )

    def test_rejected_request_is_explicit(self):
        design = ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp1-s0"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = execute(
                [request(0, None, 0, 200000, 200000, 4, 0)],
                "Serial",
                0,
                out,
                hardware_profile=design / "hardware.json",
                experiment_config=design / "study_nvl4.json",
                record_timeline=True,
            )
            self.assertEqual(result["status"], "incomplete")
            row = csv_rows(out / "request-timeline.csv")[0]
            self.assertEqual(row["status"], "rejected")
            self.assertTrue(row["reason"])
            self.assertEqual(row["p_prefill_start_s"], "")

    def test_failure_closes_partial_csv(self):
        design = ROOT / "config/five-stage-llama7b-h100-sxm80gb-tp1-s0"

        def probe(engine, admission, env):
            def fail(worker_id, latency):
                raise RuntimeError("injected compute failure")
                yield

            engine.run_complete_compute = fail

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(Exception, "injected compute failure"):
                execute(
                    [request(0, None, 0, 128, 128, 4, 0)],
                    "Serial",
                    0,
                    out,
                    probe=probe,
                    hardware_profile=design / "hardware.json",
                    experiment_config=design / "study_nvl4.json",
                    record_timeline=True,
                )
            schema = json.loads((out / "timeline-schema.json").read_text())
            self.assertEqual(schema["status"], "failed")
            self.assertEqual(
                csv_rows(out / "request-timeline.csv")[0]["status"], "unfinished"
            )
            self.assertTrue(
                any(
                    r["status"] == "failed_compute"
                    for r in csv_rows(out / "worker-steps.csv")
                )
            )


if __name__ == "__main__":
    unittest.main()
