"""以具體搬移量、獨立逐層排程與統計邊界驗證三方法。"""

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from algorithm import CacheState, CostModel, Turn, ours, ppd, serial
from algorithm.comparison import compare_turns, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CostModel()
        self.turn = Turn(32_768, 512, CacheState.FF, output_tokens=32)

    def test_published_calibration_and_units(self) -> None:
        # 正文常數的 ST(32768) 為 9.692976 ms，四捨五入約 9.70 ms。
        self.assertAlmostEqual(self.model.decode_step_seconds(32_768), 0.009692976)
        self.assertAlmostEqual(self.model.move_seconds(1), 0.00000524)
        # ΔC=1、C=1：13.4e9 projection ops 加 524288 attention ops。
        one_token = Turn(1, 1, CacheState.FF)
        expected_prefill_seconds = 13_400_524_288 / 268_800_000_000_000
        self.assertAlmostEqual(
            self.model.prefill_seconds(one_token),
            expected_prefill_seconds,
        )

    def test_serial_moves_the_right_tokens_in_every_state(self) -> None:
        expected_transfers = {
            "HH": (0, 512),
            "FH": (32_256, 512),
            "HF": (0, 32_768),
            "FF": (32_256, 32_768),
        }
        for state, expected_tokens in expected_transfers.items():
            with self.subTest(state=state):
                turn = replace(self.turn, cache_state=state)
                result = serial(turn, self.model)
                actual_tokens = (
                    result.store_to_prefill_tokens,
                    result.prefill_to_decode_tokens,
                )
                self.assertEqual(actual_tokens, expected_tokens)

                expected_stall = sum(expected_tokens) * 524_000 / 100_000_000_000
                self.assertAlmostEqual(result.movement_stall_seconds, expected_stall)
                self.assertEqual(result.prefill_device, "P")
                self.assertEqual(result.first_decode_device, "D")

    def test_ppd_d_hit_avoids_both_transfers(self) -> None:
        for state in (CacheState.HH, CacheState.FH):
            turn = replace(self.turn, cache_state=state)
            result = ppd(turn, self.model)
            self.assertEqual(result.ttft_seconds, result.reference_seconds)
            self.assertEqual(result.store_to_prefill_tokens, 0)
            self.assertEqual(result.prefill_to_decode_tokens, 0)
            self.assertEqual(result.prefill_device, "D")
            self.assertEqual(result.first_decode_device, "D")

    def test_ppd_d_miss_is_exactly_serial_fallback(self) -> None:
        for state in (CacheState.HF, CacheState.FF):
            turn = replace(self.turn, cache_state=state)
            serial_result = serial(turn, self.model)
            expected_result = replace(
                serial_result,
                method="PPD",
                path="serial_fallback",
            )
            self.assertEqual(ppd(turn, self.model), expected_result)

    def test_ours_uses_ppd_for_both_decode_hit_states(self) -> None:
        for state in (CacheState.HH, CacheState.FH):
            turn = replace(self.turn, cache_state=state)
            ppd_result = ppd(turn, self.model)
            expected_result = replace(ppd_result, method="Ours")
            self.assertEqual(ours(turn, self.model), expected_result)

    def test_handoff_removes_stall_but_preserves_full_context_transfer(self) -> None:
        turn = replace(self.turn, cache_state=CacheState.HF)
        for bandwidth in (0.1, 25, 100, 1600):
            model = replace(self.model, bandwidth_gb_s=bandwidth)
            result = ours(turn, model)
            self.assertEqual(result.first_decode_device, "P")
            self.assertEqual(result.prefill_to_decode_tokens, 32_768)
            self.assertEqual(result.ttft_seconds, model.reference_seconds(turn))
            self.assertGreater(serial(turn, model).ttft_seconds, result.ttft_seconds)

    def test_ff_pipeline_matches_independent_layer_schedule(self) -> None:
        # 逐層模擬三個資源的可用時間，獨立核對 Eq. 2 的封閉公式。
        for layers in (1, 2, 32):
            for bandwidth in (0.1, 25, 100, 1600):
                for new_tokens in (0, 512, 32_768):
                    with self.subTest(
                        layers=layers, bandwidth=bandwidth, new=new_tokens
                    ):
                        model = replace(
                            self.model,
                            layers=layers,
                            bandwidth_gb_s=bandwidth,
                        )
                        turn = replace(self.turn, new_tokens=new_tokens)
                        fetch_per_layer = (
                            model.move_seconds(turn.prefix_tokens) / layers
                        )
                        compute_per_layer = model.prefill_seconds(turn) / layers
                        send_per_layer = (
                            model.move_seconds(turn.context_tokens) / layers
                        )
                        store_free = 0.0
                        prefill_free = 0.0
                        decode_port_free = 0.0
                        for _ in range(layers):
                            store_free += fetch_per_layer
                            prefill_start = max(store_free, prefill_free)
                            prefill_free = prefill_start + compute_per_layer
                            send_start = max(prefill_free, decode_port_free)
                            decode_port_free = send_start + send_per_layer

                        first_step = model.decode_step_seconds(turn.context_tokens)
                        expected_ttft = decode_port_free + first_step
                        ours_result = ours(turn, model)
                        serial_result = serial(turn, model)
                        self.assertAlmostEqual(ours_result.ttft_seconds, expected_ttft)
                        self.assertLessEqual(
                            ours_result.ttft_seconds,
                            serial_result.ttft_seconds + 1e-12,
                        )

    def test_reference_and_ordering_hold_for_varied_inputs(self) -> None:
        rng = random.Random(2026)
        for _ in range(100):
            context = rng.randint(1, 65_536)
            new_tokens = rng.randint(0, context)
            state = rng.choice(list(CacheState))
            turn = Turn(context, new_tokens, state)
            serial_result = serial(turn, self.model)
            ppd_result = ppd(turn, self.model)
            ours_result = ours(turn, self.model)
            self.assertEqual(
                serial_result.reference_seconds, ppd_result.reference_seconds
            )
            self.assertEqual(
                ppd_result.reference_seconds, ours_result.reference_seconds
            )
            self.assertGreaterEqual(
                ours_result.ttft_seconds + 1e-12, ours_result.reference_seconds
            )
            self.assertLessEqual(
                ours_result.ttft_seconds, ppd_result.ttft_seconds + 1e-12
            )
            self.assertLessEqual(
                ppd_result.ttft_seconds, serial_result.ttft_seconds + 1e-12
            )

    def test_bandwidth_changes_only_transfer_cost(self) -> None:
        slower = replace(self.model, bandwidth_gb_s=25)
        normal_result = serial(self.turn, self.model)
        slower_result = serial(self.turn, slower)
        self.assertEqual(
            normal_result.reference_seconds, slower_result.reference_seconds
        )
        expected_stall = 4 * normal_result.movement_stall_seconds
        self.assertAlmostEqual(slower_result.movement_stall_seconds, expected_stall)

    def test_first_turn_has_no_prefix_fetch_and_is_not_returning(self) -> None:
        first = Turn(512, 512, CacheState.FF, is_returning=False)
        for evaluate in (serial, ppd, ours):
            result = evaluate(first, self.model)
            self.assertEqual(result.store_to_prefill_tokens, 0)
        summary = summarize([serial(first, self.model)])
        self.assertEqual(summary["returning_turns"], 0)

    def test_comparison_preserves_input_objects_and_states(self) -> None:
        turns = [replace(self.turn, cache_state=state) for state in CacheState]
        results = compare_turns(iter(turns), self.model)
        self.assertEqual(len(results), 12)
        for index, turn in enumerate(turns):
            group_start = index * 3
            group = results[group_start : group_start + 3]
            method_names = [result.method for result in group]
            self.assertEqual(method_names, ["Serial", "PPD", "Ours"])
            for result in group:
                self.assertIs(result.turn, turn)

    def test_affected_threshold_is_strict_and_metrics_exclude_first_turns(self) -> None:
        result = serial(self.turn, self.model)
        boundary = replace(result, ttft_seconds=2 * result.reference_seconds)
        affected = replace(result, ttft_seconds=3 * result.reference_seconds)
        first_turn = Turn(512, 512, CacheState.FF, is_returning=False)
        first = replace(affected, turn=first_turn)
        self.assertFalse(boundary.is_affected())
        summary = summarize([boundary, affected, first])
        self.assertEqual(summary["returning_turns"], 2)
        self.assertEqual(summary["affected_turns"], 1)
        self.assertEqual(summary["affected_share"], 0.5)
        self.assertEqual(summary["median_affected_slowdown"], 3)
        self.assertAlmostEqual(summary["summed_ttft_inflation"], 1.5)
        self.assertFalse(affected.is_affected(3))

    def test_no_data_and_no_affected_are_distinct(self) -> None:
        self.assertIsNone(summarize([])["affected_share"])
        turn = replace(self.turn, cache_state=CacheState.HH)
        hit = ppd(turn, self.model)
        summary = summarize([hit])
        self.assertEqual(summary["affected_share"], 0)
        self.assertIsNone(summary["median_affected_slowdown"])

    def test_summed_inflation_weights_reference_time_instead_of_averaging_ratios(
        self,
    ) -> None:
        result = serial(self.turn, self.model)
        short = replace(result, reference_seconds=0.01, ttft_seconds=0.02)
        long = replace(result, reference_seconds=0.02, ttft_seconds=0.08)
        summary = summarize([short, long])
        self.assertAlmostEqual(summary["summed_ttft_inflation"], 0.10 / 0.03 - 1)
        self.assertEqual(summary["affected_share"], 0.5)

    def test_reject_invalid_inputs_and_configuration(self) -> None:
        invalid_turn_fields = (
            {"context_tokens": 0},
            {"new_tokens": -1},
            {"new_tokens": 32_769},
            {"context_tokens": True},
            {"new_tokens": 1.5},
            {"cache_state": "HF?"},
            {"output_tokens": 0},
            {"is_returning": "false"},
            {"is_returning": False},
        )
        for changes in invalid_turn_fields:
            with self.subTest(turn=changes), self.assertRaises(ValueError):
                replace(self.turn, **changes)
        invalid_model_fields = (
            {"layers": 0},
            {"bandwidth_gb_s": 0},
            {"bandwidth_gb_s": float("nan")},
            {"kv_bytes_per_token": -1},
            {"prefill_utilization": 1.1},
        )
        for changes in invalid_model_fields:
            with self.subTest(config=changes), self.assertRaises(ValueError):
                replace(self.model, **changes)
        for threshold in (1, 0, float("inf"), float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                summarize([], threshold)


class CommandLineTests(unittest.TestCase):
    def test_demo_generates_comparable_json_and_correct_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            command = subprocess.run(
                [sys.executable, "-m", "algorithm", "--demo", "--output", str(output)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(command.returncode, 0, command.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["synthetic_demo"])
            self.assertEqual(report["input_turns"], 5)
            self.assertEqual(len(report["results"]), 15)
            returning_counts = []
            for summary in report["metrics"].values():
                returning_counts.append(summary["returning_turns"])
            self.assertEqual(returning_counts, [4, 4, 4])

            ours_by_state = report["metrics"]["Ours"]["by_cache_state"]
            self.assertEqual(ours_by_state["HF"]["affected_turns"], 0)

    def test_raw_trace_does_not_silently_become_a_normalized_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            raw_request = {
                "timestamp": 0,
                "input_length": 512,
                "output_length": 32,
                "hash_ids": [1],
            }
            source.write_text(json.dumps(raw_request) + "\n")
            command = subprocess.run(
                [sys.executable, "-m", "algorithm", "--input", str(source)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(command.returncode, 0)
            self.assertIn(f"{source}:1:", command.stderr)


if __name__ == "__main__":
    unittest.main()
