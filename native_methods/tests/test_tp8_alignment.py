"""Shared 70B/FP16 TP8 deployment with two independent native time models."""

import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from native_methods.hardware_profile import (
    ROOT,
    load_profile,
    validate_splitwise_timing,
)

sys.path.insert(0, str(ROOT / "TokenSim"))
sys.path.insert(0, str(ROOT / "splitwise-sim"))


class TP8AlignmentTests(unittest.TestCase):
    def test_bundled_profile_identity_and_payload_geometry(self):
        deployment = load_profile()
        timing = validate_splitwise_timing(
            ROOT / deployment.splitwise_profile, deployment
        )
        self.assertEqual(timing["kind"], "bundled")
        self.assertEqual(deployment.model_id, "llama-2-70b")
        self.assertEqual(deployment.model_dtype, "fp16")
        self.assertEqual(deployment.total_gpus, 64)
        self.assertEqual(deployment.kv_bytes_per_token, 327680)
        self.assertEqual(deployment.kv_bytes_per_token_per_gpu, 40960)
        self.assertEqual(deployment.kv_blocks, 98398)

    def test_splitwise_allocates_eight_gpus_per_instance(self):
        from native_method_runtime import MethodSimulation

        deployment = load_profile()
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            simulation = MethodSimulation(
                [], "Serial", ROOT / deployment.splitwise_profile, Path(directory)
            )
            try:
                self.assertEqual(sum(len(d.processors) for d in simulation.devices), 64)
                for device in simulation.devices:
                    self.assertEqual(device.model.parallelism.tensor_parallelism, 8)
                    self.assertEqual(
                        device.cache.capacity_bytes, deployment.kv_capacity_bytes
                    )
                    self.assertEqual(
                        device.memory,
                        deployment.weights_bytes
                        + 8 * deployment.runtime_reserved_bytes,
                    )
                    self.assertLessEqual(
                        device.memory + device.cache.capacity_bytes, device.max_memory
                    )
            finally:
                simulation.finish()

    def test_three_methods_record_identical_hardware_in_both_simulators(self):
        from native_method_runtime import execute as splitwise_execute
        from TokenSim.research.native_runtime import execute as token_execute

        rows = [
            dict(
                source_id="0",
                previous_source_id=None,
                conversation=0,
                turn=0,
                context_tokens=64,
                new_tokens=64,
                output_tokens=3,
                arrival_seconds=0.0,
            ),
            dict(
                source_id="1",
                previous_source_id="0",
                conversation=0,
                turn=1,
                context_tokens=81,
                new_tokens=17,
                output_tokens=5,
                arrival_seconds=1.0,
            ),
        ]
        for method in ("Serial", "PPD", "MigrationTax"):
            configs = []
            for backend in (splitwise_execute, token_execute):
                with self.subTest(
                    method=method, backend=backend.__module__
                ), tempfile.TemporaryDirectory() as directory:
                    result = backend(rows, method, 0, directory)
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["completed"], 2)
                    config = json.loads(
                        (Path(directory) / "effective-config.json").read_text()
                    )
                    configs.append(config["hardware_alignment"])
            self.assertEqual(configs[0], configs[1])


if __name__ == "__main__":
    unittest.main()
