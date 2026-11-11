"""Alignment geometry, native capacities, and timing provenance boundaries."""

from dataclasses import replace
import contextlib
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from native_methods.hardware_profile import (
    ROOT,
    load_profile as read_profile,
    validate_splitwise_timing,
)
from native_methods.tests.profile_fixtures import write_timing_fixture

sys.path.insert(0, str(ROOT / "TokenSim"))
sys.path.insert(0, str(ROOT / "splitwise-sim"))
LEGACY = ROOT / "native_methods/profiles/llama3-8b-h100-tp1.json"


def load_profile():
    return read_profile(LEGACY)


class HardwareProfileTests(unittest.TestCase):
    def test_splitwise_instances_use_one_gpu_and_physical_budget(self):
        from native_method_runtime import MethodSimulation

        deployment = load_profile()
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            simulation = MethodSimulation(
                [],
                "Serial",
                write_timing_fixture(directory, deployment),
                Path(directory),
                deployment=deployment,
            )
            try:
                self.assertEqual(len(simulation.devices), 8)
                for instance in simulation.devices:
                    self.assertEqual(len(instance.processors), 1)
                    self.assertEqual(
                        instance.processors[0].memory_size, deployment.gpu_memory_bytes
                    )
                    self.assertEqual(instance.model.architecture.num_layers, 32)
                    self.assertEqual(instance.model.parallelism.tensor_parallelism, 1)
                    self.assertEqual(
                        instance.model.size.total_size, deployment.weights_bytes
                    )
                    self.assertEqual(
                        instance.cache.capacity_bytes, deployment.kv_capacity_bytes
                    )
                    self.assertEqual(
                        instance.memory,
                        deployment.weights_bytes + deployment.runtime_reserved_bytes,
                    )
            finally:
                simulation.finish()

    def test_native_catalog_memory_and_geometry_match(self):
        from TokenSim.hardware import HardwareContext
        from TokenSim.config.cache_config import CacheConfig

        deployment = load_profile()
        hardware = HardwareContext.load(ROOT / "TokenSim/data")
        model = hardware.model(deployment.model_id)
        gpu = hardware.device(deployment.gpu_id)
        cache = CacheConfig(
            16, gpu, model, max_num_batched_tokens=4096, max_num_seqs=16
        )
        self.assertEqual(model.total_params() * 2, deployment.weights_bytes)
        self.assertEqual(cache.num_gpu_blocks, deployment.kv_blocks)
        self.assertEqual(cache.size_per_token, deployment.kv_bytes_per_token)
        self.assertEqual(deployment.total_gpus, 8)
        self.assertEqual(deployment.kv_blocks, 31314)
        self.assertEqual(deployment.kv_bytes_per_token, 131072)
        self.assertLessEqual(
            deployment.weights_bytes
            + deployment.runtime_reserved_bytes
            + deployment.kv_capacity_bytes,
            deployment.gpu_memory_bytes,
        )

    def test_unsupported_parallelism_and_memory_fail_before_execution(self):
        deployment = load_profile()
        for changes in (
            dict(tensor_parallel=8),
            dict(tensor_parallel=True),
            dict(gpus_per_instance=8),
            dict(gpu_memory_bytes=deployment.weights_bytes),
            dict(block_tokens=32),
            dict(model_dtype="fp8"),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(deployment, **changes)

    def test_missing_aligned_table_does_not_use_historical_timings(self):
        from native_method_runtime import execute, PinnedDatabase

        with tempfile.TemporaryDirectory() as directory, patch(
            "native_method_runtime.ROOT", Path(directory)
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Aligned Splitwise timing"):
                execute([], "Serial", 0, directory, hardware_profile=LEGACY)
            self.assertFalse(list(Path(directory).iterdir()))
        with self.assertRaisesRegex(ValueError, "no rows"):
            PinnedDatabase(ROOT / "splitwise-sim/data/perf_model.csv", load_profile())

    def test_metadata_identity_and_checksum_are_required(self):
        deployment = load_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = write_timing_fixture(directory, deployment)
            metadata_path = path.with_suffix(".metadata.json")
            metadata = dict(
                model=deployment.model_id,
                hardware=deployment.splitwise_hardware,
                tensor_parallel=1,
                model_dtype="bf16",
                kv_dtype="bf16",
                gpu_id=deployment.gpu_id,
                time_units="milliseconds",
                kind="analytical",
                source="Synthetic unit-test fixture; not experimental measurements",
                csv_sha256=sha256(path.read_bytes()).hexdigest(),
            )
            metadata_path.write_text(json.dumps(metadata))
            self.assertEqual(
                validate_splitwise_timing(path, deployment)["kind"], "analytical"
            )
            for changes in (
                dict(model="llama2-70b"),
                dict(tensor_parallel=8),
                dict(model_dtype="fp16"),
                dict(time_units="seconds"),
                dict(csv_sha256="invalid"),
                dict(source=""),
                dict(kind="unknown"),
            ):
                metadata_path.write_text(json.dumps(dict(metadata, **changes)))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_splitwise_timing(path, deployment)

    def test_three_native_methods_record_h100_and_aligned_capacity(self):
        spec = importlib.util.spec_from_file_location(
            "aligned_token_test", ROOT / "TokenSim/TokenSim/research/native_runtime.py"
        )
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
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
            )
        ]
        for method in ("Serial", "PPD", "MigrationTax"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                result = backend.execute(
                    rows, method, 0, directory, hardware_profile=LEGACY
                )
                self.assertEqual(result["status"], "completed")
                effective = json.loads(
                    (Path(directory) / "effective-config.json").read_text()
                )
                self.assertEqual(
                    effective["hardware_alignment"], load_profile().describe()
                )
                self.assertTrue(
                    all(
                        g["hardware"] == "h100_sxm"
                        for g in effective["cluster"]["worker_groups"]
                    )
                )
