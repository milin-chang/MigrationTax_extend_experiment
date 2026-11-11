"""Synthetic, short-duration native tables for behavior tests only."""

import csv
from pathlib import Path

from native_methods.hardware_profile import load_profile


def write_timing_fixture(directory, deployment=None):
    deployment = deployment or load_profile()
    path = Path(directory) / "synthetic-timing.csv"
    columns = [
        "model",
        "hardware",
        "tensor_parallel",
        "prompt_size",
        "batch_size",
        "token_size",
        "prompt_time",
        "token_time",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for prompt in (512, 2048, 32768):
            for batch in (1, 2, 4, 16):
                writer.writerow(
                    dict(
                        model=deployment.splitwise_model,
                        hardware=deployment.splitwise_hardware,
                        tensor_parallel=deployment.tensor_parallel,
                        prompt_size=prompt,
                        batch_size=batch,
                        token_size=128,
                        prompt_time=1 + prompt * batch / 1024,
                        token_time=1 + prompt * batch / 65536,
                    )
                )
    return path
