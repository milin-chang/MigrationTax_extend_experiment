"""Shared physical configuration; each simulator retains its native timing."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "native_methods/profiles/llama2-70b-h100-tp8.json"


@dataclass(frozen=True)
class HardwareProfile:
    model_id: str
    model_dtype: str
    kv_dtype: str
    model_parameters: int
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    gpu_id: str
    splitwise_hardware: str
    gpu_memory_bytes: int
    runtime_reserved_bytes: int
    gpus_per_instance: int
    tensor_parallel: int
    pipeline_parallel: int
    prefill_instances: int
    decode_instances: int
    block_tokens: int
    splitwise_profile: str
    splitwise_model_id: str = ""
    _source_sha256: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if (
            self.gpus_per_instance != self.tensor_parallel
            or self.tensor_parallel not in (1, 2, 4, 8)
            or self.pipeline_parallel != 1
        ):
            raise ValueError(
                "Aligned replicas require TP1/2/4/8, PP1 and one GPU per TP rank"
            )
        if (self.prefill_instances, self.decode_instances) != (4, 4):
            raise ValueError("This adapter currently requires a fixed 4P+4D topology")
        if self.model_dtype not in ("fp16", "bf16") or self.kv_dtype not in (
            "fp16",
            "bf16",
        ):
            raise ValueError("Aligned profiles require FP16/BF16 weights and KV")
        if self.block_tokens not in (16, 64):
            raise ValueError(
                "The native method adapters require 16-token or 64-token blocks"
            )
        for name in (
            "model_parameters",
            "layers",
            "hidden_size",
            "attention_heads",
            "kv_heads",
            "head_dim",
            "gpu_memory_bytes",
            "block_tokens",
            "gpus_per_instance",
            "tensor_parallel",
            "pipeline_parallel",
            "prefill_instances",
            "decode_instances",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.runtime_reserved_bytes) is not int
            or self.runtime_reserved_bytes < 0
        ):
            raise ValueError("Runtime memory reserve must be nonnegative")
        if (
            self.hidden_size != self.attention_heads * self.head_dim
            or self.attention_heads % self.kv_heads
            or self.kv_heads % self.tensor_parallel
        ):
            raise ValueError("Inconsistent attention geometry")
        if self.kv_blocks <= 0:
            raise ValueError("GPU memory cannot hold the model and one KV block")
        relative = Path(self.splitwise_profile)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Timing profile must be relative to the checkout")

    @property
    def weights_bytes(self):
        return self.model_parameters * 2

    @property
    def kv_bytes_per_token(self):
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    @property
    def kv_blocks(self):
        available = (
            self.gpu_memory_bytes
            - self.runtime_reserved_bytes
            - self.weights_bytes_per_gpu
        )
        return available // (self.block_tokens * self.kv_bytes_per_token_per_gpu)

    @property
    def weights_bytes_per_gpu(self):
        return self.weights_bytes // self.tensor_parallel

    @property
    def kv_bytes_per_token_per_gpu(self):
        return self.kv_bytes_per_token // self.tensor_parallel

    @property
    def splitwise_model(self):
        return self.splitwise_model_id or self.model_id

    @property
    def kv_capacity_bytes(self):
        return self.kv_blocks * self.block_tokens * self.kv_bytes_per_token

    @property
    def total_gpus(self):
        return (self.prefill_instances + self.decode_instances) * self.gpus_per_instance

    def describe(self):
        return dict(
            {
                key: value
                for key, value in asdict(self).items()
                if not key.startswith("_")
            },
            weights_bytes=self.weights_bytes,
            weights_bytes_per_gpu=self.weights_bytes_per_gpu,
            kv_bytes_per_token=self.kv_bytes_per_token,
            kv_bytes_per_token_per_gpu=self.kv_bytes_per_token_per_gpu,
            kv_blocks=self.kv_blocks,
            kv_capacity_bytes=self.kv_capacity_bytes,
            total_gpus=self.total_gpus,
            source_sha256=self._source_sha256,
            scope="Model, physical GPUs and raw KV capacity; native admission, store and scheduling policies remain simulator-specific",
        )


def load_profile(path=None):
    data = Path(path or SPEC_PATH).read_bytes()
    profile = HardwareProfile(**json.loads(data))
    object.__setattr__(profile, "_source_sha256", sha256(data).hexdigest())
    return profile


def validate_splitwise_timing(path, spec):
    """Require an explicitly sourced table for this exact deployment."""
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Aligned Splitwise timing data is missing: provide "
            + str(path)
            + " and "
            + str(metadata_path)
            + ". The historical Llama2-70B table cannot supply aligned 8B timings."
        )
    metadata = json.loads(metadata_path.read_text())
    expected = dict(
        model=spec.splitwise_model,
        hardware=spec.splitwise_hardware,
        tensor_parallel=spec.tensor_parallel,
        model_dtype=spec.model_dtype,
        kv_dtype=spec.kv_dtype,
        gpu_id=spec.gpu_id,
        time_units="milliseconds",
    )
    for field, value in expected.items():
        if metadata.get(field) != value or type(metadata.get(field)) is not type(value):
            raise ValueError("Timing metadata does not match aligned " + field)
    if (
        metadata.get("kind") not in ("measurement", "analytical", "bundled")
        or not isinstance(metadata.get("source"), str)
        or not metadata["source"].strip()
    ):
        raise ValueError("Timing metadata must identify its kind and nonempty source")
    if metadata.get("csv_sha256") != sha256(path.read_bytes()).hexdigest():
        raise ValueError("Timing CSV checksum does not match its metadata")
    return dict(
        metadata, metadata_sha256=sha256(metadata_path.read_bytes()).hexdigest()
    )
