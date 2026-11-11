"""共同資料與公式。內部時間一律用秒，資料量一律用 bytes。

對應稿件 §III-A、§III-B、§V-A。這一層接受已知 cache 狀態，
不自行推算 LRU、裝置路由、排隊或原始 trace 的對話關係。
"""

from dataclasses import asdict, dataclass
from enum import Enum
import math


class CacheState(str, Enum):
    """第一碼看 P，第二碼看 D；H 為命中，F 為未命中。"""

    HH = "HH"
    FH = "FH"
    HF = "HF"
    FF = "FF"

    @property
    def prefill_hit(self) -> bool:
        return self.value[0] == "H"

    @property
    def decode_hit(self) -> bool:
        return self.value[1] == "H"


def require_integer(name: str, value: int, minimum: int) -> None:
    # bool 在 Python 是 int 的子類別，但不是有效的 token 數。
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Turn:
    """單輪共同輸入；new_tokens 是論文的 ΔC，不只是使用者新打的文字。"""

    context_tokens: int
    new_tokens: int
    cache_state: CacheState
    output_tokens: int = 1
    is_returning: bool = True
    request_id: str = ""

    def __post_init__(self) -> None:
        require_integer("context_tokens", self.context_tokens, 1)
        require_integer("new_tokens", self.new_tokens, 0)
        require_integer("output_tokens", self.output_tokens, 1)
        if self.new_tokens > self.context_tokens:
            raise ValueError("new_tokens cannot exceed context_tokens")
        if type(self.is_returning) is not bool:
            raise ValueError("is_returning must be a boolean")
        if not isinstance(self.request_id, str):
            raise ValueError("request_id must be a string")
        try:
            state = CacheState(self.cache_state)
        except (ValueError, TypeError) as error:
            raise ValueError("cache_state must be HH, FH, HF, or FF") from error
        object.__setattr__(self, "cache_state", state)
        if not self.is_returning and (
            self.prefix_tokens != 0 or state != CacheState.FF
        ):
            raise ValueError(
                "a first turn must have new_tokens=context_tokens and state FF"
            )

    @property
    def prefix_tokens(self) -> int:
        return self.context_tokens - self.new_tokens


@dataclass(frozen=True)
class CostModel:
    """稿件正文的預設常數。GB、MB 採十進位；所有設定皆可明確覆寫。"""

    parameters: int = 6_700_000_000
    layers: int = 32
    hidden_size: int = 4_096
    kv_bytes_per_token: int = 524_000
    bandwidth_gb_s: float = 100.0
    prefill_peak_tflops: float = 384.0
    prefill_utilization: float = 0.70
    decode_weight_seconds: float = 0.00291
    decode_attention_seconds_per_token: float = 0.000000207

    def __post_init__(self) -> None:
        integer_fields = ("parameters", "layers", "hidden_size", "kv_bytes_per_token")
        for name in integer_fields:
            require_integer(name, getattr(self, name), 1)

        positive_number_fields = (
            "bandwidth_gb_s",
            "prefill_peak_tflops",
            "prefill_utilization",
            "decode_weight_seconds",
            "decode_attention_seconds_per_token",
        )
        for name in positive_number_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.prefill_utilization > 1:
            raise ValueError("prefill_utilization cannot exceed 1")

    def prefill_seconds(self, turn: Turn) -> float:
        """PF(ΔC,C)：projection 與 attention 的 operation count / 有效算力。"""
        projection_operations = 2 * self.parameters * turn.new_tokens
        attention_operations = (
            4 * self.layers * self.hidden_size * turn.new_tokens * turn.context_tokens
        )
        achieved_flops = self.prefill_peak_tflops * 1e12 * self.prefill_utilization
        total_operations = projection_operations + attention_operations
        return total_operations / achieved_flops

    def move_seconds(self, tokens: int) -> float:
        """MV(n)：n 個 token 的 KV bytes / 接收頻寬。"""
        require_integer("tokens", tokens, 0)
        transfer_bytes = tokens * self.kv_bytes_per_token
        bandwidth_bytes_per_second = self.bandwidth_gb_s * 1e9
        return transfer_bytes / bandwidth_bytes_per_second

    def decode_step_seconds(self, context_tokens: int) -> float:
        """ST(C)：讀權重的固定時間，加上隨 context 增長的 KV 讀取時間。"""
        require_integer("context_tokens", context_tokens, 1)
        attention_seconds = self.decode_attention_seconds_per_token * context_tokens
        return self.decode_weight_seconds + attention_seconds

    def reference_seconds(self, turn: Turn) -> float:
        """同一輪的零搬移參考值；不改變 cache 狀態。"""
        prefill_seconds = self.prefill_seconds(turn)
        first_step_seconds = self.decode_step_seconds(turn.context_tokens)
        return prefill_seconds + first_step_seconds

    def pipeline_cache_ready_seconds(
        self,
        turn: Turn,
        store_to_prefill_tokens: int,
        prefill_to_decode_tokens: int,
    ) -> float:
        """Eq. 2 的 pipeline 部分：最後一層 KV 到達 D 的時間。

        各層大小與計算成本均等；三個階段使用獨立資源，不含競爭。
        呼叫端還要加上 D 的第一個 decode step。
        """
        store_per_layer = self.move_seconds(store_to_prefill_tokens) / self.layers
        compute_per_layer = self.prefill_seconds(turn) / self.layers
        send_per_layer = self.move_seconds(prefill_to_decode_tokens) / self.layers
        fill_and_drain = store_per_layer + compute_per_layer + send_per_layer
        slowest_stage = max(
            store_per_layer,
            compute_per_layer,
            send_per_layer,
        )
        remaining_layers = self.layers - 1
        steady_state = remaining_layers * slowest_stage
        return fill_and_drain + steady_state


def serial_transfer_tokens(turn: Turn) -> tuple[int, int]:
    """§III-B：由相同 arrival cache 狀態決定 Serial 的兩段搬移量。"""
    if turn.cache_state.prefill_hit:
        store_to_prefill = 0
    else:
        store_to_prefill = turn.prefix_tokens

    if turn.cache_state.decode_hit:
        prefill_to_decode = turn.new_tokens
    else:
        prefill_to_decode = turn.context_tokens

    return store_to_prefill, prefill_to_decode


@dataclass(frozen=True)
class TurnResult:
    """單輪第一個 token 的結果，不代表完整回覆或 handoff 的完成時間。

    搬移量只計 Eq. 1 的 prefix／input KV；HF 在 P 上產生的新 output KV
    仍需交接，但不在本輪 TTFT 路徑上，本物件不估計其傳輸量或 k。
    """

    turn: Turn
    method: str
    path: str
    prefill_device: str
    first_decode_device: str
    store_to_prefill_tokens: int
    prefill_to_decode_tokens: int
    ttft_seconds: float
    reference_seconds: float

    @property
    def slowdown(self) -> float:
        return self.ttft_seconds / self.reference_seconds

    @property
    def movement_stall_seconds(self) -> float:
        return self.ttft_seconds - self.reference_seconds

    def is_affected(self, threshold: float = 2.0) -> bool:
        # 嚴格大於，剛好兩倍不算；避免先除法再比較帶來額外捨入。
        return self.ttft_seconds > threshold * self.reference_seconds

    def to_dict(self, threshold: float = 2.0) -> dict:
        result = asdict(self)
        result.update(
            slowdown=self.slowdown,
            movement_stall_seconds=self.movement_stall_seconds,
            affected=self.is_affected(threshold),
        )
        return result
