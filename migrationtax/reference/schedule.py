"""Events 到請求時間軸；分組與位移規則在固定資料上已核對。

本模組不讀 cache-state dump。ServeGen 的副本倍率由呼叫端明確提供，
不把已對齊的時間軸視為 WALL 硬體推導已完成。
"""

import csv
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

@dataclass(frozen=True)
class Request:
    conversation: int
    turn: int
    arrival_seconds: float
    context_tokens: int
    new_tokens: int
    output_tokens: int
    source_order: int = 0



def load_mooncake_events(package: Path, wrap_start: bool, *, events_dir: Path | None = None) -> tuple[list[Request], dict]:
    directory = package / "events" if events_dir is None else events_dir
    path = directory / "events_mooncake.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Mooncake events are empty")
    duration = max(float(row["t_rel_s"]) for row in rows)
    if duration <= 0:
        raise ValueError("Mooncake event window must be positive")
    first_times = {}
    for row in rows:
        cid = int(row["conv_id"])
        arrival = float(row["t_rel_s"])
        first_times[cid] = min(first_times.get(cid, arrival), arrival)

    requests = []
    changed_times = 0
    for source_order, row in enumerate(rows):
        cid = int(row["conv_id"])
        original_time = float(row["t_rel_s"])
        first_time = first_times[cid]
        if wrap_start:
            start = first_time % duration
        else:
            start = first_time
        # 括號保留原始相對間隔的浮點求值順序，避免同時刻事件順序被捨入改變。
        arrival = start + (original_time - first_time)
        if arrival != original_time:
            changed_times += 1
        context = int(row["C_t"])
        new_tokens = max(0, min(context, int(row["dC_t"])))
        requests.append(Request(
            conversation=cid, turn=int(row["turn"]), arrival_seconds=arrival,
            context_tokens=context, new_tokens=new_tokens,
            output_tokens=int(row["output_tokens"]),
            source_order=source_order,
        ))
    requests.sort(key=lambda request: (request.arrival_seconds, request.conversation, request.turn))
    return requests, {
        "duration_seconds": duration, "wrap_conversation_start": wrap_start,
        "changed_timestamps": changed_times,
        "source_events_sha256": sha256(path.read_bytes()).hexdigest(),
    }



def load_servegen_events(package: Path, seed: int, repeat_factor: float, *, events_dir: Path | None = None) -> tuple[list[Request], dict]:
    """從 events 生成副本時間軸，不讀取 dump 的 cell、phit 或 dmiss。

    seed+1000 的 PCG64 位移序列已從四個作者時間軸獨立核對。
    repeat_factor 必須明確提供；目前由作者請求成員推定的 6.9411
    尚不是論文公開的 load 換算公式，不能冒充已驗證的負載規格。
    """
    import numpy as np
    if not math.isfinite(repeat_factor) or repeat_factor < 1:
        raise ValueError("repeat_factor must be finite and >= 1")
    directory = package / "events" if events_dir is None else events_dir
    path = directory / "events_servegen.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("ServeGen events are empty")
    groups = {}
    for source_order, row in enumerate(rows):
        cid = int(row["conv_id"])
        groups.setdefault(cid, []).append((source_order, row))
    duration = max(float(row["t_rel_s"]) for row in rows)
    if duration <= 0:
        raise ValueError("ServeGen event window must be positive")
    rng = np.random.default_rng(1000 + seed)
    full_copies = math.floor(repeat_factor)
    fractional_copy = repeat_factor - full_copies
    copy_count = math.ceil(repeat_factor)
    requests = []
    shifts = []
    counts = []
    for copy_index in range(copy_count):
        shift = 0.0
        if copy_index > 0:
            shift = float(rng.uniform(0, duration))
        shifts.append(shift)
        selected_count = 0
        for cid in sorted(groups):
            if copy_index == full_copies:
                if float(rng.random()) >= fractional_copy:
                    continue
            selected_count += 1
            source_events = groups[cid]
            first_time = float(source_events[0][1]["t_rel_s"])
            start = (first_time + shift) % duration
            for source_order, row in source_events:
                relative_time = float(row["t_rel_s"]) - first_time
                context = int(row["C_t"])
                new_tokens = max(0, min(context, int(row["dC_t"])))
                requests.append(Request(
                    conversation=copy_index * len(groups) + cid,
                    turn=int(row["turn"]), arrival_seconds=start + relative_time,
                    context_tokens=context, new_tokens=new_tokens,
                    output_tokens=int(row["output_tokens"]),
                    source_order=copy_index * len(rows) + source_order,
                ))
        counts.append(selected_count)
    requests.sort(key=lambda request: (request.arrival_seconds, request.conversation, request.turn))
    return requests, {
        "duration_seconds": duration, "repeat_factor": repeat_factor,
        "repeat_factor_load_formula_verified": False,
        "schedule_rng": "numpy PCG64 seeded with 1000 + seed",
        "numpy_version": np.__version__,
        "source_events_sha256": sha256(path.read_bytes()).hexdigest(),
        "shifts_seconds": shifts, "conversations_per_copy": counts,
    }
