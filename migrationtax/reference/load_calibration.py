"""由明列的 workload rate 上限計算副本倍率；不宣稱已獨立推導該上限。"""

import csv
from hashlib import sha256
import math
from pathlib import Path


def derive_repeat_factor(events: Path, target_load: float, wall_rps: float) -> tuple[float, dict]:
    for name, value in (("target_load", target_load), ("wall_rps", wall_rps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    with events.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("load calibration events are empty")
    times = [float(row["t_rel_s"]) for row in rows]
    duration = max(times) - min(times)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("event time window must be finite and positive")
    original_rps = len(rows) / duration
    target_rps = target_load * wall_rps
    factor = target_rps / original_rps
    if factor < 1:
        raise ValueError("requested rate requires downsampling; this converter only superposes copies")
    return factor, {
        "mode": "supplied_WALL_calibration",
        "formula": "target_load * wall_rps * observed_duration_seconds / original_rows",
        "target_load": target_load, "wall_rps": wall_rps,
        "observed_duration_seconds": duration, "original_rows": len(rows),
        "original_rps": original_rps, "target_rps": target_rps,
        "repeat_factor": factor, "WALL_independently_derived_from_hardware": False,
        "events_file": str(events), "events_sha256": sha256(events.read_bytes()).hexdigest(),
        "version_caveat": "MigrationTax reports load .8; X1/M1 metadata records frac .53 / '.53 x WALL'. The earlier fixed WALL value reproduces inputs, but its version and hardware derivation remain unverified.",
    }
