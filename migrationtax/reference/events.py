"""公開 raw trace → events；轉換函式不讀取作者 events 或 cache dump。"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "conv_id", "turn", "t_arrival_s", "t_rel_s", "input_tokens",
    "output_tokens", "C_t", "dC_t", "K_C_GB", "K_dC_GB", "gap_s", "cumulative",
)


def event_row(conversation: int, turn: int, arrival: float, origin: float,
              context: int, output: int, parent: dict | None) -> dict:
    delta = context
    gap = ""
    cumulative = True
    if parent is not None:
        delta = context - parent["C_t"]
        gap = arrival - parent["t_arrival_s"]
        cumulative = context >= parent["C_t"] + parent["output_tokens"]
    return {
        "conv_id": conversation, "turn": turn,
        "t_arrival_s": arrival, "t_rel_s": arrival - origin,
        "input_tokens": context, "output_tokens": output,
        "C_t": context, "dC_t": delta,
        "K_C_GB": context * 0.000524,
        "K_dC_GB": max(delta, 0) * 0.000524,
        "gap_s": gap, "cumulative": cumulative,
    }


def convert_mooncake(path: Path) -> list[dict]:
    """依完整 prefix 選 parent，允許最後一個未滿 block 隨新增文字改變。

    每個 hash 已包含之前所有 blocks（官方 Appendix A.2），因此 hash ID
    可直接代表 trie 節點。節點保留最近通過的 request；先找最深共用節點，
    再檢查是否覆蓋 parent 除最後一個 block 外的全部 prefix。
    只有共同的第一個 block 不足以判成同一對話。

    這是由資料關係推定的轉換，不代表已取得或核定作者原始程式。
    """
    with path.open(encoding="utf-8") as source:
        raw = [json.loads(line) for line in source if line.strip()]
    raw.sort(key=lambda row: row["timestamp"])
    if not raw:
        raise ValueError("Mooncake raw trace is empty")
    origin = raw[0]["timestamp"] / 1000
    latest_at_prefix = {}
    generated = []
    next_conversation = 0
    for source_index, request in enumerate(raw):
        parent_index = None
        shared_blocks = 0
        for depth, block in enumerate(request["hash_ids"], 1):
            if block not in latest_at_prefix:
                break
            parent_index = latest_at_prefix[block]
            shared_blocks = depth

        parent = None
        if parent_index is not None and shared_blocks >= 2:
            parent_block_count = len(raw[parent_index]["hash_ids"])
            if shared_blocks >= parent_block_count - 1:
                parent = generated[parent_index]

        if parent is None:
            conversation = next_conversation
            next_conversation += 1
            turn = 0
        else:
            conversation = parent["conv_id"]
            turn = parent["turn"] + 1
        generated.append(event_row(
            conversation, turn, request["timestamp"] / 1000, origin,
            request["input_length"], request["output_length"], parent,
        ))
        for block in request["hash_ids"]:
            latest_at_prefix[block] = source_index

    generated.sort(key=lambda row: (row["t_arrival_s"], row["conv_id"], row["turn"]))
    return generated


def convert_servegen(path: Path) -> list[dict]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    timestamps = [float(turn["timestamp"]) for conversation in conversations for turn in conversation]
    if not timestamps:
        raise ValueError("ServeGen raw trace is empty")
    origin = min(timestamps)
    generated = []
    for conversation, turns in enumerate(conversations):
        parent = None
        for turn, request in enumerate(turns):
            row = event_row(
                conversation, turn, float(request["timestamp"]), origin,
                request["input_token_count"], request["output_token_count"], parent,
            )
            generated.append(row)
            parent = row
    return generated


def compare_events(generated: list[dict], reference: Path) -> dict:
    """只在轉換完成後比較；整數／布林／空值須相同，浮點誤差另列。"""
    with reference.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("reference events columns differ")
        targets = list(reader)
    mismatches = Counter()
    first_mismatches = []
    float_errors = {}
    for index, (row, target) in enumerate(zip(generated, targets)):
        for field in FIELDS:
            value = row[field]
            expected = target[field]
            if isinstance(value, bool):
                same = str(value) == expected
            elif value == "" or expected == "":
                same = value == expected
            elif isinstance(value, int):
                same = value == int(expected)
            else:
                error = abs(value - float(expected))
                float_errors[field] = max(float_errors.get(field, 0.0), error)
                tolerance = 1e-12 if field.startswith("K_") else 1e-9
                same = math.isclose(value, float(expected), abs_tol=tolerance, rel_tol=0)
            if not same:
                mismatches[field] += 1
                if len(first_mismatches) < 5:
                    first_mismatches.append({"row_index": index, "field": field, "generated": value, "reference": expected})
    row_difference = len(generated) - len(targets)
    return {
        "generated_rows": len(generated), "reference_rows": len(targets),
        "row_count_difference": row_difference,
        "mismatches_by_column": dict(mismatches),
        "max_float_errors": float_errors,
        "all_values_match": row_difference == 0 and not mismatches,
        "first_mismatches": first_mismatches,
        "time_tolerance_seconds": 1e-9, "kv_tolerance_GB": 1e-12,
        "reference_sha256": sha256(reference.read_bytes()).hexdigest(),
    }


def generate_event_files(trace_dir: Path, output_dir: Path,
                         reference_dir: Path | None = None) -> dict:
    """轉換與保存兩份 CSV；不提供 reference_dir 時完全不讀作者資料。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prediction_inputs": "public raw traces only",
        "mooncake_rule_status": "inferred prefix-parent rule; author source identity unverified",
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "traces": {},
    }
    converters = (
        ("mooncake", "mooncake/conversation_trace.jsonl", convert_mooncake),
        ("servegen", "servegen/conversations_hashed.json", convert_servegen),
    )
    for name, source_name, convert in converters:
        source = trace_dir / source_name
        rows = convert(source)
        output = output_dir / f"events_{name}.csv"
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "source": str(source), "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "output": str(output), "output_sha256": sha256(output.read_bytes()).hexdigest(),
            "rows": len(rows), "conversations": len({row["conv_id"] for row in rows}),
        }
        if reference_dir is not None:
            reference = reference_dir / output.name
            result["comparison"] = compare_events(rows, reference)
            result["byte_identical_to_reference"] = output.read_bytes() == reference.read_bytes()
        manifest["traces"][name] = result
        print(name, result["rows"], result.get("comparison", {}).get("all_values_match", "not compared"), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="只從公開 raw trace 重建兩份 events CSV")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "Trace")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/events")
    parser.add_argument("--reference-dir", type=Path, help="可選；僅在轉換完成後讀取作者 events 比較")
    args = parser.parse_args()
    generate_event_files(args.trace_dir, args.output_dir, args.reference_dir)


if __name__ == "__main__":
    main()
