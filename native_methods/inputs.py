"""Attach explicit source ancestry to checksum-locked normalized timelines."""

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mooncake_parents(raw_path):
    from migrationtax.reference.events import event_row

    raw = [json.loads(s) for s in raw_path.open() if s.strip()]
    raw.sort(key=lambda r: r["timestamp"])
    origin = raw[0]["timestamp"] / 1000
    latest, generated, parent_ids = {}, [], []
    conversation = 0
    for index, request in enumerate(raw):
        parent_index, shared = None, 0
        for depth, block in enumerate(request["hash_ids"], 1):
            if block not in latest:
                break
            parent_index, shared = latest[block], depth
        if (
            parent_index is None
            or shared < 2
            or shared < len(raw[parent_index]["hash_ids"]) - 1
        ):
            parent_index = None
        parent = None if parent_index is None else generated[parent_index]
        cid, turn = (
            (conversation, 0)
            if parent is None
            else (parent["conv_id"], parent["turn"] + 1)
        )
        if parent is None:
            conversation += 1
        generated.append(
            event_row(
                cid,
                turn,
                request["timestamp"] / 1000,
                origin,
                request["input_length"],
                request["output_length"],
                parent,
            )
        )
        parent_ids.append(parent_index)
        for block in request["hash_ids"]:
            latest[block] = index
    order = sorted(
        range(len(raw)),
        key=lambda i: (
            generated[i]["t_arrival_s"],
            generated[i]["conv_id"],
            generated[i]["turn"],
        ),
    )
    positions = {old: new for new, old in enumerate(order)}
    return {
        positions[i]: None if p is None else positions[p]
        for i, p in enumerate(parent_ids)
    }


def load_requests(path, trace, *, limit=None):
    rows = [json.loads(s) for s in Path(path).open()]
    if trace == "mooncake":
        parents = mooncake_parents(ROOT / "Trace/mooncake/conversation_trace.jsonl")
        by_event = {r["source_order"]: i for i, r in enumerate(rows)}
    else:
        parents = None
        by_turn = {(r["conversation"], r["turn"]): i for i, r in enumerate(rows)}
    selected = rows if limit is None else rows[:limit]
    for index, row in enumerate(selected):
        if any(k in row for k in ("cell", "state", "P", "D", "expected_cell")):
            raise ValueError("Cache state and device answers are forbidden inputs")
        if trace == "mooncake":
            parent_event = parents[row["source_order"]]
            parent = None if parent_event is None else by_event[parent_event]
        else:
            parent = (
                None
                if row["turn"] == 0
                else by_turn[(row["conversation"], row["turn"] - 1)]
            )
        if parent is not None:
            if parent >= index or rows[parent]["conversation"] != row["conversation"]:
                raise ValueError(
                    "Ancestry must precede this request on the unchanged timeline"
                )
            if (
                row["context_tokens"] - row["new_tokens"]
                > rows[parent]["context_tokens"]
            ):
                raise ValueError(
                    "Inherited input prefix exceeds its actual parent input"
                )
        row.update(
            source_id=str(index),
            previous_source_id=None if parent is None else str(parent),
        )
    return selected


def manifest(path, trace, rows):
    return dict(
        source_file=str(Path(path).resolve()),
        sha256=sha256(Path(path).read_bytes()).hexdigest(),
        trace=trace,
        requests=len(rows),
        output_tokens=sum(r["output_tokens"] for r in rows),
        source_arrivals_preserved=True,
        source_turns_preserved=True,
        ancestry="Mooncake pinned prefix-trie parent mapped through source_order; ServeGen explicit prior turn",
        author_answers_read=False,
    )
