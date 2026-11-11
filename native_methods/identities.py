"""Compact synthetic ancestry, without cross-conversation reuse assumptions."""

from dataclasses import dataclass
from hashlib import blake2b


@dataclass(frozen=True)
class Span:
    source: str
    start: int
    length: int


def prefix(spans, count):
    result = []
    for s in spans:
        n = min(count, s.length)
        if n:
            result.append(Span(s.source, s.start, n))
            count -= n
        if not count:
            break
    if count:
        raise ValueError("Prefix exceeds actual ancestor input")
    return result


def input_spans(rows):
    built = {}
    for r in rows:
        k = r["context_tokens"] - r["new_tokens"]
        parent = r["previous_source_id"]
        inherited = prefix(built[parent], k) if parent is not None else []
        if r["new_tokens"]:
            inherited.append(Span("input:" + r["source_id"], 0, r["new_tokens"]))
        built[r["source_id"]] = inherited
    return built


def block_hashes(spans, size):
    parts, count, result = [], 0, []
    for s in spans:
        offset = 0
        while offset < s.length:
            n = min(size - count, s.length - offset)
            parts.append((s.source, s.start + offset, n))
            count += n
            offset += n
            if count == size:
                result.append(
                    int.from_bytes(
                        blake2b(repr(parts).encode(), digest_size=8).digest(), "little"
                    )
                )
                parts, count = [], 0
    if count:
        result.append(
            int.from_bytes(
                blake2b(repr(parts).encode(), digest_size=8).digest(), "little"
            )
        )
    return result
