"""Bold displayed best values within each bandwidth without changing numbers."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

NOTE = "粗體比較同一頻寬下的 Serial、PPD、MigrationTax：延遲、TPOT 與 affected share 越低越好，吞吐量越高越好；以表中六位小數比較，並列者一併加粗。reference 僅作基準，不參與排名；decode miss 為診斷資料，不列最佳。"


def cells(line):
    return [value.strip().replace("**", "") for value in line.split("|")[1:-1]]


def emphasize(text):
    lines = text.splitlines()
    section, groups = "", {}
    for index, line in enumerate(lines):
        if line.startswith("## "):
            section = line
        if not line.startswith("| "):
            continue
        values = cells(line)
        if values[0] not in {"50", "100", "200"}:
            continue
        if section.startswith("## Affected share"):
            best = min(map(Decimal, values[1:]))
            values[1:] = [
                f"**{value}**" if Decimal(value) == best else value
                for value in values[1:]
            ]
            lines[index] = "| " + " | ".join(values) + " |"
        elif section == "## 請求與整批指標":
            groups.setdefault(values[0], []).append((index, values))
    for bandwidth, rows in groups.items():
        if len(rows) != 3 or {row[1][1] for row in rows} != {
            "Serial",
            "PPD",
            "MigrationTax",
        }:
            raise ValueError(
                "A bandwidth group does not contain all three methods: " + bandwidth
            )
        for column in range(2, 8):
            best = (max if column == 7 else min)(
                Decimal(row[1][column]) for row in rows
            )
            for _, values in rows:
                if Decimal(values[column]) == best:
                    values[column] = "**" + values[column] + "**"
        for index, values in rows:
            lines[index] = "| " + " | ".join(values) + " |"
    if len(groups) != 3:
        raise ValueError("Expected three complete bandwidth groups")
    result = "\n".join(lines) + "\n"
    if NOTE not in result:
        position = next(
            index for index, line in enumerate(lines) if line.startswith("## ")
        )
        lines[position:position] = [NOTE, ""]
        result = "\n".join(lines) + "\n"
    original_tables = [
        cells(line) for line in text.splitlines() if line.startswith("|")
    ]
    updated_tables = [
        cells(line) for line in result.splitlines() if line.startswith("|")
    ]
    if original_tables != updated_tables:
        raise ValueError("Formatting changed table content")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.report_dir
    report = directory / "results.md"
    raw = directory / "results.json"
    if not json.loads(raw.read_text())["complete"]:
        parser.error("Only complete results may be ranked")
    original = report.read_text()
    original_digest = sha256(report.read_bytes()).hexdigest()
    report.write_text(emphasize(original))
    path = directory / "verification.json"
    if path.exists():
        verification = json.loads(path.read_text())
        verification["report_sha256"] = sha256(report.read_bytes()).hexdigest()
        verification["report_formatting"] = dict(
            bold_best=True,
            displayed_precision=6,
            reference_excluded=True,
            diagnostic_decode_miss_unranked=True,
            table_content_unchanged=True,
            prior_report_sha256=original_digest,
            numeric_results_sha256=sha256(raw.read_bytes()).hexdigest(),
            formatter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        path.write_text(json.dumps(verification, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
