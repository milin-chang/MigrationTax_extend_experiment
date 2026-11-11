"""Export four Mooncake panels with tables, using the completed delivery only."""

import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "native_research/reports/five-stage-llama7b-h100-tp8-s0-01/delivery-mooncake-complete"
)
OUTPUT = ROOT / "native_research/reports/mooncake-four-part-s0-01"
METHODS = ("Serial", "PPD", "MigrationTax")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(document):
    if len(document["runs"]) != 12 or len(document["comparisons"]) != 9:
        raise ValueError("Expected exactly 12 Mooncake conditions and nine comparisons")
    if any(row["trace"] != "mooncake" for row in document["runs"]):
        raise ValueError("The completed delivery contains another trace")
    reference = next(row for row in document["runs"] if row["reference"])
    baseline = {
        row["source_id"]: row
        for row in read_csv(SOURCE / (reference["id"] + "-requests.csv"))
    }
    checks, hashes, sanity, p50_checks = [], {}, {}, []
    comparisons = {(row["link"], row["method"]): row for row in document["comparisons"]}
    for record in document["runs"]:
        path = SOURCE / (record["id"] + "-requests.csv")
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        rows = read_csv(path)
        receipt = ROOT / (record["output"] + "-job") / "audit.json"
        if json.loads(receipt.read_text())["status"] != "passed":
            raise ValueError("An included condition lacks a passed audit")
        if len(rows) != record["requests"] or len(
            {row["source_id"] for row in rows}
        ) != len(rows):
            raise ValueError("Population or source IDs differ")
        client = [float(row["native_prefill_latency_s"]) for row in rows]
        mean, median, maximum = (
            statistics.mean(client),
            statistics.median(client),
            max(client),
        )
        if not math.isclose(maximum, record["max_ttft_s"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Reported client TTFT Max differs from request maximum")
        if not math.isclose(median, record["p50_ttft_s"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Reported client TTFT P50 differs from request median")
        p50_checks.append(
            dict(
                condition=record["id"],
                phase=record["phase"],
                link=record["link"],
                method=record["method"],
                requests=len(rows),
                client_ttft_p50_s=median,
                native_client_ttft_p50_s=record["p50_ttft_s"],
                client_ttft_avg_s=mean,
                client_ttft_max_s=maximum,
            )
        )
        record["p50_ttft_s"] = median
        record["max_ttft_s"] = maximum
        p95 = sorted(float(row["native_prefill_latency_s"]) for row in rows)[
            math.ceil(0.95 * len(rows)) - 1
        ]
        if not math.isclose(
            mean, record["mean_ttft_s"], abs_tol=1e-10
        ) or not math.isclose(p95, record["p95_ttft_s"], abs_tol=1e-10):
            raise ValueError("Reported TTFT differs from request events")
        d_first = [float(row["d_output_sampled_latency_s"]) for row in rows]
        record["p50_d_first_s"] = statistics.median(d_first)
        record["p95_d_first_s"] = sorted(d_first)[math.ceil(0.95 * len(d_first)) - 1]
        if not math.isclose(
            statistics.mean(d_first), record["mean_d_first_observed_s"], abs_tol=1e-10
        ):
            raise ValueError("Reported D-first differs from samples")
        if record["phase"] == 0:
            if len(rows) != 200:
                raise ValueError("Sanity population differs")
            sanity[record["link"]] = {row["source_id"]: row for row in rows}
        else:
            if (
                len(rows) != 12031
                or sum(int(row["source_turn"]) > 0 for row in rows) != 3959
            ):
                raise ValueError("Formal request or returning population differs")
            if sum(int(row["generated"]) for row in rows) != 4122048:
                raise ValueError("Formal output token count differs")
            if not record["reference"]:
                affected = misses = 0
                for row in rows:
                    ref = baseline[row["source_id"]]
                    if any(
                        row[key] != ref[key]
                        for key in ("arrival_s", "source_turn", "generated")
                    ):
                        raise ValueError("Reference pairing differs")
                    if int(row["source_turn"]) > 0:
                        affected += float(
                            row["d_output_sampled_latency_s"]
                        ) > 2 * float(ref["d_output_sampled_latency_s"])
                        misses += row["decode_miss_at_original_arrival"] == "True"
                comparison = comparisons[record["link"], record["method"]]
                if not math.isclose(
                    100 * affected / 3959,
                    comparison["affected_d_percent"],
                    abs_tol=1e-9,
                ):
                    raise ValueError("Affected percentage differs from raw pairs")
                if (
                    affected != comparison["affected_d_count"]
                    or comparison["missing_d_samples"]
                ):
                    raise ValueError("Affected counts or D samples differ")
                if not math.isclose(
                    100 * misses / 3959, comparison["decode_miss_percent"], abs_tol=1e-9
                ):
                    raise ValueError("Decode miss count differs")
                checks.append(
                    dict(
                        link=record["link"],
                        method=record["method"],
                        affected=affected,
                        returning=3959,
                    )
                )
    if sanity["nvl4"].keys() != sanity["ib400"].keys():
        raise ValueError("Sanity source IDs differ")
    later = sum(
        float(sanity["ib400"][key]["d_output_sampled_latency_s"])
        > float(row["d_output_sampled_latency_s"])
        for key, row in sanity["nvl4"].items()
    )
    if later != 198:
        raise ValueError("Sanity paired count differs")
    return dict(
        status="passed",
        audited_conditions=12,
        independent_comparison_checks=checks,
        sanity_slow_later=later,
        source_csv_sha256=hashes,
        source_results_sha256=sha256(
            (SOURCE / "results.json").read_bytes()
        ).hexdigest(),
        ttft_p50_checks=p50_checks,
    )


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", length=0)


def table(
    ax,
    headers,
    labels,
    values,
    directions,
    reference_column=False,
    note="Green: best among finite-link methods. Cache miss is diagnostic.",
):
    ax.axis("off")
    text = [
        [label]
        + [str(value) if isinstance(value, str) else f"{value:.6f}" for value in row]
        for label, row in zip(labels, values)
    ]
    count = len(headers)
    widths = [0.39] + [0.61 / count] * count
    result = ax.table(
        cellText=text,
        colLabels=["Metric / unit"] + headers,
        cellLoc="center",
        colWidths=widths,
        bbox=[0, 0.07, 1, 0.9],
    )
    result.auto_set_font_size(False)
    result.set_fontsize(10)
    for (row, column), cell in result.get_celld().items():
        cell.set_edgecolor("#d8dfe7")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eaf0f7")
            cell.set_text_props(weight="bold", color="#213951")
        elif column == 0:
            cell.set_text_props(ha="left")
            cell.set_facecolor("#f4f6f8")
        elif reference_column and column == 1:
            cell.set_facecolor("#f0f0f0")
    for index, (row, direction) in enumerate(zip(values, directions), start=1):
        if direction is None:
            continue
        start = 1 if reference_column else 0
        eligible = [value for value in row[start:] if isinstance(value, (float, int))]
        best = min(eligible) if direction == "min" else max(eligible)
        for column, value in enumerate(row, start=1):
            if reference_column and column == 1:
                continue
            if isinstance(value, (float, int)) and f"{value:.6f}" == f"{best:.6f}":
                result[index, column].set_facecolor("#dff2e5")
                result[index, column].set_text_props(weight="bold", color="#175b32")
    ax.text(
        0,
        0,
        note,
        transform=ax.transAxes,
        fontsize=8.2,
        color="#536276",
    )


def draw_panel(figure, slot, number, document):
    inner = slot.subgridspec(2, 1, height_ratios=[1.8, 1.65], hspace=0.25)
    ax, tab = figure.add_subplot(inner[0]), figure.add_subplot(inner[1])
    style_axes(ax)
    if number == 0:
        rows = [
            next(
                row
                for row in document["runs"]
                if row["phase"] == 0 and row["link"] == link
            )
            for link in ("nvl4", "ib400")
        ]
        values = [row["mean_d_first_observed_s"] for row in rows]
        bars = ax.bar([0, 1], values, color=COLORS[:2], width=0.55)
        ax.set_xticks([0, 1], ["data_nvl4\n450 GB/s", "data_ib400\n50 GB/s"])
        ax.set_ylabel("Mean D-first (seconds)")
        ax.set_ylim(0, max(values) * 1.45)
        ax.set_title("0 | Sanity: 200 requests, Serial", loc="left", fontweight="bold")
        ax.bar_label(
            bars, labels=[f"{value:.6f} s" for value in values], padding=5, fontsize=11
        )
        ax.text(
            0.5,
            0.96,
            f"50 / 450 = {values[1] / values[0]:.2f}x\n198 / 200 requests later at 50 GB/s",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        labels = [
            "Mean D-first (s)",
            "Avg client TTFT (s)",
            "Client TTFT P50 (s)",
            "Client TTFT Max (s)",
            "D-first P50 (s)",
            "D-first P95 (s)",
            "Mean lifecycle (s)",
            "Mean TPOT (s/token)",
        ]
        keys = [
            "mean_d_first_observed_s",
            "mean_ttft_s",
            "p50_ttft_s",
            "max_ttft_s",
            "p50_d_first_s",
            "p95_d_first_s",
            "mean_lifecycle_s",
            "mean_tpot_s",
        ]
        table(
            tab,
            ["450 GB/s", "50 GB/s"],
            labels,
            [[row[key] for row in rows] for key in keys],
            ["min"] * len(keys),
            note="Green: lower latency in the same-request sanity pair.",
        )
        return
    phase, link, bandwidth = {
        1: (1, "nvl4", 450),
        3: (3, "pcie6", 128),
        4: (4, "ib400", 50),
    }[number]
    rows = [
        next(
            row
            for row in document["runs"]
            if row["phase"] == phase and row["link"] == link and row["method"] == method
        )
        for method in METHODS
    ]
    comparisons = [
        next(
            row
            for row in document["comparisons"]
            if row["link"] == link and row["method"] == method
        )
        for method in METHODS
    ]
    values = [row["affected_d_percent"] for row in comparisons]
    bars = ax.bar(range(3), values, color=COLORS, width=0.6)
    ax.set_xticks(range(3), METHODS)
    ax.set_ylabel("Affected returning turns (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"{number} | Mooncake: data_{link}, {bandwidth} GB/s",
        loc="left",
        fontweight="bold",
    )
    ax.bar_label(
        bars,
        labels=[
            f'{row["affected_d_percent"]:.2f}%\n{row["affected_d_count"]:,} / 3,959'
            for row in comparisons
        ],
        padding=5,
        fontsize=10,
    )
    labels = [
        "Avg client TTFT (s)",
        "Client TTFT P50 (s)",
        "Client TTFT P95 (s)",
        "Client TTFT Max (s)",
        "Mean lifecycle (s)",
        "Mean TPOT (s/token)",
        "Affected D-first (%)",
        "Decode miss (%)",
    ]
    keys = [
        "mean_ttft_s",
        "p50_ttft_s",
        "p95_ttft_s",
        "max_ttft_s",
        "mean_lifecycle_s",
        "mean_tpot_s",
    ]
    values = [[row[key] for row in rows] for key in keys]
    values += [
        [row["affected_d_percent"] for row in comparisons],
        [row["decode_miss_percent"] for row in comparisons],
    ]
    headers = list(METHODS)
    if number == 1:
        reference = next(row for row in document["runs"] if row["reference"])
        headers.insert(0, "Serial inf")
        for index, key in enumerate(keys):
            values[index].insert(0, reference[key])
        values[-2].insert(0, "--")
        values[-1].insert(0, "--")
    table(
        tab, headers, labels, values, ["min"] * 7 + [None], reference_column=number == 1
    )


def footer(figure, *, combined=False):
    text = "Formal: 12,031 requests; 3,959 returning turns. Affected = D-first > 2x same-request Serial inf.\nTP8 / FP16 / 4P+4D (64 simulated GPUs). Native analytical compute; seed 0, no seed aggregation."
    figure.text(
        0.5,
        0.012 if combined else 0.018,
        text,
        ha="center",
        va="bottom",
        fontsize=10 if combined else 8.3,
        color="#4c5c70",
    )


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_fonts import times_new_roman

    font_faces = times_new_roman()

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "svg.fonttype": "path",
            "font.size": 11,
            "axes.titlepad": 13,
            "svg.hashsalt": "mooncake-four-parts-s0",
        }
    )
    document = json.loads((SOURCE / "results.json").read_text())
    verification = verify(document)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = []
    figure = plt.figure(figsize=(17, 13.5), facecolor="white")
    outer = figure.add_gridspec(
        2, 2, left=0.055, right=0.985, bottom=0.09, top=0.91, hspace=0.31, wspace=0.2
    )
    for slot, number in zip(outer, (0, 1, 3, 4)):
        draw_panel(figure, slot, number, document)
    figure.suptitle(
        "TokenSim | Mooncake shifted v1 | llama-7b | H100 SXM 80GB",
        fontsize=20,
        y=0.974,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.936,
        "Sanity + 450 / 128 / 50 GB/s  |  Reference: Serial data_inf (1e15 bytes/s)",
        ha="center",
        fontsize=12,
        color="#42556b",
    )
    footer(figure, combined=True)
    for suffix in ("png", "svg"):
        path = OUTPUT / ("mooncake-four-parts." + suffix)
        figure.savefig(path, dpi=200, facecolor="white")
        figures.append(path.name)
    plt.close(figure)
    for number, name in (
        (0, "00-sanity"),
        (1, "01-nvl4-450"),
        (3, "03-pcie6-128"),
        (4, "04-ib400-50"),
    ):
        figure = plt.figure(figsize=(9.4, 7.6), facecolor="white")
        slot = figure.add_gridspec(1, 1, left=0.11, right=0.975, bottom=0.15, top=0.86)[
            0
        ]
        draw_panel(figure, slot, number, document)
        figure.suptitle(
            "TokenSim | Mooncake | llama-7b | H100 SXM 80GB",
            fontsize=15,
            y=0.97,
            fontweight="bold",
        )
        footer(figure)
        for suffix in ("png", "svg"):
            path = OUTPUT / f"{name}.{suffix}"
            figure.savefig(path, dpi=220, facecolor="white")
            figures.append(path.name)
        plt.close(figure)
    from .plot_mooncake_ttft_p50 import draw

    figures.extend(draw(document, OUTPUT))
    from .plot_mooncake_ttft_summary import draw as draw_summary

    figures.extend(draw_summary(document, OUTPUT))
    with (OUTPUT / "ttft-p50.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(verification["ttft_p50_checks"][0])
        )
        writer.writeheader()
        writer.writerows(verification["ttft_p50_checks"])
    verification.update(
        figures=figures,
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        font_family="Times New Roman",
        selected_font_faces=font_faces,
    )
    (OUTPUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    with ZipFile(
        OUTPUT / "mooncake-figures.zip", "w", compression=ZIP_DEFLATED
    ) as bundle:
        for filename in figures:
            bundle.write(OUTPUT / filename, arcname=filename)
    print(
        json.dumps(
            dict(output=str(OUTPUT), figures=figures, checks=verification["status"])
        )
    )


if __name__ == "__main__":
    main()
