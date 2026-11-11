"""Plot verified mean and maximum client TTFT without changing the P50 plot."""

import csv

from .plot_mooncake_four_parts import COLORS, METHODS, style_axes, table


def draw(document, destination):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15.5, 12.5), facecolor="white")
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=[2.6, 0.95, 2.6, 0.95],
        left=0.065,
        right=0.985,
        bottom=0.12,
        top=0.87,
        hspace=0.48,
        wspace=0.2,
    )
    reference = next(row for row in document["runs"] if row["reference"])
    groups = [
        [
            next(
                row
                for row in document["runs"]
                if row["phase"] > 0 and row["link"] == link and row["method"] == method
            )
            for method in METHODS
        ]
        for link in ("nvl4", "pcie6", "ib400")
    ]
    for band, (key, label) in enumerate(
        (("mean_ttft_s", "Avg TTFT"), ("max_ttft_s", "TTFT Max"))
    ):
        maximum = max(row[key] for group in groups for row in group) * 1.28
        for column, (group, bandwidth) in enumerate(zip(groups, (450, 128, 50))):
            ax = figure.add_subplot(grid[band * 2, column])
            style_axes(ax)
            values = [row[key] for row in group]
            bars = ax.bar(range(3), values, color=COLORS, width=0.6)
            ax.set_xticks(range(3), METHODS, fontsize=10)
            ax.set_ylim(0, maximum)
            ax.set_title(f"{label} | {bandwidth} GB/s", fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"Client {label} (seconds)")
            ax.axhline(reference[key], color="#707b88", linestyle="--", linewidth=1.3)
            ax.bar_label(
                bars,
                labels=[f"{value:.6f}" for value in values],
                padding=5,
                fontsize=10,
            )
        bottom = figure.add_subplot(grid[band * 2 + 1, :])
        table(
            bottom,
            list(METHODS),
            [f"{bandwidth} GB/s: {label} (s)" for bandwidth in (450, 128, 50)],
            [[row[key] for row in group] for group in groups],
            ["min"] * 3,
            note="Green: lowest value among methods at the same bandwidth; reference excluded.",
        )
    figure.suptitle(
        "TokenSim | Mooncake | Avg TTFT and TTFT Max",
        fontsize=19,
        y=0.97,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.93,
        "llama-7b / FP16 | H100 SXM 80GB | TP8 | 4P+4D (64 simulated GPUs) | seed 0",
        ha="center",
        fontsize=11,
        color="#42556b",
    )
    figure.text(
        0.5,
        0.05,
        f'Dashed reference (Serial inf): Avg = {reference["mean_ttft_s"]:.6f} s; Max = {reference["max_ttft_s"]:.6f} s.\n12,031 completed requests per condition, including initial and returning turns.\nClient TTFT starts at original arrival. Max is the slowest request, not a percentile. Native analytical compute.',
        ha="center",
        fontsize=10,
        va="center",
        color="#4c5c70",
    )
    files = []
    for suffix in ("png", "svg"):
        path = destination / f"ttft-avg-max-comparison.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        files.append(path.name)
    plt.close(figure)
    with (destination / "ttft-summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        keys = [
            "phase",
            "link",
            "method",
            "requests",
            "mean_ttft_s",
            "p50_ttft_s",
            "p95_ttft_s",
            "max_ttft_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in document["runs"])
    return files
