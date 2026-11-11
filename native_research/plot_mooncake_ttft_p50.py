"""Draw the verified per-request median client TTFT for the formal conditions."""

from .plot_mooncake_four_parts import COLORS, METHODS, style_axes, table


def draw(document, destination):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15.5, 7.3), facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[2.7, 1.05],
        left=0.065,
        right=0.985,
        bottom=0.15,
        top=0.83,
        hspace=0.45,
        wspace=0.2,
    )
    reference = next(row for row in document["runs"] if row["reference"])
    groups = []
    for link in ("nvl4", "pcie6", "ib400"):
        groups.append(
            [
                next(
                    row
                    for row in document["runs"]
                    if row["phase"] > 0
                    and row["link"] == link
                    and row["method"] == method
                )
                for method in METHODS
            ]
        )
    maximum = max(row["p50_ttft_s"] for group in groups for row in group) * 1.27
    for column, (group, bandwidth) in enumerate(zip(groups, (450, 128, 50))):
        ax = figure.add_subplot(grid[0, column])
        style_axes(ax)
        values = [row["p50_ttft_s"] for row in group]
        bars = ax.bar(range(3), values, color=COLORS, width=0.6)
        ax.set_xticks(range(3), METHODS, fontsize=10)
        ax.set_ylim(0, maximum)
        ax.set_title(f'{bandwidth} GB/s | data_{group[0]["link"]}', fontweight="bold")
        if column == 0:
            ax.set_ylabel("Client TTFT P50 (seconds)")
        ax.axhline(
            reference["p50_ttft_s"], color="#707b88", linestyle="--", linewidth=1.3
        )
        ax.bar_label(
            bars, labels=[f"{value:.6f}" for value in values], padding=5, fontsize=10
        )
    bottom = figure.add_subplot(grid[1, :])
    table(
        bottom,
        list(METHODS),
        ["450 GB/s: TTFT P50 (s)", "128 GB/s: TTFT P50 (s)", "50 GB/s: TTFT P50 (s)"],
        [[row["p50_ttft_s"] for row in group] for group in groups],
        ["min"] * 3,
        note="Green: lowest median among the three methods at the same KV bandwidth.",
    )
    figure.suptitle(
        "TokenSim | Mooncake | Client TTFT P50", fontsize=19, y=0.965, fontweight="bold"
    )
    figure.text(
        0.5,
        0.9,
        "llama-7b / FP16 | H100 SXM 80GB | TP8 | 4P+4D (64 simulated GPUs) | seed 0",
        ha="center",
        fontsize=11,
        color="#42556b",
    )
    figure.text(
        0.5,
        0.065,
        f'Dashed line: shared Serial inf reference P50 = {reference["p50_ttft_s"]:.6f} s.\nMedian of 12,031 completed requests per condition, including initial and returning turns.\nOriginal arrival to first client output; not a median across seeds. Native analytical compute.',
        ha="center",
        fontsize=10,
        va="center",
        color="#4c5c70",
    )
    files = []
    for suffix in ("png", "svg"):
        path = destination / f"ttft-p50-comparison.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        files.append(path.name)
    plt.close(figure)
    return files
