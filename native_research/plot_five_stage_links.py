"""Plot measured affected share and mean TTFT for the two traces."""


def draw(document, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, trace in enumerate(("mooncake", "servegen")):
        for method, color in zip(
            ("Serial", "PPD", "MigrationTax"), ("#2a78d6", "#eb6834", "#1baf7a")
        ):
            comparisons = sorted(
                (
                    row
                    for row in document["comparisons"]
                    if row["trace"] == trace and row["method"] == method
                ),
                key=lambda row: row["bandwidth_gb_per_s"],
            )
            rows = sorted(
                (
                    row
                    for row in document["runs"]
                    if row["phase"] > 0
                    and not row["reference"]
                    and row["trace"] == trace
                    and row["method"] == method
                ),
                key=lambda row: row["bandwidth_gb_per_s"],
            )
            observed = [
                row for row in comparisons if row["affected_d_percent"] is not None
            ]
            axes[0, column].plot(
                [row["bandwidth_gb_per_s"] for row in observed],
                [row["affected_d_percent"] for row in observed],
                marker="o",
                label=method,
                color=color,
            )
            axes[1, column].plot(
                [row["bandwidth_gb_per_s"] for row in rows],
                [row["mean_ttft_s"] for row in rows],
                marker="o",
                label=method,
                color=color,
            )
        axes[0, column].set_title(f"{trace}: affected D-first")
        axes[0, column].set_ylabel("Returning turns (%)")
        axes[0, column].set_ylim(0, 100)
        axes[1, column].set_title(f"{trace}: mean client TTFT")
        axes[1, column].set_ylabel("Seconds")
    for ax in axes.flat:
        ax.set_xlabel("KV endpoint bandwidth (GB/s per direction)")
        ax.set_xticks([50, 128, 450])
        ax.grid(alpha=0.25)
        ax.legend()
    figure.suptitle("TokenSim | llama-7b FP16 | H100 SXM 80GB | TP8 | seed 0")
    figure.savefig(destination / "five-stage-trends.png", dpi=180)
    figure.savefig(destination / "five-stage-trends.svg")
    plt.close(figure)
