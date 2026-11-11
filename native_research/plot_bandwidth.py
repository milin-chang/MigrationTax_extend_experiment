"""Plot completed bandwidth-sweep measurements with standard plotting tools."""


def draw(document, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("Serial", "PPD", "MigrationTax")
    colors = ("#2a78d6", "#eb6834", "#1baf7a")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for method, color in zip(methods, colors):
        compared = sorted(
            (row for row in document["comparisons"] if row["method"] == method),
            key=lambda row: row["bandwidth_gb_per_s"],
        )
        runs = sorted(
            (
                row
                for row in document["runs"]
                if row["method"] == method and not row["reference"]
            ),
            key=lambda row: row["bandwidth_gb_per_s"],
        )
        for ax, rows, key in (
            (axes[0, 0], compared, "affected_d_percent"),
            (axes[0, 1], compared, "affected_client_percent"),
            (axes[1, 0], runs, "mean_ttft_s"),
            (axes[1, 1], runs, "mean_tpot_s"),
        ):
            ax.plot(
                [r["bandwidth_gb_per_s"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                label=method,
                color=color,
            )
    for ax, title, unit in (
        (axes[0, 0], "Affected: first output sampled on D", "Returning turns (%)"),
        (axes[0, 1], "Affected: first client output", "Returning turns (%)"),
        (axes[1, 0], "Mean client TTFT", "Seconds"),
        (axes[1, 1], "Mean per-request TPOT", "Seconds / token"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Communication bandwidth (GB/s per direction)")
        ax.set_ylabel(unit)
        ax.set_xticks([50, 100, 200])
        ax.grid(alpha=0.25)
        ax.legend()
    first = document["runs"][0]
    figure.suptitle(
        f'TokenSim | ServeGen {first["requests"]:,} requests | seed 0 | '
        f'{first["model_id"]} / H100 / TP{first["tensor_parallel"]}'
    )
    figure.savefig(destination / "bandwidth-trends.png", dpi=180)
    figure.savefig(destination / "bandwidth-trends.svg")
    plt.close(figure)
