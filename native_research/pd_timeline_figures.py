"""Plot verified P/D clocks, native scheduling steps and matched request examples."""

from .pd_timeline_analysis import STEP_PARTS
from .plot_mooncake_four_parts import COLORS, METHODS, style_axes, table
from .plot_mooncake_link_study import LINKS, caption, digest, save_figure
from .plot_ttft_decomposition import PARTS

STEP_COLORS = ("#b17bb8", "#c0b071", "#477fad", "#91a5bb", "#df775b")
STEP_LABELS = (
    "Blocking load",
    "Before compute",
    "Compute service",
    "Between compute",
    "After compute",
)


def condition(rows, link, method):
    return next(r for r in rows if r["link"] == link and r["method"] == method)


def title(figure, tp, text):
    figure.suptitle(
        f"TokenSim | Mooncake | TP{tp} | {text}", y=0.98, fontsize=21, weight="bold"
    )
    caption(
        figure,
        f"llama-7b / FP16 | H100 SXM | 4P+4D ({8*tp} GPUs) | seed 0 | 12,031 requests",
        y=0.944,
        fontsize=12,
    )


def overview(data, target, tp):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure = plt.figure(figsize=(20, 18), facecolor="white")
    grid = figure.add_gridspec(
        4,
        3,
        left=0.065,
        right=0.985,
        top=0.86,
        bottom=0.13,
        height_ratios=(1.1, 1.1, 1.1, 1.15),
        hspace=0.65,
        wspace=0.28,
    )
    title(figure, tp, "Where is the first-token delay?")
    figure.legend(
        [Patch(color=c) for _, _, c in PARTS],
        [label for _, label, _ in PARTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.916),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    for j, (link, bandwidth) in enumerate(LINKS):
        rows = [condition(data["requests"], link, m) for m in METHODS]
        ax = figure.add_subplot(grid[0, j])
        bottom = [0.0] * 3
        for key, _, color in PARTS:
            vals = [r[f"mean_{key}_s"] for r in rows]
            ax.bar(range(3), vals, bottom=bottom, color=color, width=0.65)
            bottom = [a + b for a, b in zip(bottom, vals)]
        best = min(r["mean_ttft_s"] for r in rows)
        for i, row in enumerate(rows):
            ax.annotate(
                f'{row["mean_ttft_s"]:.3f}',
                (i, bottom[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                weight="bold" if row["mean_ttft_s"] == best else "normal",
                color="#175b32" if row["mean_ttft_s"] == best else "#303e4b",
            )
        ax.set_ylim(0, max(bottom) * 1.23)
        ax.set_ylabel("Mean client TTFT (s)")
        ax.set_title(f"{bandwidth} GB/s | Full client wait", weight="bold", fontsize=13)
        ax.set_xticks(range(3), METHODS, fontsize=10)
        style_axes(ax)
        ax = figure.add_subplot(grid[1, j])
        p = [r["gate_p_wait_mean"] for r in rows]
        d = [r["gate_d_after_p_wait_mean"] for r in rows]
        ax.bar(range(3), p, color="#df775b", width=0.65, label="P budget acquisition")
        ax.bar(
            range(3),
            d,
            bottom=p,
            color="#845c51",
            width=0.65,
            label="D after P / D-only",
        )
        ax.set_xticks(range(3), METHODS, fontsize=10)
        if not any(p + d):
            ax.set_ylim(0, 1)
            ax.set_yticks([0], ["0 s"])
            ax.text(
                0.5,
                0.5,
                "All 12,031 requests per method: 0 s",
                transform=ax.transAxes,
                ha="center",
                fontsize=11,
            )
        ax.set_ylabel("Mean gate wait (s)")
        ax.set_title("Admission only: P then D", weight="bold", fontsize=12)
        style_axes(ax)
        if j == 0:
            ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax = figure.add_subplot(grid[2, j])
        for k, (field, label, color) in enumerate(
            (
                ("p_prompt_span_mean", "P prompt window", "#7da176"),
                ("d_queue_to_compute_mean", "D ADD to compute", "#9c90ba"),
                ("d_queue_to_output_mean", "D ADD to first output", "#477fad"),
            )
        ):
            ax.bar(
                [i + (k - 1) * 0.23 for i in range(3)],
                [r[field] * 1000 for r in rows],
                width=0.22,
                label=label,
                color=color,
            )
        style_axes(ax)
        ax.set_xticks(range(3), METHODS, fontsize=10)
        ax.set_ylabel("Conditional mean duration (ms)")
        ax.set_title("Engine-side windows (overlap)", weight="bold", fontsize=12)
        if j == 0:
            ax.legend(fontsize=8.7, frameon=False, loc="upper right")
        tab = figure.add_subplot(grid[3, j])
        keys = (
            "mean_ttft_s",
            "p50_ttft_s",
            "p95_ttft_s",
            "mean_parent_wait_s",
            "gate_p_wait_mean",
            "gate_d_after_p_wait_mean",
            "mean_after_admission_s",
            "d_queue_to_compute_mean",
            "d_queue_to_output_mean",
            "d_first_decode_batch_size_mean",
        )
        table(
            tab,
            list(METHODS),
            [
                "Avg TTFT (s)",
                "TTFT P50 (s)",
                "TTFT P95 (s)",
                "Parent wait (s)",
                "P gate wait (s)",
                "D gate wait (s)",
                "After admission (s)",
                "D ADD to compute (s)",
                "D ADD to output (s)",
                "First D decode batch (requests)",
            ],
            [[r[key] for r in rows] for key in keys],
            ["min"] * 3 + [None] * 7,
            note="Bold green: lowest client TTFT; other rows diagnose timing.",
        )
    caption(
        figure,
        "TTFT = parent completion wait + KV admission wait + after-admission time. Each top stack includes all requests.\n"
        "P gate + D gate = admission; D-only fast paths contribute zero P gate. Acquisition order can hide D pressure behind P wait.\n"
        "Engine windows overlap and are conditional on observed timestamps; P spans can include layer-transfer barriers.\n"
        "D ADD-to-output includes scheduling, load and compute. A short internal D queue does not establish spare D capacity.\n"
        "HBM is fixed; KV links vary. Native analytical compute. Exact counts and field formulas: PD_TIMELINE.md / pd-requests.csv.",
        y=0.055,
        fontsize=11,
    )
    return save_figure(figure, target, "figure")


def worker_steps(data, target, tp):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(2, 3, figsize=(19, 11), facecolor="white")
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.83, bottom=0.19, hspace=0.48, wspace=0.28
    )
    title(figure, tp, "Native scheduling steps and batches")
    figure.legend(
        [Patch(color=c) for c in STEP_COLORS],
        STEP_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.908),
        ncol=5,
        frameon=False,
        fontsize=11,
    )
    for j, (link, bandwidth) in enumerate(LINKS):
        ax = axes[0, j]
        bars = [
            next(
                r
                for r in data["steps"]
                if (r["link"], r["method"], r["role"]) == (link, method, role)
            )
            for method in METHODS
            for role in ("prefill", "decode")
        ]
        bottom = [0.0] * 6
        for key, color in zip(STEP_PARTS, STEP_COLORS):
            vals = [r[f"{key}_mean"] * 1000 for r in bars]
            ax.bar(range(6), vals, bottom=bottom, color=color, width=0.7)
            bottom = [a + b for a, b in zip(bottom, vals)]
        ax.set_xticks(
            range(6),
            [f"{m}\n{role}" for m in ("Serial", "PPD", "Ours") for role in ("P", "D")],
            fontsize=10,
        )
        ax.set_title(f"{bandwidth} GB/s | Mean step span", fontsize=13, weight="bold")
        ax.set_ylabel("Elapsed time per scheduling step (ms)")
        style_axes(ax)
        ax = axes[1, j]
        for method, color in zip(METHODS, COLORS):
            rows = [
                r
                for r in data["batch_histogram"]
                if (r["link"], r["method"], r["role"]) == (link, method, "decode")
            ]
            total = sum(r["steps"] for r in rows)
            count, ys = 0, []
            for row in rows:
                count += row["steps"]
                ys.append(100 * count / total)
            ax.step(
                [r["batch_size"] for r in rows],
                ys,
                where="post",
                color=color,
                label=method,
                linewidth=2,
            )
        ax.set_ylim(0, 103)
        ax.set_xlim(0, 65)
        ax.set_xlabel("Scheduled D batch size (requests)")
        ax.set_ylabel("Cumulative fraction of D steps (%)")
        ax.set_title("All D scheduling steps: batch CDF", fontsize=12, weight="bold")
        style_axes(ax)
        if j == 0:
            ax.legend(frameon=False, fontsize=10)
    caption(
        figure,
        "Each scheduling step counts once on the TP-group leader; durations are not multiplied by batch size.\n"
        "Step span = blocking load + before compute + compute service + between compute + after compute.\n"
        "FF layer reads use explicit transfers: zero blocking load does not mean zero data loading. Step span excludes final Store writeback wait.\n"
        "D-step CDF includes native mixed/recompute steps; it is step-weighted, not the request-weighted first-decode batch distribution.\n"
        "Group means can change with batching. These are recorded simulated durations, not measured hardware utilization.",
        y=0.075,
        fontsize=11,
    )
    return save_figure(figure, target, "worker-steps")


def sample_timeline(data, target, tp):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(19, 10), facecolor="white")
    grid = figure.add_gridspec(
        2, 3, left=0.065, right=0.985, top=0.84, bottom=0.20, hspace=0.65, wspace=0.28
    )
    ax = figure.add_subplot(grid[0, :])
    title(figure, tp, f"Same returning request: ID {data['selection']['source_id']}")
    rows = [
        next(r for r in data["sample_requests"] if r["method"] == m) for m in METHODS
    ]
    for i, row in enumerate(rows):
        left = 0
        parts = [
            float(row["parent_wait_s"]),
            float(row["admission_wait_s"]),
            float(row["first_client_output_at_s"]) - float(row["submitted_s"]),
        ]
        for duration, (_, _, color) in zip(parts, PARTS):
            ax.barh(i, duration, left=left, color=color, height=0.55)
            left += duration
        ax.annotate(
            f'{float(row["ttft_s"]):.6f} s',
            (left, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(range(3), METHODS)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(r["ttft_s"]) for r in rows) * 1.28)
    ax.set_xlabel("Time since original arrival (s)")
    ax.set_title(
        "Same source ID at 450 GB/s | Full client wait", fontsize=13, weight="bold"
    )
    style_axes(ax)
    for j, row in enumerate(rows):
        ax = figure.add_subplot(grid[1, j])
        origin = float(row["submitted_s"])

        def bar(y, start, end, color):
            if row[start] and row[end]:
                a, b = float(row[start]), float(row[end])
                ax.barh(
                    y,
                    (b - a) * 1000,
                    left=(a - origin) * 1000,
                    height=0.55,
                    color=color,
                )

        bar(0, "p_queue_enter_s", "p_prefill_start_s", "#b6bdc8")
        bar(1, "p_prefill_start_s", "p_prefill_end_s", "#7da176")
        bar(2, "d_queue_enter_s", "d_first_compute_start_s", "#9c90ba")
        bar(3, "d_first_compute_start_s", "first_d_output_at_s", "#477fad")
        client = (float(row["first_client_output_at_s"]) - origin) * 1000
        ax.axvline(client, color="#c14435", linestyle="--", label="Client first token")
        if row["d_first_decode_step_end_s"]:
            decode = (float(row["d_first_decode_step_end_s"]) - origin) * 1000
            ax.plot(decode, 3, "D", color="#253f68", markersize=5)
        labels = []
        for role, short in (("prefill", "P"), ("decode", "D")):
            step = next(
                (
                    s
                    for s in data["sample_steps"]
                    if s["method"] == row["method"] and s["role"] == role
                ),
                None,
            )
            if step:
                labels.append(
                    f"{short} first batch={step['batch_size']}, load={float(step['load_wait_s'])*1000:.3f} ms"
                )
        ax.set_yticks(
            range(4),
            [
                "P ADD to prompt",
                "P prompt window",
                "D ADD to compute",
                "D compute to output",
            ],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Time since submission (ms)")
        ax.set_title(
            f"{row['method']} | {row['plan_branch']}", fontsize=12, weight="bold"
        )
        ax.text(
            0.02,
            -0.30,
            "\n".join(labels),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )
        style_axes(ax)
    handles = [
        Line2D([0], [0], color=c, linewidth=7, label=label) for _, label, c in PARTS
    ]
    handles += [
        Line2D([0], [0], color="#c14435", linestyle="--", label="Client first token"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#253f68",
            linestyle="None",
            label="First D decode-step end",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    caption(
        figure,
        "Example selection: median Serial TTFT rank among returning IDs with P/D observations in all three methods at 450 GB/s.\n"
        "The same ID is shown for all methods; it is not each method's median. Each detail panel has its own time scale.\n"
        "P prompt window includes internal gaps/barriers; D compute-to-output is a window, not pure kernel service.\n"
        "Exact request timestamps, first 12 P/D scheduling steps and up to 64 explicit MT transfers are exported as pd-sample-*.csv.",
        y=0.065,
        fontsize=11,
    )
    return save_figure(figure, target, "matched-request-timeline")


def across_tp(groups, target):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    figure.subplots_adjust(left=0.065, right=0.985, top=0.77, bottom=0.25, wspace=0.25)
    figure.suptitle(
        "Mooncake | Client TTFT and internal D wait across TP",
        fontsize=20,
        weight="bold",
        y=0.97,
    )
    caption(
        figure,
        "llama-7b / FP16 | H100 SXM | 4P+4D | seed 0 | 450 GB/s | 12,031 requests",
        y=0.89,
        fontsize=12,
    )
    for ax, (key, label) in zip(
        axes,
        (
            ("mean_ttft_s", "Mean client TTFT (s)"),
            ("mean_admission_wait_s", "Mean KV admission wait (s)"),
            ("d_queue_to_compute_mean", "Mean D ADD-to-compute (s)"),
        ),
    ):
        for method, color in zip(METHODS, COLORS):
            values = [
                condition(data["requests"], "nvl4", method)[key] for data in groups
            ]
            ax.plot(range(3), values, marker="o", color=color, label=method)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylabel(label)
        ax.set_xticks(range(3), ["TP1\n8 GPUs", "TP4\n32 GPUs", "TP8\n64 GPUs"])
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=11)
    caption(
        figure,
        "Y axes use symmetric log scaling (linear within +/-10 microseconds); zeros remain visible.\n"
        "TP increases total GPUs and aggregate KV capacity here: this is not a fixed-GPU TP speedup comparison.\n"
        "A short D engine queue can coexist with long adapter admission waits; pressure can remain outside the engine.",
        y=0.09,
        fontsize=11,
    )
    return save_figure(figure, target, "tp-comparison")
