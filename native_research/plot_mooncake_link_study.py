"""Verify completed Mooncake events and draw TP1 link and deployment comparisons."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile, ZIP_DEFLATED

from .aligned_audit import ROOT, read
from .plot_mooncake_four_parts import COLORS, METHODS, draw_panel, style_axes, table
from .plot_fonts import times_new_roman
from .report_bandwidth import write_csv

LINKS = (("nvl4", 450), ("pcie6", 128), ("ib400", 50))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(source, tp):
    document = read(source / "results.json")
    document["runs"] = [row for row in document["runs"] if row["phase"] > 0 and row.get("trace", "mooncake") == "mooncake"]
    assert len(document["runs"]) == 10 and len(document["comparisons"]) == 9
    reference = next(row for row in document["runs"] if row["reference"])
    baseline_path = ROOT / reference["output"] / "requests.jsonl"
    baseline = {r["source_id"]: r for r in map(json.loads, baseline_path.read_text().splitlines())}
    comparisons = {(r["link"], r["method"]): r for r in document["comparisons"]}
    checks = []
    for record in document["runs"]:
        output = ROOT / record["output"]
        receipt = output.with_name(output.name + "-job")
        audit = read(receipt / "audit.json")
        job = read(receipt / "job.json")
        condition = read(output / "condition.json")
        config = read(output / "effective-config.json")
        assert audit["status"] == job["status"] == "passed" and job["returncode"] == 0
        assert condition["status"] == "completed"
        assert config["hardware_alignment"]["tensor_parallel"] == tp
        assert config["hardware_alignment"]["total_gpus"] == tp * 8
        rows = {r["source_id"]: r for r in map(json.loads, (output / "requests.jsonl").read_text().splitlines())}
        assert len(rows) == record["requests"] == audit["requests"] == 12031
        assert rows.keys() == baseline.keys()
        assert sum(r["source_turn"] > 0 for r in rows.values()) == 3959
        assert sum(r["generated"] for r in rows.values()) == audit["output_tokens"] == 4122048
        ttft = [r["native_prefill_latency_s"] for r in rows.values()]
        assert all(math.isfinite(value) and value >= 0 for value in ttft)
        actual = dict(mean_ttft_s=statistics.mean(ttft), p50_ttft_s=statistics.median(ttft),
                      p95_ttft_s=sorted(ttft)[math.ceil(.95 * len(ttft)) - 1], max_ttft_s=max(ttft),
                      mean_lifecycle_s=statistics.mean(r["native_total_time_s"] for r in rows.values()),
                      mean_tpot_s=statistics.mean(r["native_decode_time_sum_s"] / max(1, r["generated"] - 1) for r in rows.values()))
        for key, value in actual.items():
            assert math.isclose(record[key], value, rel_tol=0, abs_tol=1e-9), (record["id"], key)
        affected = misses = 0
        for key, row in rows.items():
            ref = baseline[key]
            assert all(row[field] == ref[field] for field in ("arrival_s", "source_turn", "generated"))
            assert row["d_output_sampled_latency_s"] is not None
            if row["source_turn"] > 0:
                affected += row["d_output_sampled_latency_s"] > 2 * ref["d_output_sampled_latency_s"]
                misses += bool(row["decode_miss_at_original_arrival"])
        if not record["reference"]:
            comparison = comparisons[record["link"], record["method"]]
            assert comparison["returning_requests"] == 3959 and comparison["missing_d_samples"] == 0
            assert comparison["affected_d_count"] == affected
            assert math.isclose(comparison["affected_d_percent"], 100 * affected / 3959, abs_tol=1e-10)
            assert math.isclose(comparison["decode_miss_percent"], 100 * misses / 3959, abs_tol=1e-10)
        checks.append(dict(id=record["id"], requests=12031, returning=3959,
            input_sha256=condition["input_sha256"], requests_sha256=digest(output / "requests.jsonl"),
            audit_sha256=digest(receipt / "audit.json"), recalculated=actual, status="passed"))
    return document, checks


def save_figure(figure, destination, name):
    import matplotlib.pyplot as plt
    result = []
    for suffix in ("png", "svg"):
        path = destination / f"{name}.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        result.append(path.name)
    plt.close(figure)
    return result


def caption(figure, text, y=.04, fontsize=10):
    figure.text(.5, y, text, ha="center", va="center", fontsize=fontsize, color="#4c5c70")


def panels(document, destination):
    import matplotlib.pyplot as plt
    result = []
    for number, name in ((1, "01-nvl4-450"), (3, "03-pcie6-128"), (4, "04-ib400-50")):
        figure = plt.figure(figsize=(11.5, 9), facecolor="white")
        slot = figure.add_gridspec(1, 1, left=.1, right=.98, bottom=.17, top=.85)[0]
        draw_panel(figure, slot, number, document)
        figure.suptitle("TokenSim | Mooncake shifted v1 | TP1", fontsize=18, y=.97, weight="bold")
        caption(figure, "llama-7b / FP16 | H100 SXM 80GB | 4P+4D (8 simulated GPUs) | seed 0", y=.923)
        caption(figure, "12,031 requests; 3,959 returning turns. Affected = D-first > 2x same-request Serial inf.\nTTFT begins at original arrival; lifecycle includes writeback. Native analytical compute.\nBold green cells: best finite-link method; decode miss is diagnostic.", y=.058, fontsize=9)
        result += save_figure(figure, destination, name)
    figure = plt.figure(figsize=(25, 10), facecolor="white")
    grid = figure.add_gridspec(1, 3, left=.045, right=.985, bottom=.15, top=.84, wspace=.21)
    for slot, phase in zip(grid, (1, 3, 4)):
        draw_panel(figure, slot, phase, document)
    figure.suptitle("TokenSim | Mooncake shifted v1 | TP1 | Stages 1, 3 and 4", y=.97, fontsize=22, weight="bold")
    caption(figure, "llama-7b / FP16 | H100 SXM 80GB | 4P+4D (8 simulated GPUs) | seed 0", y=.918, fontsize=13)
    caption(figure, "12,031 requests; 3,959 returning turns. One shared Serial inf reference.\nAffected = D-first > 2x same-request reference; TTFT starts at original arrival. Native analytical compute.", y=.05, fontsize=11)
    result += save_figure(figure, destination, "mooncake-tp1-three-stages")
    return result


def affected(document, destination):
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
    figure.subplots_adjust(left=.1, right=.98, top=.82, bottom=.2)
    style_axes(ax)
    for m, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = [next(r["affected_d_percent"] for r in document["comparisons"] if r["link"] == link and r["method"] == method) for link, _ in LINKS]
        bars = ax.bar([x + (m - 1) * .25 for x in range(3)], values, width=.24, label=method, color=color)
        labels = ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=5, fontsize=11)
        for x, (value, label) in enumerate(zip(values, labels)):
            best = min(r["affected_d_percent"] for r in document["comparisons"] if r["link"] == LINKS[x][0])
            if round(value, 6) == round(best, 6):
                label.set_weight("bold")
    ax.set_xticks(range(3), [str(bandwidth) for _, bandwidth in LINKS])
    ax.set_xlabel("KV communication bandwidth (GB/s)")
    ax.set_ylabel("Affected returning turns (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    figure.suptitle("TokenSim | Mooncake | TP1 affected share", y=.975, fontsize=18, weight="bold")
    caption(figure, "llama-7b / FP16 | H100 SXM 80GB | 8 GPUs | seed 0", y=.9, fontsize=11)
    caption(figure, "D-first > 2x same-request Serial inf. Denominator: all 3,959 returning turns.\nBold: lowest share at the same bandwidth. Native analytical compute.", y=.063)
    return save_figure(figure, destination, "affected-share")


def ttft(document, destination, metrics, filename):
    import matplotlib.pyplot as plt
    n = len(metrics)
    figure = plt.figure(figsize=(16, 7.8 if n == 1 else 13.5), facecolor="white")
    grid = figure.add_gridspec(n * 2, 3, height_ratios=[2.7, 1.05] * n,
        left=.065, right=.985, bottom=.16 if n == 1 else .12, top=.82 if n == 1 else .87,
        hspace=.5, wspace=.22)
    reference = next(row for row in document["runs"] if row["reference"])
    groups = [[next(r for r in document["runs"] if r["link"] == link and r["method"] == method)
               for method in METHODS] for link, _ in LINKS]
    for band, (key, label) in enumerate(metrics):
        upper = max(r[key] for group in groups for r in group) * 1.27
        for column, (group, (_, bandwidth)) in enumerate(zip(groups, LINKS)):
            ax = figure.add_subplot(grid[band * 2, column]); style_axes(ax)
            values = [r[key] for r in group]
            bars = ax.bar(range(3), values, width=.6, color=COLORS)
            ax.set_xticks(range(3), METHODS, fontsize=11)
            ax.set_ylim(0, upper)
            ax.set_title(f"{label} | {bandwidth} GB/s", weight="bold")
            if column == 0:
                ax.set_ylabel(f"Client {label} (seconds)")
            ax.axhline(reference[key], color="#707b88", linestyle="--", linewidth=1.3)
            labels = ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=11)
            for value, annotation in zip(values, labels):
                if round(value, 6) == min(round(v, 6) for v in values): annotation.set_weight("bold")
        bottom = figure.add_subplot(grid[band * 2 + 1, :])
        table(bottom, list(METHODS), [f"{bandwidth} GB/s: {label} (s)" for _, bandwidth in LINKS],
              [[row[key] for row in group] for group in groups], ["min"] * 3,
              note="Bold green: lowest value at the same bandwidth; reference excluded. Tables retain six decimals.")
    figure.suptitle("TokenSim | Mooncake | TP1 | " + " and ".join(label for _, label in metrics), y=.975, fontsize=19, weight="bold")
    caption(figure, "llama-7b / FP16 | H100 SXM 80GB | 4P+4D (8 simulated GPUs) | seed 0", y=.91 if n == 1 else .93, fontsize=11)
    references = "; ".join(f"{label} = {reference[key]:.6f} s" for key, label in metrics)
    caption(figure, f"Dashed line: shared Serial inf ({references}).\n12,031 completed requests per condition; includes initial and returning turns.\nOriginal arrival to first client output. Native analytical compute.", y=.065 if n == 1 else .048, fontsize=10)
    return save_figure(figure, destination, filename)


def deployment(first, second, destination, affected_metric):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor="white")
    figure.subplots_adjust(left=.065, right=.985, top=.79, bottom=.23, wspace=.24)
    legend_handles = []
    for ax, (link, bandwidth) in zip(axes, LINKS):
        style_axes(ax)
        for offset, (document, label, color, hatch) in enumerate([
            (first, "TP1: 8 GPUs", "#345f99", None),
            (second, "TP8: 64 GPUs", "#b4c2d5", "///"),
        ]):
            records = document["comparisons"] if affected_metric else document["runs"]
            key = "affected_d_percent" if affected_metric else "mean_ttft_s"
            values = [next(r[key] for r in records if r["link"] == link and r["method"] == method) for method in METHODS]
            bars = ax.bar([x + (offset - .5) * .36 for x in range(3)], values, .34, label=label,
                          color=color, hatch=hatch, edgecolor="#2a405b", linewidth=.5)
            ax.bar_label(bars, labels=[f"{v:.2f}" if affected_metric else f"{v:.3f}" for v in values], padding=4, fontsize=10)
            if ax is axes[0]: legend_handles.append(bars)
        ax.set_title(f"{bandwidth} GB/s | data_{link}", weight="bold")
        ax.set_xticks(range(3), METHODS, fontsize=11)
        if affected_metric:
            ax.set_ylim(0, 100)
        else:
            ax.set_yscale("log"); ax.set_ylim(.04, 2000)
        if ax is axes[0]: ax.set_ylabel("Affected returning turns (%)" if affected_metric else "Avg client TTFT (seconds; log scale)")
    metric = "Affected share" if affected_metric else "Avg client TTFT"
    figure.suptitle(f"TokenSim | Mooncake | TP1 vs TP8 | {metric}", y=.97, fontsize=19, weight="bold")
    figure.legend(legend_handles, ["TP1: 8 GPUs", "TP8: 64 GPUs"], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .91), fontsize=12)
    scope = "Affected share uses each deployment's own Serial inf reference." if affected_metric else "Same seconds unit; logarithmic y-axis spans the two latency ranges."
    caption(figure, "Same shifted-v1 seed-0 input: 12,031 requests / 3,959 returning turns.\n4P+4D in both deployments; GPU totals differ (8 vs 64). This is a deployment comparison.\n" + scope + " Native analytical compute.", y=.085, fontsize=10)
    return save_figure(figure, destination, "tp1-vs-tp8-affected-share" if affected_metric else "tp1-vs-tp8-avg-ttft")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "native_research/reports/five-stage-llama7b-h100-tp1-s0-01")
    parser.add_argument("--tp8-source", type=Path, default=ROOT / "native_research/reports/five-stage-llama7b-h100-tp8-s0-01/delivery-mooncake-complete")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.source / "figures"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    faces = times_new_roman()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "svg.fonttype": "path", "svg.hashsalt": "mooncake-tp1-link-study", "axes.titlepad": 13})
    first, first_checks = verify(args.source, 1)
    second, second_checks = verify(args.tp8_source, 8)
    assert {r["input_sha256"] for r in first["runs"]} == {r["input_sha256"] for r in second["runs"]}
    destination.mkdir(parents=True, exist_ok=True)
    figures = panels(first, destination) + affected(first, destination)
    figures += ttft(first, destination, [("p50_ttft_s", "TTFT P50")], "ttft-p50-comparison")
    figures += ttft(first, destination, [("p95_ttft_s", "TTFT P95")], "ttft-p95-comparison")
    figures += ttft(first, destination, [("mean_ttft_s", "Avg TTFT"), ("max_ttft_s", "TTFT Max")], "ttft-avg-max-comparison")
    figures += deployment(first, second, destination, True) + deployment(first, second, destination, False)
    keys = ["phase", "link", "method", "requests", "mean_ttft_s", "p50_ttft_s", "p95_ttft_s", "max_ttft_s", "mean_lifecycle_s", "mean_tpot_s", "output_tokens_per_s"]
    write_csv(destination / "ttft-summary.csv", [{key: r[key] for key in keys} for r in first["runs"]])
    write_csv(destination / "affected-share.csv", first["comparisons"])
    comparison = []
    for tp, document in ((1, first), (8, second)):
        for row in document["runs"]:
            comparison.append(dict(tp=tp, gpu_count=8 * tp, **{key: row[key] for key in keys}))
    write_csv(destination / "tp1-vs-tp8.csv", comparison)
    verification = dict(status="passed", simulations_launched=False,
        source_sha256=digest(args.source / "results.json"), tp8_source_sha256=digest(args.tp8_source / "results.json"),
        same_input_across_deployments=True, tp1_checks=first_checks, tp8_checks=second_checks,
        font_family="Times New Roman", selected_font_faces=faces, generator_sha256=digest(Path(__file__)),
        figures=[dict(file=name, sha256=digest(destination / name)) for name in figures],
        note="TP1 uses 8 GPUs; TP8 uses 64 GPUs. Each has its own Serial inf affected-share baseline.")
    (destination / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    lines = ["# Mooncake TP1 圖表", "", "Times New Roman；seed 0；llama-7b／FP16；H100 SXM；TP1／PP1；4P＋4D，共 8 GPU。", "",
             "本次只跑 Mooncake 階段 1、3、4，共 10 組，沒有 sanity。圖中秒數保留原始紀錄的單位，並未做毫秒轉換。", "",
             "TP1／TP8 對照的 GPU 總數不同（8／64）；affected share 各自使用該部署的 Serial inf 參考。這是部署比較。", ""]
    for name in figures:
        if name.endswith(".png"): lines.append(f"- [{name}]({name})")
    lines += ["", "每張圖另有同名 SVG。數字表見 ttft-summary.csv、affected-share.csv、tp1-vs-tp8.csv；原始事件重算與字體證據見 verification.json。", ""]
    (destination / "FIGURES.md").write_text("\n".join(lines))
    with ZipFile(destination / "mooncake-tp1-figures.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in figures + ["ttft-summary.csv", "affected-share.csv", "tp1-vs-tp8.csv", "verification.json", "FIGURES.md"]:
            bundle.write(destination / name, arcname=name)
    print(json.dumps(dict(output=str(destination), png_figures=len(figures) // 2,
                         formats=["png", "svg"], verification="passed")))


if __name__ == "__main__":
    main()
