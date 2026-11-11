"""Reconstruct client TTFT components and plot completed Mooncake studies."""

import argparse
import json
import math
from pathlib import Path
import statistics
from zipfile import ZIP_DEFLATED, ZipFile

from .aligned_audit import ROOT, read
from .plot_fonts import times_new_roman
from .plot_mooncake_four_parts import METHODS, style_axes, table
from .plot_mooncake_link_study import LINKS, caption, digest, save_figure
from .report_bandwidth import write_csv

TPS = (1, 2, 4, 8)
PARTS = (
    ("parent_wait", "Parent completion wait", "#d8ac46"),
    ("admission_wait", "KV admission wait", "#df775b"),
    ("after_admission", "After admission to first token", "#477fad"),
)


def percent_label(percent):
    """Keep one decimal, truncating only shares greater than 99.5 percent."""
    shown = math.floor(percent * 10) / 10 if percent > 99.5 else percent
    return f"{shown:.1f}%"


def collect(source, tp, repeat=False):
    document = read(source / "results.json")
    runs = [r for r in document["runs"] if r["phase"] > 0 and r.get("trace", "mooncake") == "mooncake"]
    if len(runs) != 10:
        raise ValueError(f"Expected 10 formal Mooncake conditions in {source}")
    summaries, checks = [], []
    for run in runs:
        output = ROOT / run["output"]
        job = output.with_name(output.name + "-job")
        audit, receipt = read(job / "audit.json"), read(job / "job.json")
        condition, config = read(output / "condition.json"), read(output / "effective-config.json")
        if not (audit["status"] == receipt["status"] == "passed" and receipt["returncode"] == 0
                and condition["status"] == "completed"):
            raise ValueError(f"Condition has not passed: {output}")
        if config["hardware_alignment"]["tensor_parallel"] != tp:
            raise ValueError("TP mismatch")
        requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        index = {r["source_id"]: r for r in requests}
        routes = {str(r["request_id"]): r for r in map(json.loads,
                  (output / "native-route-decisions.jsonl").read_text().splitlines())}
        if len(requests) != len(index) or len(index) != 12031 or index.keys() != routes.keys():
            raise ValueError("Request or route IDs are missing/duplicated")
        if sum(r["source_turn"] > 0 for r in requests) != 3959:
            raise ValueError("Returning population mismatch")
        if sum(r["generated"] for r in requests) != audit["output_tokens"] or audit["requests"] != 12031:
            raise ValueError("Audit population mismatch")
        durations = []
        maximum_route_error = maximum_sum_error = 0.0
        for row in requests:
            arrival = row["arrival_s"]
            parent_id = row["previous_source_id"]
            ready = arrival if parent_id is None else max(arrival, index[parent_id]["completed_at"])
            parent = ready - arrival
            admission = row["native_budget_wait_s"]
            submitted = ready + admission
            first = row["first_client_output_at_s"]
            total = row["native_prefill_latency_s"]
            after = total - parent - admission
            route_error = abs(routes[row["source_id"]]["arrival_decision_s"] - submitted)
            maximum_route_error = max(maximum_route_error, route_error)
            if not all(math.isfinite(x) for x in (parent, admission, after, total)):
                raise ValueError("Nonfinite component")
            if min(parent, admission, after) < -1e-8 or route_error > 1e-8:
                raise ValueError("Inconsistent admission or component clocks")
            if not math.isclose(first - submitted, after, rel_tol=0, abs_tol=1e-8):
                raise ValueError("First-output clock does not match reconstructed segment")
            maximum_sum_error = max(maximum_sum_error, abs(parent + admission + after - total))
            durations.append((parent, admission, max(0.0, after), total))
        means = [statistics.mean(r[i] for r in durations) for i in range(4)]
        if means[3] <= 0 or not math.isclose(means[3], run["mean_ttft_s"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Mean TTFT differs from existing report")
        result = dict(tp=tp, gpu_count=8 * tp, repeat=repeat, link=run["link"], method=run["method"],
                      reference=run["reference"], bandwidth_gb_per_s=run["bandwidth_gb_per_s"],
                      requests=12031, returning_requests=3959, mean_ttft_s=means[3])
        for i, (key, _, _) in enumerate(PARTS):
            result[f"mean_{key}_s"] = means[i]
            result[f"{key}_percent"] = 100 * means[i] / means[3]
        if not math.isclose(sum(result[f"{k}_percent"] for k, _, _ in PARTS), 100, abs_tol=1e-8):
            raise ValueError("Component shares do not sum to 100%")
        summaries.append(result)
        checks.append(dict(tp=tp, repeat=repeat, id=run["id"], output=run["output"],
            input_sha256=condition["input_sha256"], requests=12031,
            requests_sha256=digest(output / "requests.jsonl"),
            routes_sha256=digest(output / "native-route-decisions.jsonl"),
            audit_sha256=digest(job / "audit.json"), max_route_clock_error_s=maximum_route_error,
            max_component_sum_error_s=maximum_sum_error, status="passed"))
    return summaries, checks


def panel(ax, rows, title):
    style_axes(ax)
    ordered = [next(r for r in rows if r["method"] == method) for method in METHODS]
    bottom = [0.0] * 3
    for key, _, color in PARTS:
        values = [r[f"{key}_percent"] for r in ordered]
        ax.bar(range(3), values, bottom=bottom, width=.62, color=color,
               edgecolor="white", linewidth=.6)
        for i, percent in enumerate(values):
            if percent >= 5:
                ax.text(i, bottom[i] + percent / 2, percent_label(percent), ha="center",
                        va="center", color="white" if key == "after_admission" else "#292929",
                        fontsize=10, weight="bold")
            bottom[i] += percent
    best = min(round(r["mean_ttft_s"], 6) for r in ordered)
    for i, row in enumerate(ordered):
        winner = round(row["mean_ttft_s"], 6) == best
        ax.text(i, 104, f'{row["mean_ttft_s"]:.6f} s', ha="center", va="bottom",
                fontsize=10, color="#175b32" if winner else "#303e4b",
                weight="bold" if winner else "normal")
    ax.set_ylim(0, 117)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(3), METHODS, fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold", pad=11)
    return ordered


def legend(figure, y):
    from matplotlib.patches import Patch
    figure.legend([Patch(facecolor=color) for _, _, color in PARTS],
                  [label for _, label, _ in PARTS], ncol=3, frameon=False,
                  loc="upper center", bbox_to_anchor=(.5, y), fontsize=12)


def overall(rows, destination):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 3, figsize=(18, 16), facecolor="white")
    figure.subplots_adjust(left=.06, right=.985, top=.87, bottom=.11, hspace=.48, wspace=.19)
    for i, tp in enumerate(TPS):
        for j, (link, bandwidth) in enumerate(LINKS):
            group = [r for r in rows if r["tp"] == tp and r["link"] == link]
            panel(axes[i, j], group, f"TP{tp} ({tp*8} GPUs) | {bandwidth} GB/s")
            if j == 0:
                axes[i, j].set_ylabel("Share of mean client TTFT")
    figure.suptitle("TokenSim | Mooncake | Client TTFT decomposition | TP1 / TP2 / TP4 / TP8",
                   y=.98, fontsize=21, weight="bold")
    legend(figure, .946)
    caption(figure, "llama-7b / FP16 | H100 SXM | 4P+4D | seed 0 | 12,031 requests per condition",
            y=.905, fontsize=12)
    caption(figure, "Bar = 100% of mean client TTFT; number above bar = mean TTFT in seconds.\n"
            "Labels use 1 decimal; shares >99.5% truncate instead of round. Segments below 5% are not labeled.\n"
            "After admission includes engine queueing, compute and transfer; reconstructed from saved timestamps and waits.\n"
            "Bold green total: lowest mean TTFT within the same TP/link. GPU count and KV capacity vary with TP.",
            y=.045, fontsize=11)
    return save_figure(figure, destination, "ttft-decomposition-all-tp")


def per_report(rows, destination, tp, repeat=False):
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(18, 11), facecolor="white")
    grid = figure.add_gridspec(2, 3, height_ratios=(2.8, 2), left=.06, right=.985,
                              top=.81, bottom=.15, hspace=.23, wspace=.19)
    for j, (link, bandwidth) in enumerate(LINKS):
        ax = figure.add_subplot(grid[0, j])
        ordered = panel(ax, [r for r in rows if r["link"] == link], f"{bandwidth} GB/s | data_{link}")
        if j == 0:
            ax.set_ylabel("Share of mean client TTFT")
        tab = figure.add_subplot(grid[1, j])
        keys = [f"mean_{k}_s" for k, _, _ in PARTS] + ["mean_ttft_s"] + [f"{k}_percent" for k, _, _ in PARTS]
        labels = ["Parent wait (s)", "Admission wait (s)", "After admission (s)", "Avg client TTFT (s)",
                  "Parent wait (%)", "Admission wait (%)", "After admission (%)"]
        table(tab, list(METHODS), labels, [[r[key] for r in ordered] for key in keys],
              [None, None, None, "min", None, None, None],
              note="Green: lowest total TTFT; component shares are diagnostic.")
    title = f"TokenSim | Mooncake | TP{tp}" + (" repeat" if repeat else "") + " | TTFT decomposition"
    figure.suptitle(title, y=.98, fontsize=21, weight="bold")
    legend(figure, .937)
    caption(figure, f"llama-7b / FP16 | H100 SXM | 4P+4D ({tp*8} GPUs) | seed 0 | 12,031 requests per condition",
            y=.875, fontsize=12)
    caption(figure, "TTFT = parent completion wait + KV admission wait + after admission to first client token.\n"
            "Parent completion includes writeback. After admission includes engine queueing, compute and transfer.\n"
            "Shares use mean seconds / mean TTFT. Labels use 1 decimal; values >99.5% truncate instead of round.\n"
            "All completed requests included; inf reference is in the CSV only. Native analytical compute.",
            y=.06, fontsize=11)
    return save_figure(figure, destination, "ttft-decomposition")


def explanation(rows, png):
    lines = ["# TTFT 三部分時間組成", "", f"![TTFT 三部分占比]({png})", "",
        "| 組成 | 定義 |", "|---|---|",
        "| 等待前輪完成 | 原始到達後，等指定前輪完整完成（含寫回）的時間；無前輪或前輪已完成則為 0。 |",
        "| 等待 KV 准入 | 前輪條件滿足後，取得 P／D 保守 KV 預算前的等待；原始欄位為 `native_budget_wait_s`。 |",
        "| 准入後至首 token | 完成准入、提交引擎後到 client 第一個 token；含引擎排隊、運算、傳輸，不是純 prefill 計算。 |", "",
        "```text", "parent_wait = max(0, parent.completed_at − arrival_s)，無 parent 時為 0",
        "admission_wait = native_budget_wait_s",
        "after_admission = client_TTFT − parent_wait − admission_wait",
        "client_TTFT = parent_wait + admission_wait + after_admission", "```", "",
        "占比＝每段平均秒數／平均 client TTFT × 100%，也等於各段累計時間／累計 TTFT；不是先逐請求算百分比再平均。", "",
        "柱內占比標籤保留小數點後一位：大於 99.5% 時採截位，其餘正常四捨五入。例如 99.996309% 顯示為 99.9%，75.176950% 顯示為 75.2%。柱高、CSV 與下方六位小數表保留原始計算值；低於 5% 的段落不在柱內標示。", "",
        "分母包含全部 12,031 個完成請求，其中 3,959 筆 returning。初輪亦納入，前輪等待為 0。", "",
        "准入後時間以既有 TTFT 減去兩段等待重建；另外核對原生路由時間戳等於到達＋前輪等待＋准入等待。沒有新增逐 step profiler，也沒有由這張圖拆分引擎內部的計算／傳輸／排隊。", "",
        "這是平均值的組成，不能將三段各自的 P50／P95 相加視為 TTFT P50／P95。占比較高不代表絕對秒數較高；圖上方另標平均 TTFT 秒數。", "",
        "## 秒數與占比", "",
        "| TP | 連線 | 方法 | Avg TTFT s | 前輪等待 s | 准入等待 s | 准入後 s | 前輪 % | 准入 % | 准入後 % |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        keys = ["mean_ttft_s", "mean_parent_wait_s", "mean_admission_wait_s", "mean_after_admission_s",
                "parent_wait_percent", "admission_wait_percent", "after_admission_percent"]
        lines.append(f'| {row["tp"]} | {row["link"]} | {row["method"]} | ' +
                     " | ".join(f"{row[key]:.6f}" for key in keys) + " |")
    lines += ["", "TP1／2／4／8 的 GPU 總數為 8／16／32／64，KV 容量隨 TP 改變。TP1 重跑沒有加入跨 TP 彙整。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "native_research/reports/mooncake-ttft-decomposition-tp1-2-4-8-s0-01")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    faces = times_new_roman()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "svg.fonttype": "path", "svg.hashsalt": "ttft-decomposition"})
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    groups, checks, sources = [], [], []
    for tp, suffix in [(1, "01"), (2, "01"), (4, "01"), (8, "01"), (1, "02")]:
        report = ROOT / f"native_research/reports/five-stage-llama7b-h100-tp{tp}-s0-{suffix}"
        source = report / "delivery-mooncake-complete" if tp == 8 else report
        summaries, verified = collect(source, tp, repeat=suffix == "02")
        checks.extend(verified)
        sources.append(dict(source=str(source), sha256=digest(source / "results.json")))
        write_csv(report / "ttft-decomposition.csv", summaries)
        figures = per_report(summaries, report, tp, suffix == "02")
        (report / "TTFT_DECOMPOSITION.md").write_text(explanation(summaries, figures[0]))
        (report / "ttft-decomposition-verification.json").write_text(json.dumps(dict(
            status="passed", checks=verified, source=sources[-1], selected_font_faces=faces,
            percent_label_policy="one decimal; truncate only values >99.5%, otherwise round; bar heights and CSV retain calculated values",
            generator_sha256=digest(Path(__file__)), figures=[dict(file=name, sha256=digest(report/name)) for name in figures]), indent=2)+"\n")
        if suffix != "02":
            groups.extend(summaries)
    if len({r["input_sha256"] for r in checks}) != 1:
        raise ValueError("Inputs differ between deployments")
    images = overall(groups, destination)
    write_csv(destination / "ttft-decomposition.csv", groups)
    (destination / "TTFT_DECOMPOSITION.md").write_text(explanation(groups, images[0]))
    (destination / "verification.json").write_text(json.dumps(dict(status="passed", conditions=50,
        plotted_finite_conditions=36, checks=checks, sources=sources, selected_font_faces=faces,
        percent_label_policy="one decimal; truncate only values >99.5%, otherwise round; bar heights and CSV retain calculated values",
        simulations_launched=False, generator_sha256=digest(Path(__file__)),
        figures=[dict(file=name, sha256=digest(destination/name)) for name in images]), indent=2)+"\n")
    with ZipFile(destination / "ttft-decomposition-figures.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in images + ["ttft-decomposition.csv", "TTFT_DECOMPOSITION.md", "verification.json"]:
            bundle.write(destination/name, arcname=name)
        for tp, suffix in [(1,"01"),(2,"01"),(4,"01"),(8,"01"),(1,"02")]:
            report = ROOT / f"native_research/reports/five-stage-llama7b-h100-tp{tp}-s0-{suffix}"
            for name in ["ttft-decomposition.png", "ttft-decomposition.svg", "ttft-decomposition.csv", "TTFT_DECOMPOSITION.md", "ttft-decomposition-verification.json"]:
                bundle.write(report/name, arcname=f"tp{tp}-run{suffix}/{name}")
    print(json.dumps(dict(output=str(destination), png_figures=6, conditions_checked=len(checks),
                         plotted_finite_conditions=36, verification="passed")))


if __name__ == "__main__":
    main()
