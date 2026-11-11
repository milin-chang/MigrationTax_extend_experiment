"""Verify and plot completed Mooncake TP1/2/4/8 link studies and TP1 repeat."""

import argparse
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .aligned_audit import ROOT
from .plot_fonts import times_new_roman
from .plot_mooncake_four_parts import COLORS, METHODS, draw_panel, style_axes, table
from .plot_mooncake_link_study import LINKS, caption, digest, save_figure, verify
from .report_bandwidth import write_csv

TPS = (1, 2, 4, 8)
METRICS = (
    ("mean_ttft_s", "Avg client TTFT", "s", True),
    ("p50_ttft_s", "Client TTFT P50", "s", True),
    ("p95_ttft_s", "Client TTFT P95", "s", True),
    ("max_ttft_s", "Client TTFT Max", "s", True),
    ("affected_d_percent", "Affected returning turns", "%", False),
    ("mean_tpot_s", "Avg TPOT", "s/token", False),
)


def records(document, key):
    return document["comparisons"] if key == "affected_d_percent" else document["runs"]


def value(document, link, method, key):
    return next(row[key] for row in records(document, key)
                if row["link"] == link and row["method"] == method)


def overview(document, tp, destination):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(25, 10), facecolor="white")
    grid = figure.add_gridspec(1, 3, left=.045, right=.985, bottom=.15,
                              top=.84, wspace=.21)
    for slot, phase in zip(grid, (1, 3, 4)):
        draw_panel(figure, slot, phase, document)
    figure.suptitle(f"TokenSim | Mooncake | TP{tp} | Stages 1, 3 and 4",
                   y=.97, fontsize=22, weight="bold")
    caption(figure, f"llama-7b / FP16 | H100 SXM 80GB | 4P+4D ({tp * 8} GPUs) | seed 0",
            y=.918, fontsize=13)
    caption(figure, "12,031 requests; 3,959 returning turns. Client TTFT starts at original arrival.\n"
            "Affected = D-first > 2x same-request Serial inf within this TP. Native analytical compute.\n"
            "Bold green: best finite-link method at the same bandwidth; decode miss is diagnostic.",
            y=.055, fontsize=11)
    return save_figure(figure, destination, f"tp{tp}-three-stages")


def comparison(documents, destination, key, label, unit, logarithmic):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(17, 9), facecolor="white")
    grid = figure.add_gridspec(2, 3, height_ratios=(3, 1.35), left=.07, right=.985,
                              bottom=.18, top=.81, hspace=.30, wspace=.23)
    all_values = [value(document, link, method, key)
                  for document in documents.values() for link, _ in LINKS for method in METHODS]
    if not all(math.isfinite(v) and v >= 0 for v in all_values):
        raise ValueError(f"Invalid {key}")
    if logarithmic and min(all_values) <= 0:
        raise ValueError(f"Logarithmic chart requires positive {key}")
    for column, (link, bandwidth) in enumerate(LINKS):
        ax = figure.add_subplot(grid[0, column])
        style_axes(ax)
        for method, color, marker in zip(METHODS, COLORS, ("o", "s", "^")):
            values = [value(documents[tp], link, method, key) for tp in TPS]
            ax.plot(range(4), values, color=color, marker=marker, linewidth=2,
                    markersize=7, label=method)
            for index, (tp, v) in enumerate(zip(TPS, values)):
                best = min(round(value(documents[tp], link, m, key), 6) for m in METHODS)
                if round(v, 6) == best:
                    ax.scatter([index], [v], s=115, facecolors="none", edgecolors=color,
                               linewidths=1.2, zorder=4)
        ax.set_title(f"{bandwidth} GB/s | data_{link}", weight="bold")
        ax.set_xticks(range(4), [f"TP{tp}\n{tp * 8} GPUs" for tp in TPS])
        ax.set_xlim(-.18, 3.18)
        if logarithmic:
            ax.set_yscale("log")
            ax.set_ylim(min(all_values) / 1.6, max(all_values) * 1.8)
        elif key == "affected_d_percent":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, max(all_values) * 1.15)
        if column == 0:
            ax.set_ylabel(f"{label} ({unit}{'; log scale' if logarithmic else ''})")
        tab = figure.add_subplot(grid[1, column])
        table(tab, list(METHODS), [f"TP{tp} ({unit})" for tp in TPS],
              [[value(documents[tp], link, method, key) for method in METHODS] for tp in TPS],
              ["min"] * 4, note="Bold green: best method within each TP and link.")
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
                  bbox_to_anchor=(.5, .92), fontsize=13)
    figure.suptitle(f"TokenSim | Mooncake | TP1 / TP2 / TP4 / TP8 | {label}",
                   fontsize=20, y=.975, weight="bold")
    scope = ("Each TP uses its own Serial inf baseline; affected shares have different reference latencies."
             if key == "affected_d_percent" else
             "Client TTFT includes parent and admission waits; original seconds retained."
             if key != "mean_tpot_s" else
             "TPOT is the equal-request mean of native decode_time_sum / max(1, output_tokens - 1).")
    caption(figure, "Same shifted-v1 seed-0 input: 12,031 requests; 3,959 returning turns.\n"
            "4P+4D fixed; total GPU count and KV capacity increase with TP. Native analytical compute.\n"
            + scope, y=.07, fontsize=11)
    return save_figure(figure, destination, f"tp-comparison-{key.replace('_s', '-s').replace('_', '-')}")


def repeat_check(original, repeated):
    rows = []
    for first in original["runs"]:
        second = next(r for r in repeated["runs"]
                      if (r["link"], r["method"]) == (first["link"], first["method"]))
        first_path = ROOT / first["output"] / "requests.jsonl"
        second_path = ROOT / second["output"] / "requests.jsonl"
        a = {r["source_id"]: r for r in map(json.loads, first_path.read_text().splitlines())}
        b = {r["source_id"]: r for r in map(json.loads, second_path.read_text().splitlines())}
        if a.keys() != b.keys():
            raise ValueError("TP1 repeat source IDs differ")
        unequal = sum(a[k] != b[k] for k in a)
        rows.append(dict(link=first["link"], method=first["method"],
                         requests=len(a), differing_request_records=unequal,
                         original_sha256=digest(first_path), repeat_sha256=digest(second_path),
                         byte_identical=digest(first_path) == digest(second_path),
                         original_mean_ttft_s=first["mean_ttft_s"],
                         repeat_mean_ttft_s=second["mean_ttft_s"],
                         delta_mean_ttft_s=second["mean_ttft_s"] - first["mean_ttft_s"]))
    return rows


def repeat_plot(original, repeated, checks, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor="white")
    figure.subplots_adjust(left=.065, right=.985, top=.80, bottom=.23, wspace=.22)
    for ax, (link, bandwidth) in zip(axes, LINKS):
        style_axes(ax)
        for offset, (doc, label, color, hatch) in enumerate([
            (original, "TP1 original", "#345f99", None),
            (repeated, "TP1 repeat", "#b4c2d5", "///"),
        ]):
            values = [value(doc, link, method, "mean_ttft_s") for method in METHODS]
            bars = ax.bar([x + (offset - .5) * .36 for x in range(3)], values, .34,
                          label=label, color=color, hatch=hatch, edgecolor="#2a405b", linewidth=.5)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
        ax.set_xticks(range(3), METHODS)
        ax.set_title(f"{bandwidth} GB/s", weight="bold")
        ax.set_ylim(0, max(value(doc, link, m, "mean_ttft_s")
                           for doc in (original, repeated) for m in METHODS) * 1.2)
    axes[0].set_ylabel("Avg client TTFT (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(.5, .92), fontsize=12)
    figure.suptitle("TokenSim | Mooncake | TP1 repeatability check", fontsize=20,
                   y=.975, weight="bold")
    unequal = sum(r["differing_request_records"] for r in checks)
    caption(figure, f"Same seed, input and configuration; 10 audited conditions per run.\n"
            f"Compared 120,310 source-paired request records: {unequal:,} differing records.\n"
            "Same-seed repeat is not an independent seed and is not averaged into the TP comparison.",
            y=.09, fontsize=11)
    return save_figure(figure, destination, "tp1-repeatability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "native_research/reports/mooncake-tp1-2-4-8-s0-01")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faces = times_new_roman()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "svg.fonttype": "path", "svg.hashsalt": "mooncake-tp-sweep",
                         "axes.titlepad": 13})
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    documents, verification = {}, {}
    for tp in TPS:
        source = ROOT / f"native_research/reports/five-stage-llama7b-h100-tp{tp}-s0-01"
        if tp == 8:
            source /= "delivery-mooncake-complete"
        documents[tp], checks = verify(source, tp)
        verification[f"tp{tp}"] = dict(source=str(source), source_sha256=digest(source / "results.json"), checks=checks)
    repeat_source = ROOT / "native_research/reports/five-stage-llama7b-h100-tp1-s0-02"
    repeated, checks = verify(repeat_source, 1)
    verification["tp1_repeat"] = dict(source=str(repeat_source), source_sha256=digest(repeat_source / "results.json"), checks=checks)
    hashes = {row["input_sha256"] for document in [*documents.values(), repeated] for row in document["runs"]}
    if len(hashes) != 1:
        raise ValueError("Inputs differ across TP or repeat")
    repeat = repeat_check(documents[1], repeated)
    figures = []
    for tp in TPS:
        figures += overview(documents[tp], tp, destination)
    for metric in METRICS:
        figures += comparison(documents, destination, *metric)
    figures += repeat_plot(documents[1], repeated, repeat, destination)
    metric_rows, affected_rows = [], []
    for tp, document in documents.items():
        for record in document["runs"]:
            metric_rows.append(dict(tp=tp, gpu_count=8 * tp,
                **{key: record[key] for key in ("link", "method", "reference", "requests",
                    "mean_ttft_s", "p50_ttft_s", "p95_ttft_s", "max_ttft_s",
                    "mean_lifecycle_s", "mean_tpot_s", "output_tokens_per_s")}))
        for record in document["comparisons"]:
            affected_rows.append(dict(tp=tp, gpu_count=8 * tp, **record))
    write_csv(destination / "request-metrics.csv", metric_rows)
    # Earlier TP8 reports include extra cache diagnostics; retain their fields
    # and leave unavailable diagnostics blank for the other deployments.
    affected_keys = list(dict.fromkeys(key for row in affected_rows for key in row))
    write_csv(destination / "affected-share.csv", [
        {key: row.get(key) for key in affected_keys} for row in affected_rows
    ])
    write_csv(destination / "tp1-repeatability.csv", repeat)
    lines = ["# Mooncake TP1／2／4／8 比較圖", "",
             "Times New Roman；llama-7b／FP16；H100 SXM；4P＋4D；seed 0。", "",
             "每個 TP 納入 10 組已稽核的 Mooncake 正式條件；另驗證 TP1 同設定重跑 10 組。TP8 採已完成的 Mooncake delivery，未納入 ServeGen 或 sanity。", "",
             "TP1／2／4／8 分別為 8／16／32／64 張模擬 GPU，KV 容量也隨 TP 改變。這是部署設定比較。", "",
             "TTFT 比較圖使用對數縱軸，所有數字仍為秒；下表保留六位小數，粗體綠底為同 TP、同頻寬下最佳方法。Affected 各自使用該 TP 的 Serial inf reference。", "",
             "## 圖片", ""]
    lines += [f"- [{name}]({name})" for name in figures if name.endswith(".png")]
    lines += ["", "## Avg client TTFT（秒）", "",
              "| TP | GPU 數 | 通訊 GB/s | Serial | PPD | MigrationTax |",
              "|---:|---:|---:|---:|---:|---:|"]
    for tp in TPS:
        for link, bandwidth in LINKS:
            values = [value(documents[tp], link, method, "mean_ttft_s") for method in METHODS]
            best = min(round(v, 6) for v in values)
            cells = [f"**{v:.6f}**" if round(v, 6) == best else f"{v:.6f}" for v in values]
            lines.append(f"| {tp} | {tp*8} | {bandwidth} | " + " | ".join(cells) + " |")
    lines += ["", "## TP1 重現性", "",
              f"逐筆比較 {sum(r['requests'] for r in repeat):,} 筆配對紀錄，完整 request record 不同者為 {sum(r['differing_request_records'] for r in repeat):,} 筆。",
              "同 seed 重跑不視為新 seed，不加入 TP1 平均；細節見 `tp1-repeatability.csv`。", "",
              "## 附件與重畫指令", "",
              "- `request-metrics.csv`：40 組正式條件的 TTFT、生命週期、TPOT 與吞吐。",
              "- `affected-share.csv`：36 組有限頻寬條件的 affected 與 decode miss。",
              "- `verification.json`：50 組原始逐筆重算、完成稽核、來源 SHA-256 與字體證據。",
              "- 每張 PNG 另附同名 SVG；`mooncake-tp-comparison-figures.zip` 包含全部圖表與彙整。", "",
              "在專案既有 Python／依賴環境下：", "", "```bash",
              "python3 -m native_research.plot_mooncake_tp_sweep", "```", "",
              "此命令重新讀取既有結果、覆寫本圖表資料夾，不啟動模擬或修改原始 run。", ""]
    (destination / "FIGURES.md").write_text("\n".join(lines))
    verification.update(status="passed", audited_conditions=50, simulations_launched=False,
        same_input_across_deployments=True, input_sha256=next(iter(hashes)),
        tp1_repeat_checks=repeat, font_family="Times New Roman", selected_font_faces=faces,
        generator_sha256=digest(Path(__file__)),
        figures=[dict(file=name, sha256=digest(destination / name)) for name in figures])
    (destination / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    with ZipFile(destination / "mooncake-tp-comparison-figures.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in figures + ["request-metrics.csv", "affected-share.csv", "tp1-repeatability.csv", "FIGURES.md", "verification.json"]:
            bundle.write(destination / name, arcname=name)
    print(json.dumps(dict(output=str(destination), png_figures=len(figures)//2,
                          audited_conditions=50, differing_repeat_records=sum(r["differing_request_records"] for r in repeat))))


if __name__ == "__main__":
    main()
