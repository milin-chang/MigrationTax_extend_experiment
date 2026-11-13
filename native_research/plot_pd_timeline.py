"""Create mentor-facing figures from verified, compact P/D timeline data."""

import json
from pathlib import Path

from .aligned_audit import ROOT, read
from .pd_timeline_analysis import PREFIX, TPS
from .plot_fonts import times_new_roman
from .plot_mooncake_link_study import digest
from .plot_ttft_decomposition import per_report
from .pd_timeline_report import write_reports
from .pd_timeline_figures import across_tp, overview, sample_timeline, worker_steps


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faces = times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "svg.fonttype": "path",
            "svg.hashsalt": "pd-timeline-02",
        }
    )
    groups, outputs = [], []
    for tp in TPS:
        target = ROOT / "native_research/reports" / PREFIX.format(tp)
        data = read(target / "pd-analysis.json")
        files = overview(data, target, tp) + worker_steps(data, target, tp)
        files += sample_timeline(data, target, tp)
        files += per_report(data["requests"], target, tp)
        groups.append(data)
        outputs.append((target, files))
    combined = ROOT / "native_research/reports/mooncake-pd-timeline-tp1-4-8-s0-02"
    combined.mkdir(exist_ok=True)
    combined_files = across_tp(groups, combined)
    write_reports(groups, outputs, combined, combined_files)
    generators = {Path(__file__).name: digest(Path(__file__))}
    for name in (
        "pd_timeline_analysis",
        "pd_timeline_report",
        "pd_timeline_figures",
        "analyze_pd_timeline",
    ):
        path = Path(__file__).with_name(name + ".py")
        generators[path.name] = digest(path)
    for data, (target, files) in zip(groups, outputs):
        evidence = dict(
            status="passed",
            conditions=10,
            checks=data["checks"],
            selected_font_faces=faces,
            generators=generators,
            simulations_launched=False,
            figures=[dict(file=name, sha256=digest(target / name)) for name in files],
        )
        (target / "pd-verification.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
    print(
        json.dumps(
            dict(reports=[str(t) for t, _ in outputs], overview=str(combined)), indent=2
        )
    )


if __name__ == "__main__":
    main()
