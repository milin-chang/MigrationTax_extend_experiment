"""Stream completed P/D recordings into compact, verified report data."""

import argparse
import json

from .aligned_audit import ROOT
from .pd_timeline_analysis import PREFIX, TPS, collect_tp
from .report_bandwidth import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=TPS, nargs="+", default=TPS)
    args = parser.parse_args()
    for tp in args.tp:
        data = collect_tp(tp)
        target = ROOT / "native_research/reports" / PREFIX.format(tp)
        for key in (
            "requests",
            "steps",
            "batch_histogram",
            "transfers",
            "sample_requests",
            "sample_steps",
            "sample_transfers",
        ):
            write_csv(target / f"pd-{key.replace('_', '-')}.csv", data[key])
        (target / "pd-analysis.json").write_text(json.dumps(data, indent=2) + "\n")
        print(f"Exported {target}", flush=True)


if __name__ == "__main__":
    main()
