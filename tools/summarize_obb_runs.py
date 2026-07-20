"""Summarize three or more seed-specific AOFS OBB metric reports."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from aofs.results import summarize_metric_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        help="three or more obb_metrics.json files inside runs with run_manifest.json",
    )
    parser.add_argument("--output", required=True, help="summary JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_metric_files(args.metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
