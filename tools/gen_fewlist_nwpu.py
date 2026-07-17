"""Generate reproducible NWPU support splits for AOFS."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from aofs.fewshot import generate_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="NWPU dataset root containing training.txt")
    parser.add_argument("--output-root", required=True, help="seed-specific split output directory")
    parser.add_argument("--profile", required=True, choices=("paper", "robust"))
    parser.add_argument("--shot", required=True, type=int, choices=(3, 5, 10))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=None,
        help="instance cap per class and image; robust defaults to 1, paper has no cap",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_per_image = args.max_per_image
    if max_per_image is None and args.profile == "robust":
        max_per_image = 1
    outputs = generate_split(
        data_root=args.data_root,
        output_root=args.output_root,
        profile=args.profile,
        shot=args.shot,
        seed=args.seed,
        max_per_image=max_per_image,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
