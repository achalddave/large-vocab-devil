"""Generate class subsets of LVIS for use with infer_specific_classes.py."""

import argparse
import json
import random
from pathlib import Path


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--groundtruth", required=True, type=Path)
    parent_parser.add_argument("--seed", type=int, default=0)
    parent_parser.add_argument("--output-dir", type=Path, required=True)

    subparsers = parser.add_subparsers(dest="command")
    parser_rand = subparsers.add_parser(
        "random", help="Select random set of classes.", parents=[parent_parser]
    )
    parser_rand.add_argument(
        "--freqs",
        choices=["r", "c", "f"],
        nargs="*",
        help="Select subset of size equal to # of classes in specified buckets.",
    )
    parser_rand.add_argument("--size", type=int)

    parser_bucket = subparsers.add_parser(
        "bucket", help="Select bucket of classes.", parents=[parent_parser]
    )
    parser_bucket.add_argument("--freqs", nargs="*", choices=["r", "c", "f"])

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    random.seed(args.seed)

    with open(args.groundtruth, "r") as f:
        classes = json.load(f)["categories"]

    by_freq = {"r": [], "c": [], "f": []}
    for c in classes:
        by_freq[c["frequency"]].append(c)

    if args.command == "random":
        if args.size is not None:
            assert args.freqs is None, "--size and --freqs are mutually exclusive"
            size = args.size
            output_name = f"selected_random{args.size}"
        elif args.freqs is not None:
            assert args.size is None, "--size and --freqs are mutually exclusive"
            output_name = f"selected_random{'+'.join(args.freqs)}"
            size = sum(len(by_freq[f]) for f in args.freqs)
        else:
            raise ValueError("Either --size or --freqs must be specified.")
        selected = {x["id"] for x in random.sample(classes, size)}
    elif args.command == "bucket":
        selected = {x["id"] for f in args.freqs for x in by_freq[f]}
        output_name = f"selected_bucket{'+'.join(args.freqs)}"
    else:
        raise ValueError(f"Unknown command: {args.command}")
    output_name += f"_seed{args.seed}"

    with open(args.output_dir / f"{output_name}.json", "w") as f:
        removed = sorted({x["id"] for x in classes} - selected)
        # We also save the list of removed ids for ease of use (e.g., because in some
        # parts of detectron2 we may not have access to all the ids easily).
        json.dump({"keep": sorted(selected), "skip": removed}, f)


if __name__ == "__main__":
    main()
