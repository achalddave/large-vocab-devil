import argparse
import json
from collections import defaultdict
from pathlib import Path

from lvis import LVIS, LVISResults
from lvdevil.utils.logger import setup_logger
from lvdevil.eval_wrapper import LVISPooledEval


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("annotations_json", type=Path)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--type", default="segm", choices=["segm", "bbox"])
    parser.add_argument("--dets-per-cat", default=-1, type=int)
    parser.add_argument("--max-dets", default=-1, type=int)
    parser.add_argument("--ious", nargs="*", type=float)
    # NOTE: We default to only using areas=all, since we don't report S/M/L APs here.
    parser.add_argument(
        "--areas",
        nargs="*",
        type=str,
        choices=["all", "small", "medium", "large"],
        default=["all"],
    )
    parser.add_argument(
        "--pools",
        nargs="*",
        choices=["all", "r", "c", "f"],
        default=["all", "r", "c", "f"],
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(output=str(args.output_dir.resolve()), name=__file__)
    log_path = args.output_dir / "log.txt"

    results = str(args.results_json)
    if args.dets_per_cat > 0:
        with open(args.results_json, "r") as f:
            results = json.load(f)

        by_cat = defaultdict(list)
        for ann in results:
            by_cat[ann["category_id"]].append(ann)
        results = []
        topk = args.dets_per_cat
        for cat_anns in by_cat.values():
            results.extend(
                sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:topk]
            )

    if args.type == "segm":
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask
        # area.  This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for x in results:
            x.pop("bbox", None)

    gt = LVIS(args.annotations_json)
    results = LVISResults(gt, results, max_dets=args.max_dets)
    lvis_eval = LVISPooledEval(gt, results, iou_type=args.type, pools=args.pools)
    params = lvis_eval.params
    params.max_dets = args.max_dets
    if args.ious:
        params.iou_thrs = args.ious
    if args.areas:
        indices = [
            i for i, area in enumerate(params.area_rng_lbl) if area in args.areas
        ]
        params.area_rng_lbl = [params.area_rng_lbl[i] for i in indices]
        params.area_rng = [params.area_rng[i] for i in indices]

    lvis_eval.run()
    lvis_eval.print_results()
    metrics = {k: v for k, v in lvis_eval.results.items() if k.startswith("AP")}
    logger.info("copypaste: %s,%s", ",".join(map(str, metrics.keys())), "path")
    logger.info(
        "copypaste: %s,%s",
        ",".join(f"{v*100:.2f}" for v in metrics.values()),
        log_path,
    )


if __name__ == "__main__":
    main()
