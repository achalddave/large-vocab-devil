import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from netcal.AbstractCalibration import AbstractCalibration
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

from lvdevil.eval_wrapper import EvalWrapper


def load_tp_fp_fn(eval_obj):
    is_lvis = hasattr(eval_obj, "eval_imgs")
    gt = eval_obj.lvis_gt if is_lvis else eval_obj.cocoGt
    eval_imgs = eval_obj.eval_imgs if is_lvis else eval_obj.evalImgs
    if not eval_imgs:
        raise ValueError("Please run evaluate() first")
    dt_matched_key = "dt_matches" if is_lvis else "dtMatches"
    gt_matched_key = "gt_matches" if is_lvis else "gtMatches"
    dt_ids_key = "dt_ids" if is_lvis else "dtIds"
    gt_ids_key = "gt_ids" if is_lvis else "gtIds"

    true_positives, false_positives, false_negatives = set(), set(), set()
    for e in eval_imgs:
        if e is None:
            continue
        matched = e[dt_matched_key] != 0  # Shape (num_detections, )
        for dt_id, is_matched in zip(e[dt_ids_key], matched.squeeze(0)):
            if is_matched.item():
                true_positives.add(dt_id)
            else:
                false_positives.add(dt_id)
        gt_matched = (e[gt_matched_key] != 0).squeeze(0)
        false_negatives |= {g for i, g in enumerate(e[gt_ids_key]) if not gt_matched[i]}
    return true_positives, false_positives, false_negatives


def evaluate(
    annotations,
    results,
    iou=0.75,
    iou_type="segm",
    dataset="lvis",
    n_bins=10,
    commercial_only=False,
    subset=1.0,
    seed=0.0,
    min_score=0.0,
    vis_dir=None,
    vis_per_class=False,
    max_dets=300,
    max_dets_per_class=-1,
):
    """
    Args:
        annotations (str, Path, or dict): Path to COCO/LVIS-style annotations, or
            dict containing the annotations.
        results (str, Path, or dict): Path to COCO/LVIS-style results, or dict
            containing the results.
        iou (float): IoU threshold to evaluate calibration at.
        iou_type (str): segm or bbox
        dataset (str): lvis or coco
        n_bins (int): Number of bins for calibration eval
        commercial_only (bool): Use only commercial images for COCO. Used to match
            Küppers et al. setting.
        subset (float): If <1.0, use a random subset of this portion for eval.
        seed (float): Used to seed the rng for subset selection.
        min_score (float): If specified, ignore detections below this threshold for
            calibration evaluation. This flag does not affect the AP calculation.
            This should generally be left at 0, but can be set to 0.3 to match the
            Küppers et al. setting.
        vis_dir (str, Path, or None): If specified, output reliability diagrams to this
            directory.
        vis_per_class (bool): If vis_dir is specified and vis_per_class is True, output
            a reliability diagram for each class.
        max_dets (int): Limit number of detections per image.
        max_dets_per_class (int): Limit number of detections per class.
    """
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        plotter = ReliabilityDiagram(bins=n_bins, detection=True, metric="ECE")
    else:
        plotter = None

    rng = random.Random(seed)
    eval_wrapper = EvalWrapper(
        annotations,
        results,
        dataset_type=dataset,
        ious=[iou],
        iou_type=iou_type,
        max_dets=max_dets,
        max_dets_per_class=max_dets_per_class,
    )
    eval_obj = eval_wrapper.construct_eval(use_cats=True)
    is_lvis = eval_wrapper.is_lvis()
    params = eval_obj.params
    gt = eval_obj.lvis_gt if is_lvis else eval_obj.cocoGt

    if commercial_only:
        # Licenses 1, 2, 3 are NonCommercial
        valid_licenses = {4, 5, 6, 7, 8}
        orig_img_ids = params.img_ids if is_lvis else params.imgIds
        img_ids = [i for i in orig_img_ids if gt.imgs[i]["license"] in valid_licenses]
        logging.info(f"Selecting {len(img_ids)}/{len(orig_img_ids)} commercial images.")
        if is_lvis:
            params.img_ids = img_ids
        else:
            params.imgIds = img_ids

    if subset < 1.0:
        img_ids = params.img_ids if is_lvis else params.imgIds
        k = int(round(len(img_ids) * subset))
        logging.info(f"Selecting {k}/{len(img_ids)} images randomly.")
        rng.shuffle(img_ids)
        if is_lvis:
            params.img_ids = img_ids[:k]
        else:
            params.imgIds = img_ids[:k]

    eval_obj.evaluate()

    # True positive set
    true_positives, false_positives, missed_gt = load_tp_fp_fn(eval_obj)

    eval_obj.accumulate()
    eval_obj.summarize()

    # Map class id to list of (detection: dict, is_matched: bool)
    class_dets = defaultdict(list)
    for dt_id in true_positives:
        ann = eval_wrapper.results.anns[dt_id]
        class_dets[ann["category_id"]].append((ann, True))
    for dt_id in false_positives:
        ann = eval_wrapper.results.anns[dt_id]
        class_dets[ann["category_id"]].append((ann, False))

    if min_score > 0.0:
        class_dets = {
            c: [x for x in dets if x[0]["score"] > min_score]
            for c, dets in class_dets.items()
        }
        # Remove empty classes.
        class_dets = {c: v for c, v in class_dets.items() if v}

    # Map class id to tuple of (scores, is_matched)
    scores_matched = {
        c: (
            np.array([d["score"] for d, _ in dets])[:, np.newaxis],  # scores, (n, 1)
            np.array([m for _, m in dets])[:, np.newaxis],  # is_matched, (n, 1)
        )
        for c, dets in class_dets.items()
    }
    classes = sorted(scores_matched.keys())

    all_scores = np.vstack([scores_matched[c][0] for c in classes])
    all_is_matched = np.vstack([scores_matched[c][1] for c in classes])

    ece = ECE([n_bins], detection=True)

    output_metrics = {}
    output_metrics["AP"] = eval_obj.results["AP"]
    if is_lvis:
        for f in ("f", "c", "r"):
            output_metrics[f"AP{f}"] = eval_obj.results[f"AP{f}"]
    output_metrics["ece-overall"] = ece.measure(all_scores, all_is_matched)
    if plotter:
        fig = plotter.plot(
            all_scores, all_is_matched, filename=vis_dir / f"overall.pdf"
        )
        plt.close(fig)

    # NOTE: Skips classes with no predictions nor groundtruth; Assigns ECE of 1.0 for
    # classes with groundtruth but no predictions.
    per_class_eces = {}
    predicted_classes = set(scores_matched.keys())
    missed_classes = {gt.anns[g]["category_id"] for g in missed_gt}
    for cid in missed_classes | predicted_classes:
        if cid not in predicted_classes:  # Present but not predicted
            # Skip class from calibration error.
            continue
        else:
            scores, is_matched = scores_matched[cid]
            per_class_eces[cid] = ece.measure(scores, is_matched)
            if plotter and vis_per_class:
                cname = gt.cats[cid].get("synset", gt.cats[cid]["name"])
                fig = plotter.plot(
                    scores, is_matched, filename=vis_dir / f"class-{cid}-{cname}.pdf"
                )
                plt.close(fig)
    output_metrics["ece-per-class"] = np.mean(list(per_class_eces.values()))

    if eval_wrapper.is_lvis():
        # Map frequency to category ids (eval_obj.freq_groups maps to indices)
        for f, indices in enumerate(eval_obj.freq_groups):
            freq = eval_obj.params.img_count_lbl[f]
            cat_ids = [eval_obj.params.cat_ids[i] for i in indices]
            cat_ids = [c for c in cat_ids if c in scores_matched]
            freq_scores = np.vstack([scores_matched[c][0] for c in cat_ids])
            freq_matched = np.vstack([scores_matched[c][1] for c in cat_ids])
            output_metrics[f"ece-freq-{freq}"] = ece.measure(freq_scores, freq_matched)
            output_metrics[f"ece-per-class-{freq}"] = np.mean(
                [per_class_eces[c] for c in cat_ids if c in per_class_eces]
            )
            if plotter:
                fig = plotter.plot(
                    freq_scores, freq_matched, filename=vis_dir / f"freq-{freq}.pdf"
                )
                plt.close(fig)

    return output_metrics


class IdentityCalibration(AbstractCalibration):
    def fit(self, *args, **kwargs):
        pass

    def transform(self, X):
        return X

    def clear(self):
        pass


class ZeroScoreCalibration(AbstractCalibration):
    def fit(self, *args, **kwargs):
        pass

    def transform(self, X):
        return np.zeros_like(X)

    def clear(self):
        pass
