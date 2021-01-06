import argparse
import json
import logging
import random
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from netcal.AbstractCalibration import AbstractCalibration
from netcal.binning import BBQ, ENIR, HistogramBinning, IsotonicRegression
from netcal.scaling import BetaCalibration, LogisticCalibration, TemperatureScaling

from fvcore.common.config import CfgNode as CN
from lvdevil import calib
from lvdevil.eval_wrapper import EvalWrapper
from lvdevil.utils.logger import setup_logger


def subset_annotations(anns, image_ids):
    anns = anns.copy()
    image_ids = set(image_ids)
    anns["images"] = [x for x in anns["images"] if x["id"] in image_ids]
    anns["annotations"] = [x for x in anns["annotations"] if x["image_id"] in image_ids]
    return anns


def subset_results(results, image_ids):
    return [x for x in results if x["image_id"] in image_ids]


def get_cfg():
    cfg = CN()
    cfg.TEST = CN()
    cfg.TEST.ANNOTATIONS = ""
    cfg.TEST.RESULTS = ""
    cfg.TEST.N_BINS = 15
    cfg.TEST.IOU = 0.75
    cfg.TEST.IOU_TYPE = "segm"
    cfg.TEST.MAX_DETS_EVAL = -1
    cfg.TEST.MAX_DETS_PER_CLASS_EVAL = -1

    # If enabled, report results on k-fold cross validation splits.
    cfg.CROSS_VAL = CN({"ENABLED": False})
    cfg.CROSS_VAL.FOLDS = 5

    cfg.TRAIN = CN()
    # What data to calibrate on. Options:
    # - none: Don't train calibration.
    # - cv-train: Train on cross-validation train split.
    # - cv-test: Train on cross-validation test split.
    # - test: Train on full test split.
    # - custom: Train on annotations and results specified in TRAIN.ANNOTATIONS,
    #       TRAIN.RESULTS.
    cfg.TRAIN.TRAIN_SET = "none"
    # Specify training annotations and results.
    cfg.TRAIN.ANNOTATIONS = ""
    cfg.TRAIN.RESULTS = ""

    cfg.METHOD = CN()
    # One of: HistogramBinning, IsotonicRegression, BBQ, ENIR, LogisticCalibration,
    # BetaCalibration, TemperatureScaling
    cfg.METHOD.NAME = "HistogramBinning"
    # One of "overall" (calibrate across all categories), "frequency" (calibrate per
    # frequency bin), or "per-class" (calibrate per class).
    # Note that "per-class" will likely error when using cross-validation, since some
    # splits will have classes in the validation split which are missing from the train
    # split.
    cfg.METHOD.GROUPING = "overall"
    # How to calibrate classes with no training
    cfg.METHOD.PER_CLASS = CN()
    # When using METHOD.GROUPING = "per-class, some classes may have no predictions in
    # the training set, or may have only false-positive predictions.
    # MISSING_SCORE, NO_TP_SCORE specify strategies for dealing with classes with no
    # predictions, or only false-positive predictions, respectively.
    # Options:
    #   - zero: Set score to zero
    #   - keep: Keep score at original value
    #   - by-frequency: Fit calibration for classes with the same frequency, and
    #       use the per-frequency calibration for this class.
    cfg.METHOD.PER_CLASS.MISSING_SCORE = "zero"
    # How to assign scores for classes with no predictions in the training set.
    cfg.METHOD.PER_CLASS.NO_TP_SCORE = "zero"
    cfg.METHOD.HIST = CN()
    # Number of bins for histogram binning. If -1, set to cfg.TEST.N_BINS.
    cfg.METHOD.HIST.N_BINS = -1
    cfg.METHOD.BBQ = CN()
    cfg.METHOD.BBQ.SCORE_FN = "BIC"  # "BIC" or "AIC"
    cfg.METHOD.ENIR = CN()
    cfg.METHOD.ENIR.SCORE_FN = "BIC"  # "BIC" or "AIC"
    # Some methods, like histogram binning, assign the same scores to many detections,
    # resulting in an undefined ranking. Setting MAINTAIN_RANK to True modifies these
    # scores so that the original ranking is respected, while also ensuring the average
    # score within a score bin stays the same.
    cfg.METHOD.MAINTAIN_RANK = True

    cfg.FEATURES = CN()
    cfg.FEATURES.ENABLED = False
    cfg.FEATURES.INSTANCE_COUNT = True
    cfg.FEATURES.POSITION = True
    cfg.FEATURES.SIZE = True

    cfg.DATASET = "lvis"
    cfg.SEED = 0
    cfg.NAME = ""
    cfg.VISUALIZE = False
    # Whether to visualize reliability matrices for each class.
    # If VISUALIZE is False, this is ignored.
    cfg.VISUALIZE_PER_CLASS = False
    # Whether to save results json. Disabled by default to save disk space.
    cfg.SAVE_RESULTS = False

    return cfg


class PerClassDetectionCalibrator:
    def __init__(self, calibrators: Dict[int, AbstractCalibration]):
        """
        Args:
            calibrators (Dict[int, AbstractCalibrator]): Maps class id to a
                calibrator for that class.
        """
        self.calibrators = calibrators

    def transform(self, detections: List[Dict], maintain_rank: bool):
        """
        Args:
            detections (List[Dict]): List of detections in COCO format.

        Returns:
            detections (List[Dict]): List of detections in COCO format, with
                calibrated scores.
        """
        dets_by_class = defaultdict(list)
        detections = [x.copy() for x in detections]
        for d in detections:
            dets_by_class[d["category_id"]].append(d)
        by_scores = defaultdict(list)
        for c, dets in dets_by_class.items():
            scores = np.array([d["score"] for d in dets])[:, None]
            new_scores = self.calibrators[c].transform(scores)
            if new_scores.ndim == 2:
                new_scores = new_scores.squeeze(1)
            for i, d in enumerate(dets):
                d["_orig_score"] = d["score"]
                d["score"] = new_scores[i]
                by_scores[new_scores[i]].append(d)
        if maintain_rank:
            # Some methods, like histogram binning, assign the same scores to many
            # detections, resulting in an undefined ranking.
            # We modify scores within bins such that the original rank is respected,
            # but the average score does not change.
            max_delta = 1e-16  # Max amount to change scores by.
            for new_score, dets in by_scores.items():
                if all(dets[0]["_orig_score"] == x["_orig_score"] for x in dets[1:]):
                    continue
                if new_score + max_delta < 1:
                    deltas = np.linspace(0, max_delta, len(dets))
                else:
                    deltas = np.linspace(-max_delta, 0, len(dets))
                for i, d in enumerate(sorted(dets, key=lambda x: x["_orig_score"])):
                    d["score"] += deltas[i]
        return detections


def fit_calibrators(
    calibrator_fn: Callable[[], AbstractCalibration],
    eval_obj,
    grouping,
    missing_calibration: str,
    no_tp_calibration: str,
):
    """
    Args:
        calibrator_fn (() -> AbstractCalibration)
        eval_obj (coco.COCOeval or lvis.LVISEval)
        grouping (str)
    """
    tp_dets, fp_dets, _ = calib.load_tp_fp_fn(eval_obj)
    class_dets = defaultdict(list)
    is_lvis = hasattr(eval_obj, "lvis_dt")
    results = eval_obj.lvis_dt if is_lvis else eval_obj.cocoDt
    for dt_id in tp_dets:
        ann = results.anns[dt_id]
        class_dets[ann["category_id"]].append((ann, True))
    for dt_id in fp_dets:
        ann = results.anns[dt_id]
        class_dets[ann["category_id"]].append((ann, False))
    # Map class id to tuple of (scores, is_matched)
    scores_matched = {
        c: (
            np.array([d["score"] for d, _ in dets])[:, np.newaxis],  # scores, (n, 1)
            np.array([m for _, m in dets])[:, np.newaxis],  # is_matched, (n, 1)
        )
        for c, dets in class_dets.items()
    }

    # Map category id to a calibrator for the category's frequency bucket.
    calibrators_by_frequency = {}
    if grouping == "frequency" or (
        grouping == "per-class"
        and "by-frequency" in (missing_calibration, no_tp_calibration)
    ):
        if not is_lvis and grouping == "frequency":
            raise NotImplementedError("grouping='frequency' only supported for LVIS")
        elif not is_lvis and grouping == "per-class":
            raise NotImplementedError("'by-frequency' only supported for LVIS")
        for freq_indices in eval_obj.freq_groups:
            freq_calib = calibrator_fn()
            freq_cids = [eval_obj.params.cat_ids[i] for i in freq_indices]
            valid_freq_cids = [c for c in freq_cids if c in scores_matched]
            freq_scores = np.vstack([scores_matched[c][0] for c in valid_freq_cids])
            freq_matched = np.vstack([scores_matched[c][1] for c in valid_freq_cids])
            freq_calib.fit(freq_scores, freq_matched)
            calibrators_by_frequency.update({c: freq_calib for c in freq_cids})

    if grouping == "overall":
        params = eval_obj.params
        cat_ids = params.cat_ids if hasattr(params, "cat_ids") else params.catIds
        classes = sorted(scores_matched.keys())
        all_scores = np.vstack([scores_matched[c][0] for c in classes])
        all_is_matched = np.vstack([scores_matched[c][1] for c in classes])
        calibrator: AbstractCalibration = calibrator_fn()
        calibrator.fit(all_scores, all_is_matched)
        return PerClassDetectionCalibrator({c: calibrator for c in cat_ids})
    elif grouping == "frequency":
        return PerClassDetectionCalibrator(calibrators_by_frequency)
    elif grouping == "per-class":
        calibrators = {}
        params = eval_obj.params
        cat_ids = params.cat_ids if hasattr(params, "cat_ids") else params.catIds

        def special_calibrator(category_id, strategy):
            if strategy == "zero":
                return calib.ZeroScoreCalibration()
            elif strategy == "keep":
                return calib.IdentityCalibration()
            elif strategy == "by-frequency":
                assert category_id in calibrators_by_frequency
                return calibrators_by_frequency[category_id]
            else:
                raise NotImplementedError

        for c, (scores_c, matched_c) in scores_matched.items():
            if not np.any(matched_c):
                calibrators[c] = special_calibrator(c, no_tp_calibration)
            else:
                calibrators[c] = calibrator_fn()
                calibrators[c].fit(scores_c, matched_c)
        for c in set(cat_ids) - set(scores_matched.keys()):
            assert c not in calibrators
            calibrators[c] = special_calibrator(c, missing_calibration)
        return PerClassDetectionCalibrator(calibrators)
    else:
        raise NotImplementedError


class CalibrationTrainer:
    def get_calibrator_fn(self, cfg):
        if cfg.METHOD.NAME == "HistogramBinning":
            bins = cfg.METHOD.HIST.N_BINS
            if bins == -1:
                bins = cfg.TEST.N_BINS
            return HistogramBinning(bins, detection=True)
        elif cfg.METHOD.NAME == "IsotonicRegression":
            return IsotonicRegression(detection=True)
        elif cfg.METHOD.NAME == "BBQ":
            return BBQ(score_function=cfg.METHOD.BBQ.SCORE_FN, detection=True)
        elif cfg.METHOD.NAME == "ENIR":
            return ENIR(score_function=cfg.METHOD.ENIR.SCORE_FN, detection=True)
        elif cfg.METHOD.NAME == "LogisticCalibration":
            return LogisticCalibration(detection=True)
        elif cfg.METHOD.NAME == "BetaCalibration":
            return BetaCalibration(detection=True)
        elif cfg.METHOD.NAME == "TemperatureScaling":
            return TemperatureScaling(detection=True)
        else:
            raise NotImplementedError

    def _train(self, cfg, annotations, results):
        train_eval_wrapper = EvalWrapper(
            annotations,
            results,
            dataset_type=cfg.DATASET,
            ious=[cfg.TEST.IOU],
            iou_type=cfg.TEST.IOU_TYPE,
            max_dets=cfg.TEST.MAX_DETS_EVAL,
            max_dets_per_class=cfg.TEST.MAX_DETS_PER_CLASS_EVAL,
        )
        train_eval_obj = train_eval_wrapper.construct_eval(use_cats=True)
        train_eval_obj.evaluate()
        det_calib = fit_calibrators(
            lambda: self.get_calibrator_fn(cfg),
            train_eval_obj,
            cfg.METHOD.GROUPING,
            missing_calibration=cfg.METHOD.PER_CLASS.MISSING_SCORE,
            no_tp_calibration=cfg.METHOD.PER_CLASS.NO_TP_SCORE,
        )
        return det_calib

    def _evaluate(self, cfg, annotations, results, vis_dir=None):
        kwargs = {}
        if cfg.VISUALIZE and vis_dir is not None:
            kwargs["vis_dir"] = vis_dir
            kwargs["vis_per_class"] = cfg.VISUALIZE_PER_CLASS
        metrics = calib.evaluate(
            annotations,
            results,
            iou=cfg.TEST.IOU,
            iou_type=cfg.TEST.IOU_TYPE,
            dataset=cfg.DATASET,
            n_bins=cfg.TEST.N_BINS,
            max_dets=cfg.TEST.MAX_DETS_EVAL,
            max_dets_per_class=cfg.TEST.MAX_DETS_PER_CLASS_EVAL,
            **kwargs,
        )
        # Compute AP with default iou range.
        eval_wrapper_iou5095 = EvalWrapper(
            annotations,
            results,
            dataset_type=cfg.DATASET,
            iou_type=cfg.TEST.IOU_TYPE,
            max_dets=cfg.TEST.MAX_DETS_EVAL,
            max_dets_per_class=cfg.TEST.MAX_DETS_PER_CLASS_EVAL,
        )
        eval_obj = eval_wrapper_iou5095.construct_eval(use_cats=True)
        eval_obj.run()
        for k in ["AP", "APs", "APm", "APl", "APr", "APc", "APf"]:
            if k in eval_obj.results:
                metrics[f"{k}-IoU50:95"] = eval_obj.results[k]

        if cfg.DATASET == "lvis":
            # Evaluate AP-pooled at IoU 50-95
            pooled_eval_obj = eval_wrapper_iou5095.construct_pooled_eval(
                pools=["all", "r", "c", "f"]
            )
            pooled_eval_obj.run()
            metrics.update(
                {
                    "AP-pooled-IoU50:95": pooled_eval_obj.results["AP-pooled-all"],
                    "AP-pooled-r-IoU50:95": pooled_eval_obj.results["AP-pooled-r"],
                    "AP-pooled-c-IoU50:95": pooled_eval_obj.results["AP-pooled-c"],
                    "AP-pooled-f-IoU50:95": pooled_eval_obj.results["AP-pooled-f"],
                }
            )
        return metrics

    def run_calibration(self, cfg, vis_dir=None):
        logger = logging.getLogger(__file__)
        with open(cfg.TEST.RESULTS, "r") as f:
            test_results = json.load(f)
        # List of (train_ann, train_results, test_ann, test_results)
        folds = []
        if cfg.CROSS_VAL.ENABLED:
            with open(cfg.TEST.ANNOTATIONS, "r") as f:
                test_anns = json.load(f)
            train_anns = deepcopy(test_anns)
            image_ids = [x["id"] for x in test_anns["images"]]
            rng = random.Random(cfg.SEED)
            rng.shuffle(image_ids)
            splits = np.array_split(image_ids, cfg.CROSS_VAL.FOLDS)
            for i in range(cfg.CROSS_VAL.FOLDS):
                test_ims = set(splits[i])
                train_ims = set(image_ids) - test_ims
                test_anns_i = subset_annotations(train_anns, test_ims)
                test_results_i = subset_results(test_results, test_ims)
                if cfg.TRAIN.TRAIN_SET == "cv-train":
                    train_anns_i = subset_annotations(test_anns, train_ims)
                    train_results_i = subset_results(test_results, train_ims)
                elif cfg.TRAIN.TRAIN_SET == "cv-test":
                    train_anns_i = test_anns_i.copy()
                    train_results_i = test_results_i.copy()
                elif cfg.TRAIN.TRAIN_SET == "custom":
                    train_anns_i = cfg.TRAIN.ANNOTATIONS
                    train_results_i = cfg.TRAIN.RESULTS
                elif cfg.TRAIN.TRAIN_SET == "test":
                    train_anns_i = cfg.TEST.ANNOTATIONS
                    train_results_i = cfg.TEST.RESULTS
                elif cfg.TRAIN.TRAIN_SET == "none":
                    train_anns_i, train_results_i = None, None
                else:
                    raise NotImplementedError(
                        f"Unknown train set: {cfg.TRAIN.TRAIN_SET}"
                    )
                folds.append(
                    (train_anns_i, train_results_i, test_anns_i, test_results_i)
                )
        else:
            if cfg.TRAIN.TRAIN_SET == "custom":
                train_anns, train_res = cfg.TRAIN.ANNOTATIONS, cfg.TRAIN.RESULTS
            elif cfg.TRAIN.TRAIN_SET == "test":
                train_anns, train_res = cfg.TEST.ANNOTATIONS, cfg.TEST.RESULTS
            elif cfg.TRAIN.TRAIN_SET == "none":
                train_anns, train_res = None, None
            elif cfg.TRAIN.TRAIN_SET in ("cv-train", "cv-test"):
                raise ValueError("cv-train, cv-test not supported if cross-val is off.")
            else:
                raise NotImplementedError(f"Unknown train set: {cfg.TRAIN.TRAIN_SET}")
            folds = [(train_anns, train_res, cfg.TEST.ANNOTATIONS, test_results)]

        all_metrics, all_results = [], []

        if cfg.TRAIN.TRAIN_SET in ("custom", "test"):
            # All training folds are the same; only train a calibrator once.
            train_calibrator = self._train(cfg, folds[0][0], folds[0][1])
        else:
            train_calibrator = None
        for i, (train_ann, train_res, test_ann, test_res) in enumerate(folds):
            if len(folds) > 1:
                logger.info(f"Running calibration on fold {i}")
                vis_dir_fold = vis_dir / f"fold-{i}"
                vis_dir_fold.mkdir(exist_ok=True, parents=True)
            else:
                vis_dir_fold = vis_dir
            if train_ann is not None:
                if train_calibrator is not None:
                    calibrator = train_calibrator
                else:
                    calibrator = self._train(cfg, train_ann, train_res)
                test_res = calibrator.transform(test_res, cfg.METHOD.MAINTAIN_RANK)
            metrics = self._evaluate(cfg, test_ann, test_res, vis_dir_fold)
            all_metrics.append(metrics)
            all_results.append(test_res)
        return all_metrics, all_results

    def __call__(self, config_file, opts, output_dir):
        logging.getLogger("calibration").setLevel(logging.WARNING)
        output_dir = Path(str(output_dir))
        output_dir.mkdir(exist_ok=True, parents=True)
        logger = setup_logger(name=__file__, output=str(output_dir / "log.txt"))
        logging.root.setLevel(logging.INFO)
        logger.info(f"config={config_file}\nopts={opts}\noutput={output_dir}")
        logger.info(f"Code path: {Path('.').resolve()}")
        with open(config_file, "r") as f:
            logger.info(f"Contents of config_file={config_file}:\n{f.read()}")
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()
        with open(output_dir / "config.yaml", "w") as f:
            f.write(cfg.dump())

        vis_dir = output_dir / "vis"
        vis_dir.mkdir(exist_ok=True, parents=True)
        all_metrics, all_results = self.run_calibration(cfg, vis_dir)

        metrics = {
            k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()
        }
        logger.info(f"AP: {metrics['AP']}")
        for ece_name, ece_value in metrics.items():
            if "AP" not in ece_name:
                logger.info(f"ECE {ece_name}: {ece_value}")
        keys = [
            "AP-IoU50:95",
            "APf-IoU50:95",
            "APc-IoU50:95",
            "APr-IoU50:95",
            "AP-pooled-IoU50:95",
            "AP-pooled-f-IoU50:95",
            "AP-pooled-c-IoU50:95",
            "AP-pooled-r-IoU50:95",
        ]
        csv_msg = ",".join(
            [f"{metrics[k]*100:.2f}" if k in metrics else "" for k in keys]
            + [str(output_dir)]
        )
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
        logger.info(f"Copypaste:\n{','.join(keys)}")
        logger.info(f"Copypaste:\n{csv_msg}")
        logger.info(f"AP: %s", metrics["AP-IoU50:95"])
        logger.info(f"AP-pooled: %s", metrics["AP-pooled-IoU50:95"])
        if cfg.SAVE_RESULTS:
            if len(all_results) == 1:
                with open(output_dir / "calibrated.json", "w") as f:
                    json.dump(all_results[0], f)
            else:
                for i, results in enumerate(all_results):
                    with open(output_dir / f"calibrated_fold{i}.json", "w") as f:
                        json.dump(results, f)
        return all_metrics


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    trainer = CalibrationTrainer()
    trainer(args.config_file, args.opts, args.output_dir)


if __name__ == "__main__":
    main()
