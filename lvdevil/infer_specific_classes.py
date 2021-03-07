import copy
import datetime
import heapq
import itertools
import json
import logging
from lvdevil.infer_topk import per_class_thresholded_inference
import os
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from math import ceil
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg as get_d2_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.modeling import build_model


logger = logging.getLogger("detectron2.infer_topk")


def get_cfg():
    cfg = get_d2_cfg()
    # JSON file indicating indices of which classes to use.
    cfg.TEST.CLASSES_LIST = ""
    return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    args.eval_only = True
    default_setup(cfg, args)
    return cfg


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.

    Copied directly from detectron2's plain_train_net.py
        https://github.com/facebookresearch/detectron2/blob/1731b99173e1e2e68582cf46cf191c79ec1dea20/tools/plain_train_net.py#L56
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


@contextmanager
def suppress_classes(model, suppress_indices, num_classes, score_threshold):
    """Suppress outputs for specific classes.

    To do this, we set the score thresholds for the suppressed classes to be infinite.
    Since we reset the score thresholds, we need the default score_threshold
    (MODEL.ROI_HEADS.SCORE_THRESH_TEST) as input.

    Args:
        model: R-CNN or Cascade R-CNN model.
        suppress_indices (list): List of class indices to suppress.
        num_classes (int): Total number of classes.
        score_threshold (float): Default score threshold, as specified by
            MODEL.ROI_HEADS.SCORE_THRESH_TEST

    Yields:
        model with specified classes suppressed.
    """
    # Set score thresholds for the suppressed classes to be infinite.
    score_thresholds = torch.full((num_classes+1,), fill_value=score_threshold)
    # Background has no threshold; not sure if this line is necessary.
    score_thresholds[-1] = 0.0
    for i in suppress_indices:
        score_thresholds[i] = float("inf")
    score_thresholds = score_thresholds.to(model.device)
    with per_class_thresholded_inference(model, score_thresholds):
        yield


def main(args):
    cfg = setup(args)

    with open(cfg.TEST.CLASSES_LIST, "r") as f:
        classes_spec = json.load(f)
        # Convert 1-indexed class ids into 0-indexed model outputs
        suppress_indices = [x - 1 for x in classes_spec["skip"]]

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    results = OrderedDict()
    with suppress_classes(
        model,
        suppress_indices,
        num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
    ):
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = get_evaluator(
                cfg,
                dataset_name,
                os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
            )
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                for max_dets, max_dets_results in results_i.items():
                    logger.info(
                        f"Evaluation results for {dataset_name},max_dets={max_dets} in "
                        f"csv format:"
                    )
                    print_csv_format(max_dets_results)
    if len(results) == 1:
        results = list(results.values())[0]

    with open(Path(cfg.OUTPUT_DIR) / "metrics_infer_topk.json", "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
